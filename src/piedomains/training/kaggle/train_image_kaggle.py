#!/usr/bin/env python3
"""Train the screenshot classifier on Kaggle, free.

Paste this into a Kaggle notebook cell (Settings → Accelerator → **GPU T4 x2**, Internet →
On). It is a plain script rather than a notebook so it can be diffed and linted;
`jupytext` or a single `%run` cell turns it into one.

**Pick T4, not P100.** A P100 is sm_60 (Pascal), and current PyTorch builds ship no
kernels for it: one run spent an hour on stage 1 and then died on `no kernel image is
available for execution on the device`, leaving no checkpoint. `ensure_usable_gpu` now
checks this in the first 30 seconds and installs a compatible build if it can, but
choosing T4 avoids the problem outright.

**Why Kaggle.** 30 GPU-hours a week at no cost, 12-hour sessions, and 200 GB of dataset
storage. The job needs 3-5 hours. Colab pay-as-you-go would be $1-2 on a T4; there is no
reason to pay it.

**The disk problem, and the shape of the answer.** The corpus is 47.58 GB across 28
tarballs. `/kaggle/working` persists only 20 GB and `/kaggle/temp` gives ~60 GB that
vanishes when the session ends. So:

1. Download one tarball at a time into `/kaggle/temp`.
2. Stream it, resize every screenshot to 224px, write the small copy to
   `/kaggle/temp/images-224`.
3. Delete the tarball before fetching the next.

Peak disk is one tarball (~1.7 GB) plus the growing 224px set (~1 GB at the current
cap), not 47.58 GB. Every archive is still *transferred* -- a .tar.gz cannot be seeked
and the domains we want are spread across all 28 -- but none are kept.

**Everything intermediate stays out of `/kaggle/working`.** That directory is the kernel
*output*, and retrieving any part of it means enumerating all of it. With the 51,138
resized JPEGs there, pulling the 350 MB model meant listing 51k files, which exhausted
the API's rate limit and left a finished run's model unreachable for hours. Only
`image-v1` is written there now -- about five files. The cost is that stage 1 cannot be
published as a reusable Dataset and is repeated per session, which is ~1 hour against a
12-hour cap.

**Resume across sessions.** `train_image.py --resume` picks up from the checkpoint in
`--out`, and it only ever writes the best epoch, so an interrupted run costs at most one
epoch rather than the lot.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

REPO = "https://github.com/themains/piedomains.git"

#: Branch to clone. The image scripts live here until the branch is merged; cloning the
#: default branch is what made the first run download 47.58 GB and then fail on a missing
#: file.
BRANCH = "taxonomy-round-2"

#: Kaggle's persistent output volume, 20 GB, and the kernel's *output* -- everything here
#: has to be enumerated file-by-file to retrieve any of it. Only the checkpoint lives
#: here: ~5 files.
WORK = Path("/kaggle/working")

#: Ephemeral scratch, ~60 GB, wiped when the session ends. Tarballs land here and are
#: deleted as soon as they have been streamed, and the 51,138 resized JPEGs stay here too.
#:
#: They used to go to WORK, which made the output 51k files. Retrieving the 350 MB model
#: then meant listing all of them, which exhausted the ListKernelSessionOutput rate limit
#: and left a completed run's model unreachable for hours. Stage 1 costs ~1 hour to
#: repeat; that is cheaper than not being able to get the model out at all.
TEMP = Path("/kaggle/temp")

#: Point this at an attached Kaggle Dataset of already-resized images to skip the download
#: and resize entirely. Stage 1 no longer produces one itself -- see TEMP above -- so this
#: is for a set uploaded deliberately, not a leftover from a previous run.
PREPARED_DATASET: str | None = None  # e.g. "/kaggle/input/piedomains-screenshots-224"

IMAGE_SIZE = 224

#: Five, with patience=2 and best-epoch-only checkpointing doing the real work. Raising
#: this from 3 to 8 did not help: that run peaked at epoch 4 with val macro-F1 0.3705 and
#: early-stopped, below the 3-epoch run's 0.3953. The two were not comparable anyway --
#: augmentation used an unseeded random.random(), now fixed -- so the honest reading is
#: that the peak is around 3-5 and the rest was run-to-run noise.
EPOCHS = 5
BATCH_SIZE = 32

#: Cap per class. The corpus is 78% four classes (adult 55,184, shopping 48,384,
#: travel 46,353, sports 43,070 of 248,003), so this both balances
#: it and brings the run inside the 12-hour session cap: 51,138 images rather than
#: 248,003, which is comparable to the 46,754 documents the text model trains on.
MAX_PER_CLASS = 3000

#: Pretrained vision encoder. ImageNet-21k pretraining optimises for object recognition,
#: and a webpage screenshot has almost no object content -- its signal is text density,
#: layout grid and colour. CLIP-family encoders transfer better to non-object domains.
#:
#: That was the hypothesis; it was then measured on identical data, GPU and step count.
#: SigLIP2 scored **0.531 accuracy / 0.397 macro-F1** against ViT's **0.335 / 0.140** --
#: ViT started at ln(42) = 3.74, which is chance, and barely moved. This constant stayed
#: on the losing baseline long after the comparison was settled, so a rerun would have
#: quietly trained the worse encoder.
#:
#: SigLIP2 also decides the preprocessing: it pretrained with a non-aspect-preserving
#: resize, which is why `images.resize_for_model` squashes rather than crops.
BACKBONE = "google/siglip2-base-patch16-224"

#: Published text model, pulled from the Hub for stage 4. Public, so no token is needed.
TEXT_MODEL = "gojiberries/piedomains-text"
TEXT_MODEL_REVISION = "38d7e6403f911902f47b9218ce5c645a06dd02fe"

#: Set this to a Hub repo to skip training and calibration and fuse an already-trained
#: checkpoint instead. Fusion cannot be done locally -- the resized screenshots live in
#: TEMP and die with the session -- so a fusion that fails for an avoidable reason
#: otherwise costs a second full GPU run. The first one did: it pulled TEXT_MODEL from the
#: Hub while a new text model was still uploading, compared the new flat class names
#: against the old prefixed ones, and refused.
IMAGE_MODEL: str | None = None  # e.g. "gojiberries/piedomains-image"
IMAGE_MODEL_REVISION = "e751348e3ca57b24cb299db7c4ce87a924a91c21"

#: Name of the attached Dataset holding the current text splits. The image model must be
#: aligned to *these*, not to an earlier version: re-preparing the text corpus reshuffled
#: the assignments, and 73% of the new test domains landed in the old training set, so a
#: model aligned to the old splits scored 0.706 on data it had trained on.
SPLITS_DATASET = "piedomains-text-v13"


def find_dataset(name: str) -> Path:
    """Locate an attached Dataset wherever Kaggle mounted it.

    Args:
        name: The Dataset's name.

    Returns:
        Path: Its directory.

    Raises:
        SystemExit: If it is not attached, listing what is.
    """
    root = Path("/kaggle/input")
    if root.exists():
        for candidate in sorted(root.rglob(name)):
            if candidate.is_dir():
                print(f"found {name} at {candidate}")
                return candidate
    listing = sorted(str(p) for p in root.rglob("*"))[:25] if root.exists() else []
    raise SystemExit(f"Dataset {name!r} not attached.\n/kaggle/input holds: {listing}")


def run(cmd: list[str], **kwargs) -> None:
    """Run a command, streaming output, and fail loudly.

    Args:
        cmd: Command and arguments.
        **kwargs: Passed through to ``subprocess.run``.

    Raises:
        subprocess.CalledProcessError: If the command exits non-zero. Deliberately not
            ``SystemExit``: stage 1 catches per-tarball failures so one bad download does
            not lose the run, and ``except SystemExit`` neither reads as intent nor
            survives someone narrowing it. Uncaught, this still aborts the job.
    """
    print("$", " ".join(cmd), flush=True)
    # Every argument is constructed in this file from module constants; nothing here
    # comes from user input.
    result = subprocess.run(cmd, **kwargs)  # noqa: S603
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def setup() -> Path:
    """Clone the repository and install what training needs.

    Returns:
        Path: The repository root.

    Raises:
        SystemExit: If the cloned branch is missing a script the run needs.
    """
    repo = TEMP / "piedomains"
    if not repo.exists():
        run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, str(repo)])

    # Check every script exists before anything expensive happens. The first run spent
    # 33 minutes downloading 47.58 GB and then died because prepare_images.py was not on
    # the branch being cloned. That should be impossible, not merely unlucky.
    required = [
        "download_corpus.py",
        "prepare_images.py",
        "prepare_text.py",
        "train_image.py",
        "calibrate.py",
        "taxonomy.py",
    ]
    training = repo / "src" / "piedomains" / "training"
    missing = [n for n in required if not (training / n).exists()]
    if missing:
        raise SystemExit(
            f"branch {BRANCH!r} is missing {missing}; nothing downloaded. "
            "Push the branch before running this."
        )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers>=4.48",
            "torchvision",
            "pillow",
            "requests",
            "tqdm",
        ]
    )

    # The training scripts are modules inside the package now, run as
    # `-m piedomains.training.x`, so the clone has to be importable. PYTHONPATH rather
    # than `pip install -e` because the clone is scratch: it exists to be read, and
    # installing it would put a second, stale piedomains ahead of anything else.
    src = repo / "src"
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)
    print(f"PYTHONPATH={os.environ['PYTHONPATH']}")
    return repo


#: Last torch whose cu121 wheels still ship Pascal (sm_60) kernels, for when Kaggle hands
#: out a P100. Kaggle's own preinstalled build is cu128 and starts at sm_70.
PASCAL_TORCH = "2.6.0"
PASCAL_TORCHVISION = "0.21.0"

#: cu124, not cu121: the cu121 index stops at torch 2.5.1, which transformers
#: rejects. Three constraints have to hold at once -- transformers needs >= 2.6,
#: the P100 needs sm_60 kernels, and the wheel has to exist on the index.
PASCAL_INDEX = "https://download.pytorch.org/whl/cu124"

#: Probe run in a subprocess, because a GPU verdict cannot be revised inside a process
#: that has already initialised CUDA. Exit 42 means "device present but this torch has no
#: kernels for it" -- the failure that wasted a whole run.
GPU_PROBE = """
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    print("no CUDA device")
    sys.exit(1)
major, minor = torch.cuda.get_device_capability()
sm = f"sm_{major}{minor}"
arches = torch.cuda.get_arch_list()
print("device", torch.cuda.get_device_name(0), sm)
print("this torch builds for", arches)
if sm not in arches:
    print(f"INCOMPATIBLE: {sm} is not in the build")
    sys.exit(42)
# Arch lists can still lie; a real kernel launch is the only proof.
torch.zeros(8, device="cuda").add_(1).sum().item()
print("OK: a real kernel launched on", sm)
"""


def ensure_usable_gpu() -> None:
    """Prove the GPU can actually run a kernel, before anything expensive.

    Kaggle assigns either a T4 (sm_75) or a P100 (sm_60), and current PyTorch builds have
    dropped Pascal. A run got a P100, spent an hour on stage 1, then died instantly on
    ``no kernel image is available for execution on the device`` -- with no checkpoint and
    the whole session wasted. Checking first costs 30 seconds.

    A capability mismatch is repaired by installing a build that covers the device rather
    than giving up, since which GPU Kaggle hands out is not ours to choose.

    Raises:
        SystemExit: If no GPU is present, or if it is still unusable after reinstalling.
            Fatal on purpose: CPU training on 51,138 images would not finish inside the
            session cap, so continuing would burn the run to no end.
    """
    probe = TEMP / "gpu_probe.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(GPU_PROBE, encoding="utf-8")

    first = subprocess.run(  # noqa: S603 -- interpreter plus a file we just wrote
        [sys.executable, str(probe)], capture_output=True, text=True, check=False
    )
    print(first.stdout.strip() or first.stderr.strip()[:600])
    if first.returncode == 0:
        return
    if first.returncode != 42:
        raise SystemExit("no usable GPU; set the accelerator on this notebook to GPU")

    print(f"\ninstalling torch {PASCAL_TORCH} for this device's compute capability...")
    # Pinned and forced. An unpinned `pip install --index-url ... torch` is a no-op here:
    # pip sees the preinstalled 2.10.0+cu128 as already satisfying `torch` and changes
    # nothing, so the second probe reported the identical incompatible build.
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--force-reinstall",
            "--no-cache-dir",
            "--index-url",
            PASCAL_INDEX,
            f"torch=={PASCAL_TORCH}",
            f"torchvision=={PASCAL_TORCHVISION}",
        ]
    )
    second = subprocess.run(  # noqa: S603 -- interpreter plus a file we just wrote
        [sys.executable, str(probe)], capture_output=True, text=True, check=False
    )
    print(second.stdout.strip() or second.stderr.strip()[:600])
    if second.returncode != 0:
        raise SystemExit(
            "GPU still unusable after reinstalling torch. Switch this notebook's "
            "accelerator to 'GPU T4 x2' (sm_75) and re-run; P100 is sm_60, which "
            "current torch builds no longer ship kernels for."
        )


def stage_one_prepare() -> Path:
    """Download, stream, resize and discard — one tarball at a time.

    Peak disk is a single tarball plus the growing resized set, so this fits in
    Kaggle's volumes where the full corpus would not.

    Returns:
        Path: Directory holding the resized dataset.

    Raises:
        SystemExit: If the tarballs cannot be enumerated, or every one of them fails.
    """
    out = TEMP / "images-224"
    if (out / "labels.json").exists():
        print(f"{out} already built; skipping stage 1")
        return out

    corpus = TEMP / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)

    # Labels first: small, and prepare_images.py needs the per-category domain lists.
    labels_dir = TEMP / "labels"
    if not (labels_dir / "shallalist_cats.txt").exists():
        run(
            [
                sys.executable,
                "-m",
                "piedomains.training.download_corpus",
                "--set",
                "labels",
                "--out",
                str(labels_dir),
            ]
        )

    # The domain lists are fetched by prepare_text.py's mirror logic on first use; a tiny
    # prepare_text run populates the cache without processing the text corpus.
    cache = TEMP / "labels" / "domains"
    if not cache.exists() or not any(cache.glob("*.txt")):
        print("populating the label cache from the Shallalist mirror...")
        run(
            [
                sys.executable,
                "-c",
                "from piedomains.training.prepare_text import load_category_map;"
                f"m = load_category_map(__import__('pathlib').Path({str(labels_dir / 'shallalist_cats.txt')!r}),"
                f" __import__('pathlib').Path({str(cache)!r}));"
                "print(f'{len(m):,} labelled domains')",
            ]
        )

    # One tarball at a time: fetch, stream it into the resized set, delete. Peak disk is
    # a single archive plus the growing 224px output rather than all 47.58 GB. A .tar.gz
    # cannot be seeked, so every archive must still be transferred -- the domains we want
    # are scattered across all 28, rare classes especially -- but nothing requires
    # keeping them.
    # --names prints one filename per line and nothing else. --list prints per-set
    # summaries ("screenshots  28 files  47.58 GB") and never a filename, which is what
    # the previous version parsed and why it enumerated nothing twice.
    listing = subprocess.run(
        [
            sys.executable,
            "-m",
            "piedomains.training.download_corpus",
            "--set",
            "screenshots",
            "--names",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    tarballs = [n for n in listing.stdout.split() if n.endswith(".tar.gz")]
    if not tarballs:
        raise SystemExit(
            "could not enumerate screenshot tarballs.\n"
            f"stdout: {listing.stdout[:400]!r}\n"
            f"stderr: {listing.stderr[:400]!r}"
        )
    print(f"{len(tarballs)} tarballs to stream")

    # A tarball that will not download is survivable; losing the run is not. The SigLIP
    # attempt died 20 minutes in because Dataverse answered one request with a 502 and the
    # whole job went with it. download_corpus.py now retries transient failures, and a
    # tarball that still fails after that is skipped so the other 27 can train.
    skipped = []
    for i, name in enumerate(tarballs, 1):
        print(f"\n[{i}/{len(tarballs)}] {name}", flush=True)
        try:
            run(
                [
                    sys.executable,
                    "-m",
                    "piedomains.training.download_corpus",
                    "--set",
                    "screenshots",
                    "--only",
                    name,
                    "--out",
                    str(corpus),
                ]
            )
            run(
                [
                    sys.executable,
                    "-m",
                    "piedomains.training.prepare_images",
                    "--corpus",
                    str(corpus),
                    "--out",
                    str(out),
                    "--label-cache",
                    str(cache),
                    "--size",
                    str(IMAGE_SIZE),
                    "--max-per-class",
                    str(MAX_PER_CLASS),
                    "--append",
                    "--index",
                    str(labels_dir / "screenshot-index.tab"),
                ]
            )
        except subprocess.CalledProcessError as exc:
            # Named, counted and reported at the end. A silently dropped tarball would
            # read as full coverage while quietly shrinking the training set.
            print(f"  SKIPPING {name}: {exc}", flush=True)
            skipped.append(name)
        finally:
            for spent in corpus.glob("*.tar.gz"):
                spent.unlink()

    if skipped:
        print(
            f"\n{len(skipped)}/{len(tarballs)} tarballs skipped: {', '.join(skipped)}"
        )
    if len(skipped) == len(tarballs):
        raise SystemExit("every tarball failed; nothing to train on")
    print(f"stage 1 complete: {out} ({len(tarballs) - len(skipped)} tarballs streamed)")
    return out


def stage_two_train(data: Path) -> Path:
    """Fine-tune the backbone, resuming if a checkpoint is present.

    Args:
        data: Directory holding the resized dataset.

    Returns:
        Path: Directory holding the trained model.
    """
    out = WORK / "image-v1"
    cmd = [
        sys.executable,
        "-m",
        "piedomains.training.train_image",
        "--data",
        str(data),
        "--out",
        str(out),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--model",
        BACKBONE,
        "--workers",
        "2",
    ]
    if (out / "state.json").exists():
        cmd.append("--resume")
    run(cmd)
    return out


def stage_three_calibrate(data: Path, model: Path) -> None:
    """Fit the temperature, without which fusion is meaningless.

    Averaging an uncalibrated image softmax against a calibrated text distribution is
    exactly the bug the original package shipped. Both sides must be on the same scale
    before they can be combined.

    Args:
        data: Directory holding the resized dataset.
        model: Directory holding the trained model.
    """
    run(
        [
            sys.executable,
            "-m",
            "piedomains.training.calibrate",
            "--model",
            str(model),
            "--data",
            str(data),
            "--modality",
            "image",
        ]
    )


def stage_four_fuse(data: Path, model: Path) -> None:
    """Fit and score late fusion, in the session where the screenshots still exist.

    This has to run here rather than locally. The resized images live in ``/kaggle/temp``
    so that the kernel output stays retrievable, which means they are gone the moment the
    session ends -- and ``fuse.py`` needs them to score every paired domain. Fusing after
    the fact would mean either re-downloading 47.58 GB or capturing tens of thousands of
    screenshots first.

    The text side comes from the Hub, and the paired text splits from an attached Dataset;
    only ``fusion.json`` and the report come back, both small.

    Skipped rather than fatal if either input is missing: the trained model is already
    saved by this point, and losing it to a fusion problem would be the worse outcome.

    Args:
        data: Directory of resized screenshots.
        model: The trained image checkpoint.
    """
    splits = find_dataset(SPLITS_DATASET)
    if not splits.exists():
        print(
            f"\nno {splits}; skipping fusion. Attach the "
            f"{SPLITS_DATASET!r} Dataset to fuse in-session."
        )
        return

    print(f"\ndownloading the text model from {TEXT_MODEL}...")
    from huggingface_hub import snapshot_download

    text_dir = snapshot_download(repo_id=TEXT_MODEL, revision=TEXT_MODEL_REVISION)

    try:
        run(
            [
                sys.executable,
                "-m",
                "piedomains.training.fuse",
                "--text",
                str(text_dir),
                "--image",
                str(model),
                "--text-data",
                str(splits),
                "--image-data",
                str(data),
                "--out",
                str(WORK / "fusion_report.json"),
            ]
        )
    except subprocess.CalledProcessError as exc:
        # fuse.py exits non-zero *by design* when fusion does not beat text alone. That is
        # a result, not a crash, and the report is already written.
        print(f"fuse.py exited {exc.returncode} -- see fusion_report.json for why")


def export_fusion_corpus(data: Path) -> None:
    """Archive the held-out screenshots so fusion can be refitted locally.

    Stage 4 exists on Kaggle only because the images die with the session, which makes
    every fusion experiment cost a fresh corpus build. The val and test splits are ~20% of
    the images and are the only ones fusion needs, so they fit in the output volume.

    Written as **one tar.gz rather than loose files** deliberately: putting tens of
    thousands of files in WORK is what previously exhausted the output-listing rate limit
    and left a finished run's model unreachable.

    Args:
        data: Directory of resized screenshots and split files.
    """
    import tarfile

    archive = WORK / "fusion-corpus.tar.gz"
    kept = 0
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("val.jsonl", "test.jsonl", "labels.json"):
            member = data / name
            if member.exists():
                tar.add(member, arcname=name)
        wanted: set[str] = set()
        for split in ("val", "test"):
            path = data / f"{split}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        wanted.add(json.loads(line)["domain"])
        for domain in sorted(wanted):
            shot = data / "images" / f"{domain}.jpg"
            if shot.exists():
                tar.add(shot, arcname=f"images/{domain}.jpg")
                kept += 1
    size = archive.stat().st_size / 1e6
    print(f"\nwrote {archive} -- {kept:,} held-out screenshots, {size:.0f} MB")
    print("  fusion can be refitted locally from this; no session needed")


def main() -> int:
    """Run the whole pipeline.

    Returns:
        int: Process exit status.
    """
    if not os.path.exists("/kaggle"):
        print("This script expects a Kaggle notebook environment.", file=sys.stderr)
        return 1

    setup()
    # Before stage 1, not after: an unusable GPU discovered later costs the whole session.
    ensure_usable_gpu()
    data = Path(PREPARED_DATASET) if PREPARED_DATASET else stage_one_prepare()
    if IMAGE_MODEL:
        from huggingface_hub import snapshot_download

        print(f"\nusing the published checkpoint {IMAGE_MODEL}; not training")
        model = Path(
            snapshot_download(repo_id=IMAGE_MODEL, revision=IMAGE_MODEL_REVISION)
        )
    else:
        model = stage_two_train(data)
        stage_three_calibrate(data, model)
    export_fusion_corpus(data)
    stage_four_fuse(data, model)

    print("\nDownload these from the notebook output:")
    print(f"  {model}/  — weights, labels.json, calibration.json, test_metrics.json")
    print("  fusion-corpus.tar.gz — the held-out screenshots, to refit fusion locally")
    print("\nFusion must beat text alone (0.818 accuracy / 0.758 macro-F1)")
    print("or screenshot classification stays opt-in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
