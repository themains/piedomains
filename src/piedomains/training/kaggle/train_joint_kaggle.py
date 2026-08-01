#!/usr/bin/env python3
r"""Measure every way of combining text and screenshots, in one Kaggle session.

Settings → Accelerator → **GPU T4 x2**, Internet → On, and attach three Datasets:
`piedomains-text-v5` (the text tower), `piedomains-text-minimal` (the paired splits) and
nothing else -- the screenshots are rebuilt here.

**Why one kernel and not three.** The resized screenshot corpus lives in `/kaggle/temp` so
that the kernel *output* stays small enough to retrieve; that means it is destroyed when
the session ends. Every step that needs paired images therefore has to run in the same
session that builds them. Locally there are 183 screenshots, against the 13,633 paired
training domains this needs.

**What gets compared**, all on the same 1,742 paired test domains:

===================  =========================================================
text only            the baseline everything is measured against
weighted per-class   fitted per-class mixing of the two probability vectors
stacked (CV)         a meta-classifier over both vectors, cross-validated
joint two-tower      one head over both *representations*
===================  =========================================================

The first three come from ``fuse.py``; the last from ``train_joint.py``.

**The bar was declared before any of this ran.** At n=1,742 the standard error on accuracy
is about 0.011, so a difference under ~0.02 is not evidence. Weighted fusion already
returned +0.001, which is exactly the kind of number that looks like a win and is not.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/themains/piedomains.git"
BRANCH = "text-representation"

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")

#: Attached Datasets, found by search: mount paths are not one convention. The splits
#: appeared under /kaggle/input/datasets/<owner>/<name> while an earlier Dataset appeared
#: at /kaggle/input/<name>, which cost two runs before a listing was printed.
TEXT_MODEL_DATASET = "piedomains-text-v5"
SPLITS_DATASET = "piedomains-text-minimal"

IMAGE_MODEL = "soodoku/piedomains-image"

IMAGE_SIZE = 224
MAX_PER_CLASS = 3000

#: Frozen towers. 13,633 paired examples against ~200M parameters overfits readily, and a
#: head over frozen encoders trains in minutes rather than hours.
JOINT_EPOCHS = 6
JOINT_BATCH = 16

PASCAL_TORCH = "2.6.0"
PASCAL_TORCHVISION = "0.21.0"
PASCAL_INDEX = "https://download.pytorch.org/whl/cu124"

GPU_PROBE = """
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    print("no CUDA device"); sys.exit(1)
major, minor = torch.cuda.get_device_capability()
sm = f"sm_{major}{minor}"
print("device", torch.cuda.get_device_name(0), sm)
print("this torch builds for", torch.cuda.get_arch_list())
if sm not in torch.cuda.get_arch_list():
    print(f"INCOMPATIBLE: {sm} is not in the build"); sys.exit(42)
torch.zeros(8, device="cuda").add_(1).sum().item()
print("OK: a real kernel launched on", sm)
"""


def run(cmd: list[str]) -> None:
    """Run a command, streaming output.

    Args:
        cmd: Command and arguments.

    Raises:
        subprocess.CalledProcessError: If the command exits non-zero.
    """
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)  # noqa: S603
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


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


def setup() -> tuple[Path, Path, Path]:
    """Clone the branch, install dependencies, and resolve both Datasets.

    Returns:
        tuple[Path, Path, Path]: Repository root, text-model dir, splits dir.

    Raises:
        SystemExit: If a required script is missing from the branch.
    """
    text_model = find_dataset(TEXT_MODEL_DATASET)
    splits = find_dataset(SPLITS_DATASET)

    repo = TEMP / "piedomains"
    if not repo.exists():
        run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, str(repo)])

    training = repo / "src" / "piedomains" / "training"
    required = ("prepare_images.py", "fuse.py", "train_joint.py", "download_corpus.py")
    missing = [n for n in required if not (training / n).exists()]
    if missing:
        raise SystemExit(f"branch {BRANCH!r} is missing {missing}; push it first")

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
            "scikit-learn>=1.3",
        ]
    )

    src = repo / "src"
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)
    return repo, text_model, splits


def ensure_usable_gpu() -> None:
    """Prove the GPU can run a kernel before an hour of data preparation.

    Raises:
        SystemExit: If no GPU, or still unusable after installing a compatible build.
    """
    probe = TEMP / "gpu_probe.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(GPU_PROBE, encoding="utf-8")

    first = subprocess.run(  # noqa: S603
        [sys.executable, str(probe)], capture_output=True, text=True, check=False
    )
    print(first.stdout.strip() or first.stderr.strip()[:600])
    if first.returncode == 0:
        return
    if first.returncode != 42:
        raise SystemExit("no usable GPU; set this notebook's accelerator to GPU")

    print(f"\ninstalling torch {PASCAL_TORCH} for this device...")
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
    second = subprocess.run(  # noqa: S603
        [sys.executable, str(probe)], capture_output=True, text=True, check=False
    )
    print(second.stdout.strip() or second.stderr.strip()[:600])
    if second.returncode != 0:
        raise SystemExit("GPU still unusable; switch the accelerator to GPU T4 x2")


def build_images(splits: Path) -> Path:
    """Stream the screenshot corpus and resize it, aligned to the text splits.

    Splits come from :func:`piedomains.training.splits.split_of`, so the screenshots and
    the text corpus cannot disagree about where a domain belongs. That used to be a flag
    you had to remember, and forgetting it put the image model's training set inside the
    domains fusion scored -- reported as 0.768 against an honest 0.429.

    Args:
        splits: Directory of prepared text splits.

    Returns:
        Path: Directory of resized screenshots.

    Raises:
        SystemExit: If every tarball fails.
    """
    out = TEMP / "images-224"
    if (out / "labels.json").exists():
        print(f"{out} already built")
        return out

    corpus = TEMP / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
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

    cache = labels_dir / "domains"
    if not cache.exists() or not any(cache.glob("*.txt")):
        run(
            [
                sys.executable,
                "-c",
                "from piedomains.training.prepare_text import load_category_map;"
                f"import pathlib; m = load_category_map(pathlib.Path({str(labels_dir / 'shallalist_cats.txt')!r}),"
                f" pathlib.Path({str(cache)!r})); print(f'{{len(m):,}} labelled domains')",
            ]
        )

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
        raise SystemExit(f"could not enumerate tarballs: {listing.stderr[:300]!r}")
    print(f"{len(tarballs)} tarballs to stream")

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
            print(f"  SKIPPING {name}: {exc}", flush=True)
            skipped.append(name)
        finally:
            for spent in corpus.glob("*.tar.gz"):
                spent.unlink()

    if skipped:
        print(f"\n{len(skipped)}/{len(tarballs)} skipped: {', '.join(skipped)}")
    if len(skipped) == len(tarballs):
        raise SystemExit("every tarball failed; nothing to train on")
    return out


def main() -> int:
    """Build images, then measure fusion and the joint model.

    Returns:
        int: Zero if the run completed; the comparison's verdict is in the reports.
    """
    if not os.path.exists("/kaggle"):
        print("This script expects a Kaggle notebook environment.", file=sys.stderr)
        return 1

    _, text_model, splits = setup()
    ensure_usable_gpu()

    from huggingface_hub import snapshot_download

    image_model = snapshot_download(repo_id=IMAGE_MODEL)
    print(f"image model at {image_model}")

    images = build_images(splits)

    # fuse.py reports text-only, image-only, weighted and STACKED side by side. A
    # non-zero exit means fusion did not beat text alone, which is a result rather than
    # a failure -- the report is already written either way.
    print("\n" + "=" * 70 + "\nFUSION: weighted and stacked\n" + "=" * 70)
    try:
        run(
            [
                sys.executable,
                "-m",
                "piedomains.training.fuse",
                "--text",
                str(text_model),
                "--image",
                image_model,
                "--text-data",
                str(splits),
                "--image-data",
                str(images),
                "--out",
                str(WORK / "fusion_report.json"),
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(f"fuse.py exited {exc.returncode} -- see fusion_report.json")

    print("\n" + "=" * 70 + "\nTRUE ENSEMBLE: joint two-tower\n" + "=" * 70)
    try:
        run(
            [
                sys.executable,
                "-m",
                "piedomains.training.train_joint",
                "--text",
                str(text_model),
                "--image",
                image_model,
                "--data",
                str(splits),
                "--images",
                str(images),
                "--out",
                str(WORK / "joint-v1"),
                "--epochs",
                str(JOINT_EPOCHS),
                "--batch-size",
                str(JOINT_BATCH),
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(f"train_joint exited {exc.returncode} -- did not clear the band")

    print("\nDownload fusion_report.json and joint-v1/ from the notebook output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
