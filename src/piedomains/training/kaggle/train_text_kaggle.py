#!/usr/bin/env python3
"""Retrain the text classifier on Kaggle, free.

Settings → Accelerator → **GPU T4 x2**, Internet → On, and attach the
`piedomains-text-v13` Dataset.

**Why Kaggle rather than a laptop.** The last local run took about five hours on an M4
and thrashed swap, and this one doubles `max_length` to 256. Kaggle gives 30 GPU-hours a
week at no cost.

**Pick T4, not P100.** A P100 is sm_60 and current PyTorch builds ship no kernels for it,
so a run dies on `no kernel image is available for execution on the device`. Worse for
this model specifically: ModernBERT compiles its encoder through TorchInductor, whose
Triton backend requires compute capability >= 7.0. `ensure_usable_gpu` checks in the first
30 seconds and installs a compatible build if it can, but choosing T4 avoids it outright.

**What changed in the data, and why a retrain is needed at all.** The cleaner used to
deduplicate tokens twice, sort them alphabetically and strip every non-ASCII character, so
the model was trained on an alphabetised set of ASCII words with no term frequency and no
order. Fixing that changes the input distribution completely, so the weights have to be
refitted. See `text_processor.clean_and_normalize_text`.

The prepared Dataset also carries two corrections the old corpus did not have: anti-bot
interstitials are refused, and domain-parking placeholders are labelled `parked` rather
than left contaminating the class of whatever the domain used to sell — which was 42% of
`drugs`.

**The label space is 43, not 47.** Three splits that asked about delivery mechanism or
legality rather than subject are collapsed — `games-misc`/`-online`,
`radiotv`/`webradio`, `downloads`/`warez` — and `aggressive` is dropped for want of data
at 9 held-out documents and recall 0.111. Relabelling the 47-class model's own predictions
puts the floor at **+0.023 accuracy and +0.023 macro-F1**; this run measures what a model
that never had to learn the impossible distinction does instead. See
`training/taxonomy.py` and `training/confusions.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/themains/piedomains.git"

#: Branch carrying the 43-class taxonomy.
BRANCH = "taxonomy-round-2"

#: Kernel output. Only the checkpoint goes here: retrieving any file means enumerating
#: every file, and a run that writes thousands of them becomes unreachable through the API.
WORK = Path("/kaggle/working")

#: Ephemeral scratch, wiped at session end.
TEMP = Path("/kaggle/temp")

#: Name of the attached Dataset holding the prepared splits, so this kernel never touches
#: the 17GB corpus. Built by `prepare_text.py` with the minimal cleaner.
#:
#: The 44-class corpus, split by `training.splits.split_of` so the text and screenshot
#: corpora cannot disagree about where a domain belongs. Earlier Datasets are kept but
#: must not be trained on: `piedomains-text-minimal` is the superseded 47-class label
#: space, and `piedomains-text-clean` had standalone punctuation stripped -- an idea
#: that looked obviously right and measured worse (Curlie 0.543 -> 0.523).
DATASET = "piedomains-text-v13"

#: 256, not 128. Under the old cleaner a page collapsed to a few hundred unique
#: alphabetised words, so 128 subword tokens rarely truncated anything. Real text with
#: order and frequency needs the room, and trafilatura's median page is 251 words.
MAX_LENGTH = 256

#: Both previous text runs peaked at epoch 3. Patience and best-epoch-only checkpointing
#: do the real work, so this is headroom rather than a target.
EPOCHS = 4
BATCH_SIZE = 32

#: Multilingual ModernBERT. The corpus is overwhelmingly English, but the cleaner no
#: longer strips non-Latin scripts, so this can finally use what it was chosen for.
BACKBONE = "jhu-clsp/mmBERT-base"

GPU_PROBE = """
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    print("no CUDA device")
    sys.exit(1)
major, minor = torch.cuda.get_device_capability()
sm = f"sm_{major}{minor}"
print("device", torch.cuda.get_device_name(0), sm)
print("this torch builds for", torch.cuda.get_arch_list())
if sm not in torch.cuda.get_arch_list():
    print(f"INCOMPATIBLE: {sm} is not in the build")
    sys.exit(42)
torch.zeros(8, device="cuda").add_(1).sum().item()
print("OK: a real kernel launched on", sm)
"""

#: The version has to satisfy two constraints at once, which is a narrow window:
#: transformers refuses torch < 2.6 outright ("we now require users to upgrade torch to
#: at least v2.6"), while Kaggle's P100 is sm_60 and current builds start at sm_70. 2.6.0
#: is the oldest release that clears the first. `ensure_usable_gpu` verifies the second at
#: runtime rather than trusting this comment, so a wrong guess costs 30 seconds.
PASCAL_TORCH = "2.6.0"

#: Pinned with torch, not left behind. torchvision registers C++ operators against a
#: specific torch build, so downgrading one alone gives
#: "operator torchvision::nms does not exist" -- and transformers imports torchvision, so
#: even a text-only model then fails with
#: "Could not import module 'ModernBertForSequenceClassification'".
PASCAL_TORCHVISION = "0.21.0"

#: cu124, not cu121: the cu121 index stops at torch 2.5.1, which transformers
#: rejects. Three constraints have to hold at once -- transformers needs >= 2.6,
#: the P100 needs sm_60 kernels, and the wheel has to exist on the index.
PASCAL_INDEX = "https://download.pytorch.org/whl/cu124"


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


def setup() -> Path:
    """Clone the branch, install dependencies, and make the package importable.

    Returns:
        Path: The directory holding the prepared splits.

    Raises:
        SystemExit: If a required script is missing.
    """
    data = find_dataset()

    repo = TEMP / "piedomains"
    if not repo.exists():
        run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, str(repo)])

    training = repo / "src" / "piedomains" / "training"
    missing = [
        n for n in ("train_text.py", "calibrate.py") if not (training / n).exists()
    ]
    if missing:
        raise SystemExit(f"branch {BRANCH!r} is missing {missing}; push it first")

    run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.48", "tqdm"])

    src = repo / "src"
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)
    print(f"PYTHONPATH={os.environ['PYTHONPATH']}")
    return data


def find_dataset() -> Path:
    """Locate the attached Dataset wherever Kaggle chose to mount it.

    Mount paths are not one convention. `piedomains-fusion-splits` appeared at
    ``/kaggle/input/<name>`` while `piedomains-text-minimal` appeared at
    ``/kaggle/input/datasets/<owner>/<name>`` -- which cost two runs before a listing was
    printed instead of a guess. So search, and if nothing matches, say what *is* there.

    Returns:
        Path: Directory containing train/val/test.jsonl.

    Raises:
        SystemExit: If the Dataset is not attached, listing the actual mount.
    """
    root = Path("/kaggle/input")
    if root.exists():
        for candidate in sorted(root.rglob(DATASET)):
            if (candidate / "train.jsonl").exists():
                print(f"found the prepared splits at {candidate}")
                return candidate

    listing = sorted(str(p) for p in root.rglob("*"))[:25] if root.exists() else []
    raise SystemExit(
        f"Dataset {DATASET!r} is not attached.\n"
        f"/kaggle/input holds: {listing}\n"
        f"Add the {DATASET!r} Dataset to this notebook; this kernel does not rebuild "
        "it from the 17GB corpus."
    )


def ensure_usable_gpu() -> None:
    """Prove the GPU can run a kernel before anything expensive starts.

    Raises:
        SystemExit: If no GPU is present, or it is still unusable after reinstalling.
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
        raise SystemExit(
            "GPU still unusable. Switch the accelerator to 'GPU T4 x2' (sm_75); P100 is "
            "sm_60, which current torch builds no longer ship kernels for."
        )


def main() -> int:
    """Train and calibrate the text model.

    Returns:
        int: Process exit status.
    """
    if not os.path.exists("/kaggle"):
        print("This script expects a Kaggle notebook environment.", file=sys.stderr)
        return 1

    data = setup()
    ensure_usable_gpu()

    out = WORK / "text-v5"
    run(
        [
            sys.executable,
            "-m",
            "piedomains.training.train_text",
            "--data",
            str(data),
            "--out",
            str(out),
            "--epochs",
            str(EPOCHS),
            "--batch-size",
            str(BATCH_SIZE),
            "--max-length",
            str(MAX_LENGTH),
            "--model",
            BACKBONE,
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "piedomains.training.calibrate",
            "--model",
            str(out),
            "--data",
            str(data),
        ]
    )

    print("\nDownload text-v5/ from the notebook output, then locally:")
    print("  PIEDOMAINS_TEXT_MODEL=models/text-v5 \\")
    print("    uv run python -m piedomains.training.validate_curlie \\")
    print("      --paired data/curlie/paired.json --limit 210")
    print("  The number to beat is 0.519 agreement on Curlie's human labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
