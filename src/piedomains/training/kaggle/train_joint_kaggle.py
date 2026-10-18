#!/usr/bin/env python3
r"""Measure every way of combining text and screenshots, in one Kaggle session.

Settings → Accelerator → **GPU T4 x2**, Internet → On, and attach three Datasets:
`piedomains-text-model` (the text tower), `piedomains-text-v13` (the splits) and
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

from ._bootstrap import ensure_usable_gpu  # inlined into the notebook by push_kernel.py

REPO = "https://github.com/themains/piedomains.git"
BRANCH = "taxonomy-round-2"

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")

#: Attached Datasets, found by search: mount paths are not one convention. The splits
#: appeared under /kaggle/input/datasets/<owner>/<name> while an earlier Dataset appeared
#: at /kaggle/input/<name>, which cost two runs before a listing was printed.
#: The trained text checkpoint, uploaded as a Dataset -- not the corpus.
TEXT_MODEL_DATASET = "piedomains-text-model"
SPLITS_DATASET = "piedomains-text-v13"

IMAGE_MODEL = "gojiberries/piedomains-image"
IMAGE_MODEL_REVISION = "2c5704bd8f3e81b81548f568b6f4b774fb093836"

IMAGE_SIZE = 224
MAX_PER_CLASS = 3000

#: Frozen towers. 13,633 paired examples against ~200M parameters overfits readily, and a
#: head over frozen encoders trains in minutes rather than hours.
JOINT_EPOCHS = 6
JOINT_BATCH = 16


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
    required = (
        "prepare_images.py",
        "fuse.py",
        "train_joint.py",
        "_finetune.py",
        "stage_images.py",
        "download_corpus.py",
    )
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
            "transformers>=5.0",
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


def build_images() -> Path:
    """Stream the screenshot corpus and resize it.

    Splits come from :func:`piedomains.training.splits.split_of`, inside
    ``prepare_images``, so the screenshots and the text corpus cannot disagree about
    where a domain belongs. That used to be a flag you had to remember, and forgetting
    it put the image model's training set inside the domains fusion scored -- reported
    as 0.768 against an honest 0.429. The work is
    :mod:`piedomains.training.stage_images`, shared with the image kernel.

    Returns:
        Path: Directory of resized screenshots.
    """
    run(
        [
            sys.executable,
            "-m",
            "piedomains.training.stage_images",
            "--temp",
            str(TEMP),
            "--size",
            str(IMAGE_SIZE),
            "--max-per-class",
            str(MAX_PER_CLASS),
        ]
    )
    return TEMP / f"images-{IMAGE_SIZE}"


def main() -> int:
    """Build images, then measure fusion and the joint model.

    Returns:
        int: Zero if the run completed; the comparison's verdict is in the reports.
    """
    if not os.path.exists("/kaggle"):
        print("This script expects a Kaggle notebook environment.", file=sys.stderr)
        return 1

    _, text_model, splits = setup()
    ensure_usable_gpu(TEMP, run)

    from huggingface_hub import snapshot_download

    image_model = snapshot_download(repo_id=IMAGE_MODEL, revision=IMAGE_MODEL_REVISION)
    print(f"image model at {image_model}")

    images = build_images()

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
