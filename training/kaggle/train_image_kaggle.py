#!/usr/bin/env python3
"""Train the screenshot classifier on Kaggle, free.

Paste this into a Kaggle notebook cell (Settings → Accelerator → GPU P100 or T4 x2,
Internet → On). It is a plain script rather than a notebook so it can be diffed and
linted; `jupytext` or a single `%run` cell turns it into one.

**Why Kaggle.** 30 GPU-hours a week at no cost, 12-hour sessions, and 200 GB of dataset
storage. The job needs 3-5 hours. Colab pay-as-you-go would be $1-2 on a T4; there is no
reason to pay it.

**The disk problem, and the shape of the answer.** The corpus is 47.58 GB across 28
tarballs. `/kaggle/working` persists only 20 GB and `/kaggle/temp` gives ~60 GB that
vanishes when the session ends. So:

1. Download one tarball at a time into `/kaggle/temp`.
2. Stream it, resize every screenshot to 224px, write the small copy to
   `/kaggle/working/images-224`.
3. Delete the tarball before fetching the next.

Peak disk is one tarball (~1.7 GB) plus the growing 224px set (~5 GB), not 47.58 GB.

**Do this once.** Publish `/kaggle/working/images-224` as a Kaggle Dataset when stage 1
finishes. Later sessions mount it read-only at `/kaggle/input/...` and skip straight to
training, which matters because stage 1 is bounded by download speed and stage 2 by the
12-hour session cap.

**Resume across sessions.** `train_image.py --resume` picks up from the checkpoint in
`--out`, and it only ever writes the best epoch, so an interrupted run costs at most one
epoch rather than the lot.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

REPO = "https://github.com/themains/piedomains.git"

#: Kaggle's persistent output volume, 20 GB. The resized dataset and the checkpoint live
#: here so they survive the session.
WORK = Path("/kaggle/working")

#: Ephemeral scratch, ~60 GB, wiped when the session ends. Tarballs land here and are
#: deleted as soon as they have been streamed.
TEMP = Path("/kaggle/temp")

#: Set this after publishing stage 1's output as a Kaggle Dataset, then attach it to the
#: notebook. Skips the download and resize entirely.
PREPARED_DATASET: str | None = None  # e.g. "/kaggle/input/piedomains-screenshots-224"

IMAGE_SIZE = 224
EPOCHS = 4
BATCH_SIZE = 32


def run(cmd: list[str], **kwargs) -> None:
    """Run a command, streaming output, and fail loudly.

    Args:
        cmd: Command and arguments.
        **kwargs: Passed through to ``subprocess.run``.

    Raises:
        SystemExit: If the command exits non-zero.
    """
    print("$", " ".join(cmd), flush=True)
    # Every argument is constructed in this file from module constants; nothing here
    # comes from user input.
    result = subprocess.run(cmd, **kwargs)  # noqa: S603
    if result.returncode != 0:
        raise SystemExit(f"command failed with {result.returncode}: {' '.join(cmd)}")


def setup() -> Path:
    """Clone the repository and install what training needs.

    Returns:
        Path: The repository root.
    """
    repo = WORK / "piedomains"
    if not repo.exists():
        run(["git", "clone", "--depth", "1", REPO, str(repo)])
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
    return repo


def stage_one_prepare(repo: Path) -> Path:
    """Download, stream, resize and discard — one tarball at a time.

    Peak disk is a single tarball plus the growing resized set, so this fits in
    Kaggle's volumes where the full corpus would not.

    Args:
        repo: Repository root.

    Returns:
        Path: Directory holding the resized dataset.
    """
    out = WORK / "images-224"
    if (out / "labels.json").exists():
        print(f"{out} already built; skipping stage 1")
        return out

    corpus = TEMP / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)

    # Labels first: small, and prepare_images.py needs the per-category domain lists.
    labels_dir = WORK / "labels"
    if not (labels_dir / "shallalist_cats.txt").exists():
        run(
            [
                sys.executable,
                str(repo / "training" / "download_corpus.py"),
                "--set",
                "labels",
                "--out",
                str(labels_dir),
            ]
        )

    # The domain lists are fetched by prepare_text.py's mirror logic on first use; a tiny
    # prepare_text run populates the cache without processing the text corpus.
    cache = WORK / "labels" / "domains"
    if not cache.exists() or not any(cache.glob("*.txt")):
        print("populating the label cache from the Shallalist mirror...")
        run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {str(repo / 'training')!r});"
                "from prepare_text import load_category_map;"
                f"m = load_category_map(__import__('pathlib').Path({str(labels_dir / 'shallalist_cats.txt')!r}),"
                f" __import__('pathlib').Path({str(cache)!r}));"
                "print(f'{len(m):,} labelled domains')",
            ]
        )

    run(
        [
            sys.executable,
            str(repo / "training" / "download_corpus.py"),
            "--set",
            "screenshots",
            "--out",
            str(corpus),
        ]
    )

    run(
        [
            sys.executable,
            str(repo / "training" / "prepare_images.py"),
            "--corpus",
            str(corpus),
            "--out",
            str(out),
            "--label-cache",
            str(cache),
            "--size",
            str(IMAGE_SIZE),
        ]
    )

    # Reclaim the scratch space before training starts.
    for tarball in corpus.glob("*.tar.gz"):
        tarball.unlink()
    print(f"stage 1 complete: {out}")
    print("PUBLISH THIS AS A KAGGLE DATASET, then set PREPARED_DATASET and re-run.")
    return out


def stage_two_train(repo: Path, data: Path) -> Path:
    """Fine-tune the backbone, resuming if a checkpoint is present.

    Args:
        repo: Repository root.
        data: Directory holding the resized dataset.

    Returns:
        Path: Directory holding the trained model.
    """
    out = WORK / "image-v1"
    cmd = [
        sys.executable,
        str(repo / "training" / "train_image.py"),
        "--data",
        str(data),
        "--out",
        str(out),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--workers",
        "2",
    ]
    if (out / "state.json").exists():
        cmd.append("--resume")
    run(cmd)
    return out


def stage_three_calibrate(repo: Path, data: Path, model: Path) -> None:
    """Fit the temperature, without which fusion is meaningless.

    Averaging an uncalibrated image softmax against a calibrated text distribution is
    exactly the bug the original package shipped. Both sides must be on the same scale
    before they can be combined.

    Args:
        repo: Repository root.
        data: Directory holding the resized dataset.
        model: Directory holding the trained model.
    """
    run(
        [
            sys.executable,
            str(repo / "training" / "calibrate.py"),
            "--model",
            str(model),
            "--data",
            str(data),
            "--modality",
            "image",
        ]
    )


def main() -> int:
    """Run the whole pipeline.

    Returns:
        int: Process exit status.
    """
    if not os.path.exists("/kaggle"):
        print("This script expects a Kaggle notebook environment.", file=sys.stderr)
        return 1

    repo = setup()
    data = Path(PREPARED_DATASET) if PREPARED_DATASET else stage_one_prepare(repo)
    model = stage_two_train(repo, data)
    stage_three_calibrate(repo, data, model)

    print("\nDownload these from the notebook output:")
    print(f"  {model}/  — weights, labels.json, calibration.json, test_metrics.json")
    print("\nThen locally:")
    print(
        "  uv run python training/fuse.py --text models/text-v4 --image models/image-v1"
    )
    print("  Fusion must beat text-only (0.725 accuracy / 0.705 macro-F1)")
    print("  or image classification ships opt-in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
