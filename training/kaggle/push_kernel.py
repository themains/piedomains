#!/usr/bin/env python3
r"""Build and push a Kaggle notebook from ``train_image_kaggle.py``.

**Why this is a script and not three manual steps.** The notebook Kaggle runs is a copy
of the training script, and twice in this project a fix was made to the script, verified
by reasoning about it, and then pushed from a stale copy -- so two consecutive runs
failed on a bug that had already been "fixed". Generating the notebook from the script on
every push removes the copy that can go stale.

The backbone substitution is asserted rather than assumed: a rename that stops it
matching raises here instead of silently pushing the baseline under the challenger's
name, which would make the bake-off compare a model against itself.

Usage::

    uv run python training/kaggle/push_kernel.py                       # ViT baseline
    uv run python training/kaggle/push_kernel.py \\
        --id soodoku/piedomains-image-siglip \\
        --backbone google/siglip2-base-patch16-224
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).parent / "train_image_kaggle.py"
BACKBONE_LINE = re.compile(r'^BACKBONE = "[^"]+"$', re.MULTILINE)


def build_source(backbone: str | None) -> str:
    """Read the training script, optionally swapping the backbone.

    Args:
        backbone: Hub id of the vision encoder to train, or ``None`` to keep the one in
            the script.

    Returns:
        str: The script source to embed in the notebook.

    Raises:
        SystemExit: If the backbone assignment cannot be found. Pushing a challenger that
            silently kept the baseline's encoder would compare the baseline to itself.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    if backbone is None:
        return source
    if not BACKBONE_LINE.search(source):
        raise SystemExit(f'no `BACKBONE = "..."` assignment found in {SCRIPT}')
    return BACKBONE_LINE.sub(f'BACKBONE = "{backbone}"', source, count=1)


def build_notebook(source: str, title: str) -> dict:
    """Wrap the script in a two-cell notebook.

    Args:
        source: The training script.
        title: Heading for the markdown cell.

    Returns:
        dict: An nbformat-4 notebook.
    """
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [f"%%writefile run_image.py\n{source}"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["!python run_image.py"],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Push a training kernel to Kaggle")
    parser.add_argument("--id", default="soodoku/piedomains-image", help="Kaggle slug")
    parser.add_argument("--backbone", default=None, help="Override BACKBONE")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset to attach, repeatable. Stage 4 needs the fusion splits mounted; "
        "without them fusion is skipped rather than fitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Write the notebook but do not push"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the notebook and push it to Kaggle.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status.

    Raises:
        SystemExit: If the backbone cannot be substituted, or the Kaggle CLI is absent.
    """
    args = build_parser().parse_args(argv)
    title = args.id.split("/")[-1]
    source = build_source(args.backbone)

    match = re.search(r'^BACKBONE = "([^"]+)"$', source, re.MULTILINE)
    print(f"{title}: backbone {match.group(1) if match else 'unknown'}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        (staging / f"{title.replace('-', '_')}.ipynb").write_text(
            json.dumps(build_notebook(source, title), indent=1), encoding="utf-8"
        )
        (staging / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": args.id,
                    "title": title,
                    "code_file": f"{title.replace('-', '_')}.ipynb",
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": True,
                    "enable_gpu": True,
                    "enable_internet": True,
                    "dataset_sources": args.dataset or [],
                    "competition_sources": [],
                    "kernel_sources": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if args.dry_run:
            print(f"dry run: notebook built for {args.id}, not pushed")
            return 0
        kaggle = shutil.which("kaggle")
        if kaggle is None:
            raise SystemExit("kaggle CLI not on PATH; try `uv run --with kaggle ...`")
        return subprocess.run(  # noqa: S603 -- resolved path, constant arguments
            [kaggle, "kernels", "push", "-p", str(staging)], check=False
        ).returncode


if __name__ == "__main__":
    sys.exit(main())
