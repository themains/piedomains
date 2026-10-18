"""Stage the screenshot corpus for training: download, resize, discard.

The Kaggle image and joint kernels both need the resized screenshot set, and both run
this after cloning the repository, the same way they run ``download_corpus`` and
``prepare_images``::

    python -m piedomains.training.stage_images --temp /kaggle/temp --size 224 --max-per-class 3000

Peak disk is a single tarball plus the growing resized set, so this fits in Kaggle's
volumes where the full corpus would not.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    """Run a command, streaming output, and fail loudly.

    Args:
        cmd: Command and arguments.

    Raises:
        subprocess.CalledProcessError: If the command exits non-zero. Not
            ``SystemExit``: per-tarball failures are caught so one bad download does
            not lose the run.
    """
    print("$", " ".join(cmd), flush=True)
    # Every argument is built here from the caller's paths and numbers.
    result = subprocess.run(cmd)  # noqa: S603
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def stage_images(temp: Path, *, size: int, max_per_class: int) -> Path:
    """Download, stream, resize and discard -- one tarball at a time.

    Args:
        temp: Scratch directory for labels, the transient corpus and the output.
        size: Side length, in pixels, of the resized screenshots.
        max_per_class: Most screenshots kept per category.

    Returns:
        Path: Directory holding the resized dataset.

    Raises:
        SystemExit: If the tarballs cannot be enumerated, or every one of them fails.
    """
    out = temp / f"images-{size}"
    if (out / "labels.json").exists():
        print(f"{out} already built; skipping")
        return out

    corpus = temp / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)

    # Labels first: small, and prepare_images.py needs the per-category domain lists.
    labels_dir = temp / "labels"
    if not (labels_dir / "shallalist_cats.txt").exists():
        _run(
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
    cache = temp / "labels" / "domains"
    if not cache.exists() or not any(cache.glob("*.txt")):
        print("populating the label cache from the Shallalist mirror...")
        _run(
            [
                sys.executable,
                "-c",
                "from piedomains.training.prepare_text import load_category_map;"
                f"m = load_category_map(__import__('pathlib').Path({str(labels_dir / 'shallalist_cats.txt')!r}),"
                f" __import__('pathlib').Path({str(cache)!r}));"
                "print(f'{len(m):,} labeled domains')",
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
            _run(
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
            _run(
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
                    str(size),
                    "--max-per-class",
                    str(max_per_class),
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
    print(f"staged {out} ({len(tarballs) - len(skipped)} tarballs streamed)")
    return out


def main(argv: list[str] | None = None) -> int:
    """Stage the resized screenshot set under ``--temp``.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Stage the screenshot corpus for training: download, resize, discard."
    )
    parser.add_argument("--temp", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--max-per-class", type=int, required=True)
    args = parser.parse_args(argv)
    stage_images(args.temp, size=args.size, max_per_class=args.max_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
