#!/usr/bin/env python3
r"""Capture homepage screenshots for domains in a prepared text split.

**Why capture our own when 248,003 already exist.** The Dataverse screenshots are 2022
captures, and inference runs on pages rendered today by a different browser. Training on
one distribution and serving on another is a silent accuracy tax that no held-out score
from the old corpus can reveal. These captures come through the same ``DataCollector`` the
library uses at inference, so what the model trains on is what it will see.

They are also the only way to fuse locally. ``fuse.py`` needs a screenshot for every
paired domain, and the Kaggle runs keep their resized corpus in ``/kaggle/temp`` so the
kernel output stays retrievable -- meaning it is gone when the session ends.

**Measured throughput, and why the training corpus is the wrong target.** Over 304
Shallalist domains this managed 0.017 domains/sec and reached only 60% of them: 253 TLS
failures, 184 exhausted retries, 56 unresolvable names, 38 timeouts, 26 refused
connections. Those are 2022 registrations being asked to answer in 2026, and every dead
one costs a full timeout. At that rate the 46,754-domain corpus is ~33 days, not the
16-32 hours a healthy domain list would take -- so point this at a *current* list
(Tranco) rather than at the training corpus.

The 60% reachability is itself worth knowing: about two in five domains the models were
trained on no longer exist.

**Resumable, because it will be interrupted.** A domain whose JPEG already exists is
skipped, so re-running continues rather than restarting.

Reuses the acquisition pipeline rather than writing a second scraper: bot-wall detection,
archive.org fallback and thin-content refusal all apply, and a domain that only yields a
challenge page is skipped rather than saved as a picture of a CAPTCHA.

Usage::

    uv run python training/capture_screenshots.py \\
        --data data/prepared-taxonomy --out data/images-self --splits val test

    # a quick slice, for checking fusion end to end
    uv run python training/capture_screenshots.py \\
        --data data/prepared-taxonomy --out /tmp/shots --splits val test --limit 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

#: Written at this edge length, matching prepare_images.py, so the two sources are
#: interchangeable as training data.
DEFAULT_SIZE = 224


def read_domains(data: Path, splits: list[str], limit: int) -> list[dict[str, Any]]:
    """Collect the domains to capture from prepared text splits.

    Args:
        data: A ``prepare_text.py`` output directory.
        splits: Split names to read.
        limit: Cap per split; ``0`` means no cap.

    Returns:
        list[dict[str, Any]]: Records carrying ``domain`` and ``category``.

    Raises:
        SystemExit: If a named split does not exist.
    """
    rows: list[dict[str, Any]] = []
    for split in splits:
        path = data / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found")
        with open(path, encoding="utf-8") as handle:
            found = [json.loads(line) for line in handle if line.strip()]
        if limit:
            found = found[:limit]
        rows.extend({"domain": r["domain"], "category": r["category"]} for r in found)
    return rows


def resize_into(source: Path, target: Path, size: int) -> bool:
    """Downscale one capture to a square JPEG.

    The top square is taken rather than the whole page: a homepage screenshot is far
    taller than it is wide, and squashing it to a square destroys the aspect ratio the
    layout signal lives in.

    Args:
        source: The captured PNG.
        target: Where to write the JPEG.
        size: Output edge length.

    Returns:
        bool: Whether a file was written.
    """
    from PIL import Image

    try:
        with Image.open(source) as img:
            img = img.convert("RGB")
            edge = min(img.width, img.height)
            img = img.crop((0, 0, edge, edge)).resize(
                (size, size), Image.Resampling.LANCZOS
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            img.save(target, "JPEG", quality=85, optimize=True)
        return True
    except Exception as exc:  # one bad capture must not stop the run
        print(f"  {target.stem}: could not convert ({exc})")
        return False


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Capture homepage screenshots")
    parser.add_argument(
        "--data", required=True, help="prepare_text.py output directory"
    )
    parser.add_argument(
        "--out", required=True, help="Where to write images/ and splits"
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Per-split cap; 0 = all")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--batch", type=int, default=16, help="Domains per fetch batch")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Capture, resize and record screenshots.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status; non-zero if nothing was captured at all.
    """
    args = build_parser().parse_args(argv)
    from ..data_collector import DataCollector

    out = Path(args.out)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)

    rows = read_domains(Path(args.data), args.splits, args.limit)
    pending = [r for r in rows if not (images / f"{r['domain']}.jpg").exists()]
    print(f"{len(rows):,} domains, {len(rows) - len(pending):,} already captured")

    written = 0
    cache = Path(tempfile.mkdtemp(prefix="piedomains-capture-"))
    try:
        for start in range(0, len(pending), args.batch):
            chunk = pending[start : start + args.batch]
            names = [r["domain"] for r in chunk]
            collector = DataCollector(cache_dir=str(cache))
            try:
                result = collector.collect_batch(names)
            except Exception as exc:  # a bad batch must not end the run
                print(f"  batch at {start} failed: {exc}")
                continue

            for entry in result.get("domains", []):
                shot = entry.get("image_path")
                if not entry.get("fetch_success") or not shot:
                    continue
                source = Path(shot)
                if not source.is_absolute():
                    source = cache / source
                if source.exists() and resize_into(
                    source, images / f"{entry['domain']}.jpg", args.size
                ):
                    written += 1

            done = min(start + args.batch, len(pending))
            print(
                f"  {done:,}/{len(pending):,} attempted, {written:,} captured",
                flush=True,
            )
            # The PNGs are only an intermediate; keeping them would grow without bound.
            shutil.rmtree(cache, ignore_errors=True)
            cache.mkdir(parents=True, exist_ok=True)
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    have = {p.stem for p in images.glob("*.jpg")}
    kept = [r for r in rows if r["domain"] in have]
    (out / "labels.json").write_text(
        json.dumps(sorted({r["category"] for r in kept}), indent=2), encoding="utf-8"
    )
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")

    print(f"\n{len(kept):,} of {len(rows):,} domains have a screenshot in {images}")
    if not kept:
        print(
            "nothing captured -- every domain failed, so this is a pipeline problem "
            "rather than a slow crawl",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
