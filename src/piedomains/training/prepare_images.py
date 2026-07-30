#!/usr/bin/env python3
"""Turn the screenshot tarballs into a compact, labelled image dataset.

**Streams, and preprocesses once.** The screenshot corpus is 47.58 GB across 28
tarballs — larger than the free disk on the machine this was written on, and larger than
Kaggle's 20 GB persistent volume. Each tarball is read member-by-member, every screenshot
is resized to the training resolution, and the original is never written to disk. The
result is ~248k images at 224px, roughly 5 GB, which fits anywhere and makes each
training epoch a cheap read instead of a full-size JPEG decode.

Run it once and publish the output; training sessions then mount it read-only.

**Labels come from the same place the text model's do.** Screenshots are named for their
domain, and `taxonomy.map_category` resolves that domain's Shallalist categories to the
47-class label set — so the taxonomy corrections (unlearnable classes removed,
`recreation`/`hobby` split, adult categories merged) apply to images for free.

Usage::

    python training/prepare_images.py --corpus data/corpus --out data/images-224
    python training/prepare_images.py --corpus data/corpus --out data/images-224 \
        --size 384 --limit 5000
"""

from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from .taxonomy import map_category

#: Screenshots whose shorter side is below this are placeholders or errors, not pages.
MIN_SIDE = 64


def load_domain_labels(cache: Path) -> dict[str, str]:
    """Resolve each domain to its single training label.

    A domain in more than one category after mapping is dropped rather than assigned
    arbitrarily: with one label per image, an ambiguous domain is a coin flip that shows
    up as noise in exactly the categories that overlap most.

    Args:
        cache: Directory of cached ``<category>.txt`` domain lists, as written by
            ``prepare_text.py``.

    Returns:
        dict[str, str]: Domain to label.

    Raises:
        SystemExit: If the cache is missing, naming the step that fills it.
    """
    if not cache.is_dir():
        raise SystemExit(
            f"{cache} not found -- run prepare_text.py first to populate the label cache"
        )
    owners: dict[str, set[str]] = defaultdict(set)
    for path in sorted(cache.glob("*.txt")):
        raw = path.stem.replace("__", "/")
        label = map_category(raw)
        if label is None:
            continue
        for line in path.read_text().splitlines():
            domain = line.strip().lower()
            if domain:
                owners[domain].add(label)
    return {d: next(iter(v)) for d, v in owners.items() if len(v) == 1}


def load_screenshot_index(path: Path) -> dict[str, str]:
    """Map a screenshot's filename stem to the domain it depicts.

    The corpus names files by an integer index, not by domain -- which is exactly why
    ``screenshot-index.tab`` (``index,full_domain``) ships alongside it. Reading the stem
    as a domain silently matches nothing: the first run through this script labelled
    0 of 202,897 screenshots and wrote empty splits.

    Args:
        path: ``screenshot-index.tab``.

    Returns:
        dict[str, str]: Filename stem to domain.

    Raises:
        SystemExit: If the index is missing, naming the step that fetches it.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run download_corpus.py --set labels first"
        )
    import csv

    with open(path, encoding="utf-8") as handle:
        return {
            row["index"].strip(): row["full_domain"].strip().lower()
            for row in csv.DictReader(handle)
            if row.get("index") and row.get("full_domain")
        }


def iter_screenshots(
    corpus: Path, index: dict[str, str]
) -> Iterator[tuple[str, bytes]]:
    """Stream ``(domain, image_bytes)`` from every screenshot tarball.

    Args:
        corpus: Directory holding ``screenshot-*.tar.gz``.
        index: Filename stem to domain, from :func:`load_screenshot_index`.

    Yields:
        tuple[str, bytes]: Domain and raw image bytes, one per member.

    Raises:
        SystemExit: If no screenshot tarballs are present.
    """
    tarballs = sorted(corpus.glob("screenshot-*.tar.gz"))
    if not tarballs:
        raise SystemExit(
            f"no screenshot-*.tar.gz in {corpus} -- "
            "run download_corpus.py --set screenshots"
        )
    for tarball in tarballs:
        print(f"  reading {tarball.name}", flush=True)
        with tarfile.open(tarball, "r|gz") as tar:  # streaming mode
            for member in tar:
                if not member.isfile():
                    continue
                stem = Path(member.name).stem
                # Files are named by index; fall back to treating the stem as a domain
                # so an already-renamed corpus still works.
                domain = index.get(stem.lstrip("0") or "0") or index.get(stem)
                if domain is None and "." in stem:
                    domain = stem.lower()
                if not domain:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                yield domain, handle.read()


def resize(raw: bytes, size: int) -> bytes | None:
    """Decode, square-crop and resize one screenshot.

    Screenshots are tall: a homepage capture is mostly below the fold, and the part that
    identifies a site is at the top. Cropping to the top square before resizing keeps
    that rather than squashing a 1280x5000 page into an unreadable strip.

    Args:
        raw: Original image bytes.
        size: Target edge length.

    Returns:
        bytes | None: JPEG bytes, or ``None`` if the image is unusable.
    """
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return None
    if min(img.size) < MIN_SIDE:
        return None

    img = img.convert("RGB")
    width, height = img.size
    edge = min(width, height)
    left = (width - edge) // 2
    img = img.crop((left, 0, left + edge, edge))  # top square, horizontally centred
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def align_splits(
    kept: list[tuple[str, str]], text_data: Path
) -> dict[str, list[tuple[str, str]]]:
    """Assign screenshots to the split their domain already has on the text side.

    **The bug this fixes.** Both preparers shuffled their own list with the same seed, but
    the lists differ -- 46,754 text documents against 44,712 screenshots -- so the same
    domain landed in unrelated splits. Roughly 80% of the domains held out from the text
    model were therefore in the image model's *training* set, and fusion, which fits and
    scores on exactly those paired domains, was reading the image model's memory. It
    showed: image-only scored 0.473 on its own held-out split and 0.768 on the paired
    domains.

    Aligning to the text splits rather than re-splitting both keeps the trained text model
    valid -- its held-out data stays held out. A domain absent from the text splits cannot
    reach fusion, so it goes to train.

    Args:
        kept: ``(domain, category)`` pairs that have a usable screenshot.
        text_data: A ``prepare_text.py`` output directory to take assignments from.

    Returns:
        dict[str, list[tuple[str, str]]]: Split name to rows.

    Raises:
        SystemExit: If no split files are found, since silently falling back to the
            unaligned split is what made the numbers wrong in the first place.
    """
    assignment: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = text_data / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    assignment[json.loads(line)["domain"].lower()] = split
    if not assignment:
        raise SystemExit(f"no train/val/test.jsonl under {text_data}")

    splits: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for domain, label in kept:
        splits[assignment.get(domain.lower(), "train")].append((domain, label))
    print(
        f"  aligned to {text_data}: "
        + ", ".join(f"{k} {len(v):,}" for k, v in splits.items())
    )
    return splits


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Turn screenshot tarballs into a labelled image dataset"
    )
    parser.add_argument(
        "--corpus", required=True, help="Directory of screenshot tarballs"
    )
    parser.add_argument(
        "--out", required=True, help="Where to write the resized dataset"
    )
    parser.add_argument(
        "--label-cache",
        default="data/labels/domains",
        help="Per-category domain lists, written by prepare_text.py",
    )
    parser.add_argument(
        "--index",
        default="data/labels/screenshot-index.tab",
        help="Maps a screenshot's filename index to its domain",
    )
    parser.add_argument("--size", type=int, default=224, help="Output edge length")
    parser.add_argument(
        "--min-docs",
        type=int,
        default=100,
        help="Drop classes with fewer images than this",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Cap images per class to balance the long tail; 0 keeps everything",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument(
        "--respect-splits",
        default="",
        help="A prepare_text.py output directory. Assign each screenshot to the split "
        "its domain already has there, so a domain held out from the text model is "
        "never in the image model's training set. Required for an honest fusion "
        "measurement -- without it the two splits are independent and ~80%% of the "
        "domains fusion scores on were trained on by the image model.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N images")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to an existing output rather than starting over. Lets a caller stream "
        "one tarball at a time -- fetch, extract, delete -- instead of staging the whole "
        "47.58 GB corpus. Splits are rewritten from the accumulated manifest each time, "
        "so the output is complete and consistent after every invocation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the resized, labelled image dataset.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status.

    Raises:
        SystemExit: If the label cache, index or split files are missing, or if no
            screenshot matched a label at all.
    """
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    labels = load_domain_labels(Path(args.label_cache))
    print(f"labelled domains: {len(labels):,}")
    index = load_screenshot_index(Path(args.index))
    print(f"screenshot index: {len(index):,} entries")

    # The manifest is the accumulator across --append invocations; the splits below are
    # derived from it, so every run leaves a consistent dataset behind.
    manifest = out / "manifest.jsonl"
    kept: list[tuple[str, str]] = []
    per_class: Counter[str] = Counter()
    if args.append and manifest.exists():
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kept.append((row["domain"], row["category"]))
            per_class[row["category"]] += 1
        print(f"resuming from {len(kept):,} images already prepared")
    elif not args.append and manifest.exists():
        manifest.unlink()

    already = {d for d, _ in kept}
    scanned = unlabelled = unusable = 0

    for domain, raw in iter_screenshots(Path(args.corpus), index):
        scanned += 1
        if scanned % 20000 == 0:
            print(f"  scanned {scanned:,}, kept {len(kept):,}", flush=True)

        label = labels.get(domain)
        if label is None:
            unlabelled += 1
            continue
        if domain in already:
            continue
        # Check the cap before decoding: resizing dominates the run, and doing it for a
        # class that is already full is pure waste.
        if args.max_per_class and per_class[label] >= args.max_per_class:
            continue

        small = resize(raw, args.size)
        if small is None:
            unusable += 1
            continue

        (out / "images" / f"{domain}.jpg").write_bytes(small)
        with open(manifest, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"domain": domain, "category": label}) + "\n")
        kept.append((domain, label))
        already.add(domain)
        per_class[label] += 1
        if args.limit and len(kept) >= args.limit:
            break

    keep_classes = {c for c, n in per_class.items() if n >= args.min_docs}
    dropped = sorted(set(per_class) - keep_classes)
    kept = [(d, c) for d, c in kept if c in keep_classes]

    # Reproducible split; not a security context.
    rng = random.Random(args.seed)  # noqa: S311
    rng.shuffle(kept)
    n = len(kept)
    splits = {
        "train": kept[: int(0.8 * n)],
        "val": kept[int(0.8 * n) : int(0.9 * n)],
        "test": kept[int(0.9 * n) :],
    }

    if args.respect_splits:
        splits = align_splits(kept, Path(args.respect_splits))
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for domain, label in rows:
                handle.write(json.dumps({"domain": domain, "category": label}) + "\n")

    classes = sorted(keep_classes)
    (out / "labels.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    # An empty dataset must fail here, not later. Unmatched members are skipped inside
    # the iterator, so `scanned` stays 0 when the index is wrong -- checking `kept` is
    # the condition that actually catches it. The first Kaggle run wrote empty splits,
    # reported success, and surfaced as `num_samples=0` deep in the training loop.
    if not kept:
        raise SystemExit(
            "matched no screenshots to a label -- refusing to write an empty dataset. "
            "Check --index (the corpus names files by index, not domain) and "
            "--label-cache."
        )

    print(f"\nscanned {scanned:,} screenshots")
    print(f"  no usable label : {unlabelled:,}")
    print(f"  undecodable     : {unusable:,}")
    print(f"  kept            : {n:,} across {len(classes)} classes")
    if dropped:
        print(f"  dropped for <{args.min_docs}: {', '.join(dropped)}")
    for name, rows in splits.items():
        print(f"    {name:5s} {len(rows):7,d}")
    print(f"wrote {out}/[train|val|test].jsonl, labels.json and images/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
