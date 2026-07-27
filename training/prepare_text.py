#!/usr/bin/env python3
r"""Turn a scraped HTML corpus into a labelled, split text dataset.

Reads a directory tree of ``<category>/<domain>.html`` (the shape the Shallalist
Dataverse tarballs unpack into), extracts text with the same cleaner the serving
path uses, applies the filtering the original training run applied, and writes
train/val/test JSONL.

Reproducing the original filtering matters: the shipped model has 39 classes,
not the 73 in shallalist_cats.txt, because training dropped `chat`, `hacking`
and `webtv` explicitly and then dropped every category with fewer than 100
documents (notebooks/04_train_model.ipynb).

Usage:
    python training/prepare_text.py --corpus data/shallalist --out data/prepared
    python training/prepare_text.py --corpus data/shallalist --out data/prepared \\
        --min-docs 100 --min-tokens 6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

#: Categories the original training run removed by name before any thresholding.
DROPPED_BY_NAME = {"chat", "hacking", "webtv"}


def iter_documents(corpus: Path) -> Iterator[tuple[str, str, Path]]:
    """Walk a ``<category>/<domain>.html`` tree.

    Args:
        corpus: Root of the unpacked corpus.

    Yields:
        tuple[str, str, Path]: ``(category, domain, html_path)`` per document.
    """
    for category_dir in sorted(p for p in corpus.iterdir() if p.is_dir()):
        category = category_dir.name
        for html_path in sorted(category_dir.rglob("*.html")):
            yield category, html_path.stem, html_path


def extract(html_path: Path) -> str:
    """Extract cleaned page text from one document.

    Uses the serving cleaner so the training and inference pipelines cannot
    drift. The domain prefix is deliberately NOT added here: deduplication has
    to run on the page text alone, or boilerplate shared across domains looks
    unique because each copy carries a different prefix.

    Args:
        html_path: File to read.

    Returns:
        str: Cleaned text, or ``""`` if the file could not be read.
    """
    from piedomains.text_processor import TextProcessor

    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return TextProcessor.process_html_to_text(html)


def model_input(domain: str, text: str) -> str:
    """Build the training input, matching the original run.

    Args:
        domain: Domain the document belongs to.
        text: Cleaned page text.

    Returns:
        str: ``"<domain-stem> <text>"``, as in notebooks/04_train_model.ipynb.
    """
    stem = domain.rsplit(".", 1)[0] if "." in domain else domain
    return f"{stem} {text}"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True, help="Unpacked <category>/*.html")
    parser.add_argument("--out", required=True, help="Output directory for JSONL")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=6,
        help="Drop documents with fewer tokens (training used >5)",
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=100,
        help="Drop categories with fewer documents (training used 100)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument(
        "--limit", type=int, help="Stop after N documents (for a smoke run)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the prepared dataset.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code.
    """
    args = build_parser().parse_args(argv)
    corpus = Path(args.corpus)
    if not corpus.is_dir():
        sys.stderr.write(f"corpus not found: {corpus}\n")
        return 1

    records: list[dict[str, str]] = []
    seen_text: Counter[str] = Counter()

    for category, domain, path in iter_documents(corpus):
        if category in DROPPED_BY_NAME:
            continue
        text = extract(path)
        if len(text.split()) < args.min_tokens:
            continue
        records.append({"domain": domain, "category": category, "text": text})
        seen_text[text] += 1
        if args.limit and len(records) >= args.limit:
            break

    # Training blanked boilerplate that appeared under more than one domain,
    # because identical navigation chrome carries no category signal. This must
    # happen before the domain prefix is added, or every copy looks unique.
    records = [r for r in records if seen_text[r["text"]] == 1]
    for record in records:
        record["text"] = model_input(record["domain"], record["text"])

    counts = Counter(r["category"] for r in records)
    keep = {c for c, n in counts.items() if n >= args.min_docs}
    dropped = sorted(set(counts) - keep)
    records = [r for r in records if r["category"] in keep]

    # Reproducible train/val/test split; not a security context.
    rng = random.Random(args.seed)  # noqa: S311
    rng.shuffle(records)
    n = len(records)
    train_end, val_end = int(0.8 * n), int(0.9 * n)
    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    labels = sorted(keep)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print(f"documents kept: {n}")
    print(f"categories: {len(labels)}")
    if dropped:
        print(f"dropped for <{args.min_docs} docs: {', '.join(dropped)}")
    for name, rows in splits.items():
        print(f"  {name:5s} {len(rows):7d}")
    print(f"wrote {out_dir}/[train|val|test].jsonl and labels.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
