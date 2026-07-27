#!/usr/bin/env python3
"""Evaluate a piedomains model against a labelled domain set.

Nothing in this repo previously measured accuracy, which is why a broken image
preprocessing path and a train/serve mismatch both survived. Every model claim
should come from a run of this script.

Usage:
    # Collect content once, then score it (re-runnable offline)
    python training/evaluate.py --labels tests/eval/labels.csv --method text
    python training/evaluate.py --labels tests/eval/labels.csv --method images
    python training/evaluate.py --labels tests/eval/labels.csv --method combined

    # Compare a change: run before, run after, diff the JSON
    python training/evaluate.py --out before.json
    python training/evaluate.py --out after.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from metrics import format_summary, summarize


def load_labels(path: Path) -> list[dict[str, str]]:
    """Read a labelled evaluation set.

    Args:
        path: CSV with ``domain,expected`` columns; ``#`` lines are comments.

    Returns:
        list[dict]: One dict per labelled domain.

    Raises:
        ValueError: If a label is not one of the model's known categories.
    """
    from piedomains.constants import classes

    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        lines = [ln for ln in handle if not ln.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        domain = (row.get("domain") or "").strip()
        expected = (row.get("expected") or "").strip()
        if not domain or not expected:
            continue
        if expected not in classes:
            raise ValueError(
                f"{path}: '{expected}' for {domain} is not one of the "
                f"{len(classes)} known categories"
            )
        rows.append({"domain": domain, "expected": expected})
    return rows


def evaluate(
    labels: list[dict[str, str]],
    method: str,
    cache_dir: str,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Classify every labelled domain and pair predictions with truth.

    Args:
        labels: Output of :func:`load_labels`.
        method: ``text``, ``images`` or ``combined``.
        cache_dir: Where fetched content lives, so reruns are offline.
        use_cache: Reuse previously fetched content.

    Returns:
        list[dict]: Rows with ``expected``, ``predicted``, ``confidence`` and
        the failure fields from the run report.
    """
    from piedomains.api import DomainClassifier

    classifier = DomainClassifier(cache_dir=cache_dir)
    domains = [row["domain"] for row in labels]
    truth = {row["domain"]: row["expected"] for row in labels}

    method_fn = {
        "text": classifier.classify_by_text,
        "images": classifier.classify_by_images,
        "combined": classifier.classify,
    }[method]
    run = method_fn(domains, use_cache=use_cache)

    rows: list[dict[str, Any]] = []
    for result in run["results"]:
        domain = result.get("domain")
        rows.append(
            {
                "domain": domain,
                "expected": truth.get(domain),
                "predicted": result.get("category"),
                "confidence": result.get("confidence"),
                "status": result.get("status"),
                "stage": result.get("stage"),
                "error_code": result.get("error_code"),
                # Provenance, so a score can be read against where the content
                # came from: a row recovered from archive.org is not evidence
                # about the live site.
                "source": result.get("source"),
                "snapshot_timestamp": result.get("snapshot_timestamp"),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--labels",
        default="tests/eval/labels.csv",
        help="CSV of domain,expected (default: tests/eval/labels.csv)",
    )
    parser.add_argument(
        "--method",
        default="text",
        choices=["text", "images", "combined"],
        help="Which classifier path to evaluate (default: text)",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/eval",
        help="Content cache, so repeat runs need no network",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Refetch instead of reusing the cache"
    )
    parser.add_argument("--out", help="Write the full result JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run an evaluation and print the summary.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code.
    """
    args = build_parser().parse_args(argv)
    labels = load_labels(Path(args.labels))
    if not labels:
        sys.stderr.write(f"no labelled domains in {args.labels}\n")
        return 1

    rows = evaluate(labels, args.method, args.cache_dir, use_cache=not args.no_cache)
    summary = summarize(rows)

    print(f"\n=== piedomains eval: method={args.method} n={len(labels)} ===")
    print(format_summary(summary))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {"method": args.method, "summary": summary, "rows": rows},
                handle,
                indent=2,
                default=str,
            )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
