#!/usr/bin/env python3
"""Decide whether a re-labelling pass may be trusted, before it rewrites anything.

A relabeller that quietly re-derives its own taxonomy looks exactly like a relabeller that
found real noise: both produce a large pile of confident disagreements. Nothing in the
verdicts themselves distinguishes the two, so the verdicts cannot be their own evidence.

Three checks, and each can fail the whole exercise:

**Against the hand-checked set.** ``tests/eval/labels.csv`` is 54 domains labelled by hand.
It is one-directional -- none of them are the class being audited -- so it only asks
whether the model reads a normal site correctly. That is necessary and nowhere near
sufficient, but it is cheap and it catches a badly-specified prompt immediately.

**Against a healthy class.** Run the identical prompt over a class nobody thinks is broken.
If it "corrects" a large share of that, the disagreements are the model's opinion rather
than the corpus's noise, and the other two numbers mean nothing.

**Against a human.** Stratified over *verdict cells* rather than classes, because the
question is not "is the model accurate" but "is it accurate on the moves it wants to
make". A relabeller at 0.95 on keeps and 0.60 on moves-to-shopping should run in
status-only mode, not correct mode -- a single pooled number hides exactly that.

The sample is deliberately non-proportional, so an unweighted overall rate is wrong; the
report gives per-cell rates and a cell-size-weighted overall.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .relabel import load_verdicts
from .splits import SPLITS, split_of


def wilson(hits: int, n: int) -> tuple[float, float]:
    """Wilson score interval, which behaves at the small counts this deals in.

    Args:
        hits: Successes.
        n: Trials.

    Returns:
        tuple[float, float]: Lower and upper bounds of the 95% interval.
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def cell_of(verdict: Any) -> str:
    """Name the verdict cell a document falls in.

    Args:
        verdict: A verdict record.

    Returns:
        str: The cell name.
    """
    if verdict.status != "ok" or verdict.category is None:
        return f"no-verdict:{verdict.status}"
    if verdict.category == verdict.old_category:
        return "keep"
    if verdict.category in ("parked", "unavailable"):
        return "move:status"
    return f"move:{verdict.category}"


def build_eval_corpus(labels_file: Path, caches: list[Path], out: Path) -> int:
    """Turn the hand-checked label set into a corpus the relabeller can read.

    This exists so the check runs through *exactly* the same code path as the real pass --
    same prompt, same parser, same validation. A bespoke evaluation harness would be
    testing something other than what ships.

    Args:
        labels_file: ``tests/eval/labels.csv``.
        caches: Directories of cached ``<domain>.html`` to draw text from.
        out: Directory to write the corpus into.

    Returns:
        int: How many domains had cached text and made it in.
    """
    from ..text_processor import TextProcessor
    from .evaluate import load_labels
    from .prepare_text import model_input

    processor = TextProcessor()
    rows = load_labels(labels_file)
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    kept = 0
    for row in rows:
        domain = row["domain"]
        html = None
        for cache in caches:
            candidate = cache / f"{domain}.html"
            if candidate.exists():
                html = candidate.read_text(encoding="utf-8", errors="replace")
                break
        if html is None:
            continue
        text = processor.extract_text(html)
        if len(text.split()) < 6:
            continue
        by_split[split_of(domain)].append(
            {
                "domain": domain,
                "category": row["expected"],
                "text": model_input(domain, text),
            }
        )
        kept += 1

    out.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for record in by_split[name]:
                handle.write(json.dumps(record) + "\n")
    labels = sorted({r["category"] for v in by_split.values() for r in v})
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return kept


def score_against_gold(
    verdicts: dict[str, Any], gold: dict[str, str]
) -> dict[str, Any]:
    """Score verdicts against hand-checked labels.

    Args:
        verdicts: Verdicts by domain.
        gold: Hand-checked label by domain.

    Returns:
        dict[str, Any]: Agreement, counts and every disagreement, so they can be read.
    """
    hits = 0
    scored = 0
    misses = []
    for domain, expected in gold.items():
        verdict = verdicts.get(domain)
        if verdict is None or verdict.status != "ok" or verdict.category is None:
            continue
        scored += 1
        if verdict.category == expected:
            hits += 1
        else:
            misses.append(
                {
                    "domain": domain,
                    "expected": expected,
                    "llm": verdict.category,
                    "confidence": verdict.confidence,
                    "evidence": verdict.evidence[:160],
                }
            )
    low, high = wilson(hits, scored)
    return {
        "scored": scored,
        "agreement": hits / scored if scored else 0.0,
        "ci95": [round(low, 4), round(high, 4)],
        "disagreements": misses,
    }


def summarize(verdicts: dict[str, Any]) -> dict[str, Any]:
    """Describe what a pass wants to do, without judging whether it is right.

    Args:
        verdicts: Verdicts by domain.

    Returns:
        dict[str, Any]: Cell counts, statuses, substance types and destinations.
    """
    cells = Counter(cell_of(v) for v in verdicts.values())
    usable = [v for v in verdicts.values() if v.status == "ok" and v.category]
    kept = [v for v in usable if v.category == v.old_category]
    return {
        "total": len(verdicts),
        "statuses": dict(Counter(v.status for v in verdicts.values())),
        "cells": dict(cells.most_common()),
        "kept": len(kept),
        "kept_share": len(kept) / len(usable) if usable else 0.0,
        "substance_of_kept": dict(Counter(v.substance_type for v in kept)),
        "destinations": dict(
            Counter(v.category for v in usable if v.category != v.old_category)
        ),
        "mean_confidence_keep": (
            round(sum(v.confidence for v in kept) / len(kept), 4) if kept else 0.0
        ),
    }


def emit_sheet(
    verdicts: dict[str, Any],
    corpus_text: dict[str, str],
    out: Path,
    per_cell: int,
    seed: int,
) -> dict[str, int]:
    """Write a stratified sample for a human to adjudicate.

    The document's **current label is withheld**. A human shown "this is labelled drugs,
    the model says shopping" is being asked to referee a disagreement rather than to read
    a page, and would inherit both sources' anchoring.

    Args:
        verdicts: Verdicts by domain.
        corpus_text: Document text by domain.
        out: CSV to write.
        per_cell: Sample size per verdict cell.
        seed: Sampling seed.

    Returns:
        dict[str, int]: How many rows were drawn from each cell.
    """
    grouped: dict[str, list[Any]] = defaultdict(list)
    for verdict in verdicts.values():
        cell = cell_of(verdict)
        # Collapse the long tail of single-destination cells so the sheet stays finite.
        if cell.startswith("move:") and cell not in ("move:status",):
            cell = "move:other" if cell != "move:shopping" else "move:shopping"
        grouped[cell].append(verdict)

    rng = random.Random(seed)  # noqa: S311 -- sampling, not security
    drawn: dict[str, int] = {}
    rows = []
    for cell, members in sorted(grouped.items()):
        members = sorted(members, key=lambda v: v.domain)
        rng.shuffle(members)
        chosen = members[:per_cell]
        drawn[cell] = len(chosen)
        for verdict in chosen:
            excerpt = " ".join(corpus_text.get(verdict.domain, "").split())[:400]
            rows.append(
                {
                    "domain": verdict.domain,
                    "cell": cell,
                    "text_excerpt": excerpt,
                    "llm_label": verdict.category or "",
                    "llm_confidence": verdict.confidence,
                    "llm_reasoning": verdict.reasoning[:300],
                    "evidence": verdict.evidence[:200],
                    "human_label": "",
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return drawn


def score_sheet(
    adjudicated: Path, verdicts: dict[str, Any], cells: dict[str, int]
) -> dict[str, Any]:
    """Score an adjudicated sheet, per cell and reweighted.

    Args:
        adjudicated: The returned CSV, with ``human_label`` filled in.
        verdicts: Verdicts by domain.
        cells: Population size of each cell, for reweighting.

    Returns:
        dict[str, Any]: Per-cell and weighted agreement for the model and the old label.
    """
    llm_hits: Counter = Counter()
    old_hits: Counter = Counter()
    totals: Counter = Counter()
    with open(adjudicated, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            human = (row.get("human_label") or "").strip().lower()
            if not human:
                continue
            verdict = verdicts.get(row["domain"])
            if verdict is None:
                continue
            cell = row.get("cell", cell_of(verdict))
            totals[cell] += 1
            if verdict.category == human:
                llm_hits[cell] += 1
            if verdict.old_category == human:
                old_hits[cell] += 1

    per_cell = {}
    for cell, n in totals.items():
        low, high = wilson(llm_hits[cell], n)
        old_low, old_high = wilson(old_hits[cell], n)
        per_cell[cell] = {
            "n": n,
            "llm_agreement": llm_hits[cell] / n,
            "llm_ci95": [round(low, 4), round(high, 4)],
            "old_label_agreement": old_hits[cell] / n,
            "old_ci95": [round(old_low, 4), round(old_high, 4)],
        }

    # Weight by how big each cell actually is in the population, not by how many of it
    # were sampled -- the sample is deliberately non-proportional.
    weight_total = sum(cells.get(c, 0) for c in totals)
    weighted_llm = 0.0
    weighted_old = 0.0
    if weight_total:
        for cell, n in totals.items():
            share = cells.get(cell, 0) / weight_total
            weighted_llm += share * (llm_hits[cell] / n)
            weighted_old += share * (old_hits[cell] / n)
    return {
        "per_cell": per_cell,
        "weighted_llm_agreement": weighted_llm,
        "weighted_old_label_agreement": weighted_old,
        "adjudicated": sum(totals.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Audit an LLM re-labelling pass")
    parser.add_argument("--verdicts", default="", help="Verdict JSONL to audit")
    parser.add_argument("--prompt-hash", default="", help="Which pass to read")
    parser.add_argument("--corpus", default="", help="Corpus the verdicts describe")
    parser.add_argument(
        "--build-eval-corpus",
        default="",
        help="Write a corpus from tests/eval/labels.csv plus cached HTML, and exit",
    )
    parser.add_argument("--labels", default="tests/eval/labels.csv")
    parser.add_argument(
        "--cache",
        action="append",
        default=None,
        help="Cached HTML directory, repeatable",
    )
    parser.add_argument(
        "--against-gold", default="", help="Score against this label CSV"
    )
    parser.add_argument("--emit-sheet", default="", help="Write a human sample here")
    parser.add_argument("--sample-per-cell", type=int, default=40)
    parser.add_argument("--adjudicated", default="", help="Score this returned sheet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the audit.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status.

    Raises:
        SystemExit: If no verdicts file is given, or the one given is empty.
    """
    args = build_parser().parse_args(argv)

    if args.build_eval_corpus:
        caches = [
            Path(c) for c in (args.cache or ["cache/eval5/html", "cache/eval/html"])
        ]
        kept = build_eval_corpus(
            Path(args.labels), caches, Path(args.build_eval_corpus)
        )
        print(f"wrote {args.build_eval_corpus}: {kept} domains with cached text")
        return 0

    if not args.verdicts:
        raise SystemExit("--verdicts is required unless building an eval corpus")

    fingerprint = args.prompt_hash
    if not fingerprint:
        first = next(
            (
                json.loads(line)
                for line in Path(args.verdicts).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            None,
        )
        if first is None:
            raise SystemExit(f"{args.verdicts} is empty")
        fingerprint = first["prompt_hash"]
    verdicts = load_verdicts(Path(args.verdicts), fingerprint)
    print(f"{len(verdicts):,} verdicts under prompt {fingerprint}")

    report: dict[str, Any] = {
        "note": (
            "Model reasoning is a post-hoc rationalisation, not evidence. These labels "
            "must never be used as evaluation gold."
        ),
        "prompt_hash": fingerprint,
        "summary": summarize(verdicts),
    }
    summary = report["summary"]
    print(f"\nkept {summary['kept']}/{len(verdicts)} ({summary['kept_share']:.1%})")
    print(f"substance of kept: {summary['substance_of_kept']}")
    print(f"statuses: {summary['statuses']}")
    print("\ntop destinations:")
    for name, n in sorted(summary["destinations"].items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {name:16} {n:4}")

    if args.against_gold:
        from .evaluate import load_labels

        gold = {
            r["domain"]: r["expected"] for r in load_labels(Path(args.against_gold))
        }
        scored = score_against_gold(verdicts, gold)
        report["against_gold"] = scored
        print(
            f"\nagainst {args.against_gold}: {scored['agreement']:.3f} on "
            f"{scored['scored']} domains, 95% CI {scored['ci95']}"
        )
        for miss in scored["disagreements"][:12]:
            print(f"  {miss['domain']:24} gold={miss['expected']:14} llm={miss['llm']}")

    if args.emit_sheet:
        corpus_text: dict[str, str] = {}
        if args.corpus:
            from .train_text import read_jsonl

            for name in SPLITS:
                for record in read_jsonl(Path(args.corpus) / f"{name}.jsonl"):
                    corpus_text[record["domain"]] = record["text"]
        drawn = emit_sheet(
            verdicts,
            corpus_text,
            Path(args.emit_sheet),
            args.sample_per_cell,
            args.seed,
        )
        report["sheet"] = drawn
        print(f"\nwrote {args.emit_sheet}: {sum(drawn.values())} rows -- {drawn}")
        print("  fill in human_label, then re-run with --adjudicated")

    if args.adjudicated:
        cells = summarize(verdicts)["cells"]
        collapsed: Counter = Counter()
        for cell, n in cells.items():
            if cell.startswith("move:") and cell not in (
                "move:status",
                "move:shopping",
            ):
                collapsed["move:other"] += n
            else:
                collapsed[cell] += n
        scored = score_sheet(Path(args.adjudicated), verdicts, dict(collapsed))
        report["adjudication"] = scored
        print(f"\nadjudicated {scored['adjudicated']} rows")
        for cell, stats in sorted(scored["per_cell"].items()):
            print(
                f"  {cell:18} n={stats['n']:3}  llm {stats['llm_agreement']:.3f} "
                f"{stats['llm_ci95']}  old-label {stats['old_label_agreement']:.3f}"
            )
        print(
            f"\nweighted: llm {scored['weighted_llm_agreement']:.3f} vs "
            f"old label {scored['weighted_old_label_agreement']:.3f}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
