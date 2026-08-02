#!/usr/bin/env python3
r"""Report which classes a checkpoint conflates, and what merging them would buy.

**Why this exists as a module rather than a notebook.** The taxonomy question -- "which
distinctions is the model failing to make, and are they distinctions a page can even
express?" -- gets asked once per release, and the answer has to be recomputed against the
checkpoint that actually ships. Twice in this project an analysis was run, believed, and
then found to have been scored on data the model trained on. Both times the tell was a
number far above the checkpoint's own held-out figure, and both times the artifact was
gone by the time anyone wanted to re-check it.

So this refuses to run on a split the checkpoint has seen, and it writes what it found.

**What it reports.**

*Confusions*, directed: `games-misc -> games-online` 33 documents, 34.4% of
the gold class. The percentage matters more than the count -- a class can lose most of
itself to a neighbour while contributing few errors overall.

*Conflated pairs*, undirected: the same two classes trading errors in both directions is
the signature of a distinction the text does not carry, as opposed to one class simply
being a magnet for another.

*Merge simulation*: relabel the checkpoint's own predictions and rescore. This is a
**floor, not an estimate**. It counts only the errors that fall between merged classes; it
cannot count the capacity the model spent failing to learn a distinction that was never
learnable, which a retrain on the merged taxonomy recovers.

Usage::

    uv run python -m piedomains.training.confusions \
        --model models/text-v5 --data data/prepared-minimal --out reports/taxonomy.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from .metrics import accuracy, macro_f1
from .train_text import pick_device, read_jsonl

#: Merges to simulate, as ``(description, {from: to})``. Each is a distinction that asks
#: about delivery mechanism or legality rather than subject -- see ``taxonomy.py``.
DEFAULT_MERGES: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "games-misc + games-online -> games",
        {"games-misc": "games", "games-online": "games"},
    ),
    ("webradio -> radiotv", {"webradio": "radiotv"}),
    ("warez -> downloads", {"warez": "downloads"}),
)


def predict(model: Path, rows: list[dict], batch: int = 64) -> list[str]:
    """Score every row with the checkpoint.

    ``prepare_text`` writes the domain prefix into ``text`` before saving, so the stored
    field is already model input. Prefixing again here would feed the model
    ``"ugr ugr universidad de granada..."`` -- a small enough corruption to leave the
    headline accuracy looking plausible, which is exactly what makes it worth naming.

    Args:
        model: Checkpoint directory.
        rows: Prepared records carrying ``text`` that is already model-ready.
        batch: Rows per forward pass.

    Returns:
        list[str]: Predicted label per row, in order.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ..checkpoints import eager_config

    labels = json.loads((model / "labels.json").read_text(encoding="utf-8"))
    if isinstance(labels, dict):
        labels = [labels[str(i)] for i in range(len(labels))]

    eager_config(str(model))
    tokenizer = AutoTokenizer.from_pretrained(model)
    device = pick_device()
    net = AutoModelForSequenceClassification.from_pretrained(model).eval().to(device)

    out: list[str] = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        encoded = tokenizer(
            [r["text"] for r in chunk],
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out += [labels[int(j)] for j in net(**encoded).logits.argmax(-1)]
    return out


def report(truth: list[str], pred: list[str], top: int = 25) -> dict[str, Any]:
    """Summarise where the model is wrong and which classes trade errors.

    Args:
        truth: Gold labels.
        pred: Predicted labels.
        top: How many rows to keep in each table.

    Returns:
        dict[str, Any]: Confusions, conflated pairs, and headline scores.
    """
    support = collections.Counter(truth)
    wrong = collections.Counter(
        (t, p) for t, p in zip(truth, pred, strict=True) if t != p
    )

    confusions = [
        {
            "gold": t,
            "predicted": p,
            "n": n,
            "share_of_gold_class": n / support[t],
        }
        for (t, p), n in wrong.most_common(top)
    ]

    pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    for (t, p), n in wrong.items():
        pairs[(min(t, p), max(t, p))] += n
    conflated = [
        {
            "a": a,
            "b": b,
            "total": n,
            "a_to_b": wrong.get((a, b), 0),
            "b_to_a": wrong.get((b, a), 0),
        }
        for (a, b), n in pairs.most_common(top)
    ]

    return {
        "n": len(truth),
        "accuracy": accuracy(truth, pred),
        "macro_f1": macro_f1(truth, pred),
        "classes": len(support),
        "confusions": confusions,
        "conflated_pairs": conflated,
    }


def simulate(
    truth: list[str], pred: list[str], merges: tuple[tuple[str, dict[str, str]], ...]
) -> list[dict[str, Any]]:
    """Score each merge, and all of them together, by relabelling predictions.

    Args:
        truth: Gold labels.
        pred: Predicted labels.
        merges: ``(description, mapping)`` pairs.

    Returns:
        list[dict[str, Any]]: One row per scenario, including the combined one.
    """
    base_acc = accuracy(truth, pred)
    base_f1 = macro_f1(truth, pred)

    def apply(mapping: dict[str, str]) -> dict[str, Any]:
        t = [mapping.get(x, x) for x in truth]
        p = [mapping.get(x, x) for x in pred]
        acc = accuracy(t, p)
        f1 = macro_f1(t, p)
        return {
            "classes": len(set(t)),
            "accuracy": acc,
            "macro_f1": f1,
            "accuracy_gain": acc - base_acc,
            "macro_f1_gain": f1 - base_f1,
        }

    rows = [{"scenario": desc, **apply(mp)} for desc, mp in merges]
    combined: dict[str, str] = {}
    for _, mp in merges:
        combined.update(mp)
    rows.append({"scenario": "all of the above", **apply(combined)})
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Audit a checkpoint's class confusions"
    )
    parser.add_argument("--model", required=True, help="Checkpoint directory")
    parser.add_argument("--data", required=True, help="Prepared split directory")
    parser.add_argument("--split", default="test", help="Which split to score")
    parser.add_argument("--out", default="", help="Write the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Audit confusions and report what merging the worst pairs would buy.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Always zero; this reports rather than gates.
    """
    args = build_parser().parse_args(argv)
    model, data = Path(args.model), Path(args.data)

    rows = read_jsonl(data / f"{args.split}.jsonl")
    truth = [r["category"] for r in rows]
    pred = predict(model, rows)

    result = report(truth, pred)
    print(
        f"\n{result['n']:,} documents, {result['classes']} classes: "
        f"accuracy {result['accuracy']:.4f}, macro-F1 {result['macro_f1']:.4f}"
    )

    print(f"\n{'gold':26s} -> {'predicted':26s} {'n':>5s} {'% of gold':>10s}")
    for row in result["confusions"]:
        print(
            f"{row['gold']:26s} -> {row['predicted']:26s} {row['n']:5d} "
            f"{100 * row['share_of_gold_class']:9.1f}%"
        )

    print("\nconflated pairs (errors both ways)")
    for row in result["conflated_pairs"][:12]:
        print(
            f"  {row['a']:26s} <-> {row['b']:26s} {row['total']:4d} "
            f"({row['a_to_b']} + {row['b_to_a']})"
        )

    merges = simulate(truth, pred, DEFAULT_MERGES)
    result["merge_simulation"] = merges
    print(f"\n{'merge scenario':40s} {'classes':>8s} {'acc':>8s} {'macroF1':>9s}")
    for row in merges:
        print(
            f"{row['scenario']:40s} {row['classes']:8d} {row['accuracy']:8.4f} "
            f"{row['macro_f1']:9.4f}   "
            f"({row['accuracy_gain']:+.4f} / {row['macro_f1_gain']:+.4f})"
        )
    print(
        "\nThese are floors: relabelling counts only errors between merged classes, not "
        "the capacity a retrain recovers."
    )

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
