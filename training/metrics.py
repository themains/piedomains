"""Classification metrics for evaluating piedomains models.

Kept dependency-free (numpy only) so the harness runs anywhere the package
runs, and importable from both ``training/evaluate.py`` and the test suite.

Example:
    >>> truth = ["news", "news", "shopping"]
    >>> pred = ["news", "shopping", "shopping"]
    >>> round(macro_f1(truth, pred), 3)
    0.75
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

__all__ = [
    "confusion_pairs",
    "expected_calibration_error",
    "macro_f1",
    "per_class_report",
    "summarize",
]


def _counts(truth: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    """Tally true/false positives and negatives per class.

    Args:
        truth: Gold labels.
        pred: Predicted labels, aligned with ``truth``.

    Returns:
        dict: Per-class ``tp``/``fp``/``fn`` counts.

    Raises:
        ValueError: If the two label sequences differ in length.
    """
    if len(truth) != len(pred):
        raise ValueError("truth and pred must be the same length")
    labels = set(truth) | set(pred)
    out = {c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in labels}
    for t, p in zip(truth, pred, strict=True):
        out[t]["support"] += 1
        if t == p:
            out[t]["tp"] += 1
        else:
            out[p]["fp"] += 1
            out[t]["fn"] += 1
    return out


def per_class_report(truth: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    """Compute precision, recall and F1 for every class.

    Args:
        truth: Gold labels.
        pred: Predicted labels.

    Returns:
        dict: Class name -> precision/recall/f1/support.
    """
    report: dict[str, dict[str, float]] = {}
    for cls, c in sorted(_counts(truth, pred).items()):
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        report[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": c["support"],
        }
    return report


def macro_f1(truth: list[str], pred: list[str]) -> float:
    """Compute the unweighted mean F1 across classes that appear in ``truth``.

    Averaging only over classes with support avoids rewarding a model for
    classes the eval set never exercises.

    Args:
        truth: Gold labels.
        pred: Predicted labels.

    Returns:
        float: Macro-averaged F1.
    """
    report = per_class_report(truth, pred)
    present = [m["f1"] for m in report.values() if m["support"] > 0]
    return sum(present) / len(present) if present else 0.0


def expected_calibration_error(
    correct: list[bool], confidence: list[float], bins: int = 10
) -> float:
    """Compute expected calibration error.

    ECE answers "when the model says 0.8, is it right 80% of the time?" — the
    question the unnormalized isotonic calibration made unanswerable.

    Args:
        correct: Whether each prediction was right.
        confidence: Reported confidence for each prediction.
        bins: Number of equal-width confidence bins.

    Returns:
        float: Weighted average gap between confidence and accuracy.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(correct) != len(confidence):
        raise ValueError("correct and confidence must be the same length")
    if not correct:
        return 0.0
    total = len(correct)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [
            i
            for i, c in enumerate(confidence)
            if (c > lo or (b == 0 and c >= lo)) and c <= hi
        ]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidence[i] for i in idx) / len(idx)
        ece += (len(idx) / total) * abs(acc - conf)
    return ece


def confusion_pairs(
    truth: list[str], pred: list[str], top: int = 10
) -> list[tuple[str, str, int]]:
    """List the most frequent misclassifications.

    Args:
        truth: Gold labels.
        pred: Predicted labels.
        top: How many pairs to return.

    Returns:
        list: ``(true_label, predicted_label, count)``, most frequent first.
    """
    pairs = Counter((t, p) for t, p in zip(truth, pred, strict=True) if t != p)
    return [(t, p, n) for (t, p), n in pairs.most_common(top)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn evaluated rows into a metrics summary.

    Args:
        rows: Dicts with ``expected``, ``predicted`` and optionally
            ``confidence``. Rows whose ``predicted`` is ``None`` count as
            errors rather than being silently dropped.

    Returns:
        dict: Headline metrics, per-class report and top confusions.
    """
    scored = [r for r in rows if r.get("predicted") is not None]
    truth = [r["expected"] for r in scored]
    pred = [r["predicted"] for r in scored]
    correct = [t == p for t, p in zip(truth, pred, strict=True)]
    conf = [float(r.get("confidence") or 0.0) for r in scored]

    return {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_unclassified": len(rows) - len(scored),
        "accuracy": (sum(correct) / len(correct)) if correct else 0.0,
        "macro_f1": macro_f1(truth, pred),
        "ece": expected_calibration_error(correct, conf),
        "mean_confidence": (sum(conf) / len(conf)) if conf else 0.0,
        "per_class": per_class_report(truth, pred),
        "top_confusions": confusion_pairs(truth, pred),
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Render a summary as a readable block.

    Args:
        summary: Output of :func:`summarize`.

    Returns:
        str: Multi-line human-readable report.
    """
    lines = [
        f"scored {summary['n_scored']}/{summary['n_total']} "
        f"({summary['n_unclassified']} produced no label)",
        f"accuracy   {summary['accuracy']:.3f}",
        f"macro-F1   {summary['macro_f1']:.3f}",
        f"ECE        {summary['ece']:.3f}  "
        f"(mean confidence {summary['mean_confidence']:.3f})",
        "",
        f"{'class':16s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>4s}",
    ]
    for cls, m in summary["per_class"].items():
        if not m["support"]:
            continue
        lines.append(
            f"{cls:16s} {m['precision']:6.3f} {m['recall']:6.3f} "
            f"{m['f1']:6.3f} {int(m['support']):4d}"
        )
    if summary["top_confusions"]:
        lines += ["", "most common confusions (true -> predicted):"]
        lines += [f"  {t:16s} -> {p:16s} {n}" for t, p, n in summary["top_confusions"]]
    return "\n".join(lines)


def is_finite(value: float) -> bool:
    """Return True when a metric is a usable finite number.

    Args:
        value: Metric value to check.

    Returns:
        bool: True if finite.
    """
    return not (math.isnan(value) or math.isinf(value))
