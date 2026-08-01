#!/usr/bin/env python3
"""Turn a probability distribution into the labels worth reporting.

**Why more than one.** The label set is not mutually exclusive, and cannot be made so
while it stays a flat list: `shopping` says what a site *does*, `automobile` says what it
is *about*, and a car dealership is honestly both. Sorting all 44 classes by which question
they answer, the error rate tracks the axis -- 1% for the two status labels, 15% for the 24
topical ones, 31% for the 13 that describe what a site *is*. Taking the argmax forces those
axes to compete for one slot and discards a true answer whenever they overlap.

**What this buys, measured on 4,673 held-out documents.** The chance the correct label is
among those reported:

===================  ========  ===================
threshold            recall    labels per domain
===================  ========  ===================
argmax only            79.7%   1.00
p >= 0.15              84.4%   1.19
p >= 0.10 (default)    86.6%   1.35
p >= 0.05              90.1%   1.86
===================  ========  ===================

At the default, 65% of domains still get exactly one label; only the ambiguous ones get
two. No retraining was needed for any of this -- the model already computed the
distribution and the argmax threw it away.

**What it does not buy, and this matters.** Those are *recall* figures. Reporting more
labels trivially raises the chance of covering the gold one, and the evaluation gold is
single-label, so there is no way to tell from it whether the second label is *correct*.
When a domain labelled `automobile` also gets `shopping`, this data cannot say whether that
is a car dealership or noise. Validating that needs a gold set annotated with every
applicable label, which does not exist yet.
"""

from __future__ import annotations

__all__ = ["top_labels"]


def top_labels(
    scores: dict[str, float], threshold: float
) -> list[dict[str, float | str]]:
    """Select the labels worth reporting, highest first.

    The argmax is always included even when it falls below the threshold: a classified row
    reporting no labels would be a worse answer than a weak one, and the caller can read
    the probability and decide.

    Args:
        scores: Label to probability, summing to 1.
        threshold: Probability floor for inclusion beyond the argmax.

    Returns:
        list[dict[str, float | str]]: ``{"category", "probability"}`` in descending
        probability order, always at least one entry. Empty only if ``scores`` is empty.
    """
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    kept = [(name, p) for name, p in ranked if p >= threshold] or ranked[:1]
    return [{"category": name, "probability": p} for name, p in kept]
