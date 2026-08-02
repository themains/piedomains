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


def project(labels: list[str]) -> tuple[list[str], list[list[int]]]:
    """Map a checkpoint's labels onto the label space the package documents.

    A checkpoint outlives the taxonomy it was trained under. The shipped model emits 47
    labels -- `hobby/cooking`, `hobby/games-online`, `webradio`, `warez`, `aggressive` --
    while :data:`piedomains.constants.classes` documents 44 flat ones. Projecting at load
    time means callers get the documented names whatever checkpoint is in use, instead of
    the package describing a vocabulary its own model cannot speak.

    It also applies the three merges without a retrain. Each collapses a distinction the
    page does not state -- whether a game is played online, whether a station also
    broadcasts, whether a download was licensed -- worth +0.023 accuracy and +0.023
    macro-F1 measured by relabelling the model's own predictions.

    **Merged classes are returned as groups, not renamed keys.** Building a dict by
    renaming would keep whichever source came last and silently discard the other's
    probability mass, which looks exactly like a working model. The caller sums the group.

    `aggressive` is dropped: excluded from the taxonomy on 9 training documents and recall
    0.111, and the pages it does claim are politics and humor. Over 4,673 held-out
    documents it carries mean mass 0.004 and is the argmax 4 times -- but reaches 0.966 on
    one, so it is negligible on average and not in the tail. The caller renormalises.

    Args:
        labels: The checkpoint's own labels, in output order.

    Returns:
        tuple[list[str], list[list[int]]]: The projected labels, and per projected label
        the source indices whose probabilities sum into it. Dropped source labels appear
        in no group.

    Example:
        >>> project(["news", "webradio", "radiotv"])
        (['news', 'radiotv'], [[0], [1, 2]])
        >>> project(["hobby/cooking", "aggressive"])
        (['cooking'], [[0]])
    """
    from .training.taxonomy import EXCLUDED, MERGED, MERGED_PATHS, SPLIT_PARENTS

    def target(label: str) -> str | None:
        """Resolve one checkpoint label to its current name.

        Args:
            label: A label as the checkpoint emits it.

        Returns:
            str | None: The current name, or None when the class is excluded.
        """
        if label in MERGED_PATHS:
            return MERGED_PATHS[label]
        if label in MERGED:
            return MERGED[label]
        parent = label.split("/", 1)[0]
        if parent in EXCLUDED:
            return None
        # A split parent is filing structure, not part of the answer: `recreation/sports`
        # is reported as `sports`. The children are unique across the label space.
        if "/" in label and parent in SPLIT_PARENTS:
            return label.split("/", 1)[1]
        return label

    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        name = target(label)
        if name is None:
            continue
        if name not in groups:
            order.append(name)
            groups[name] = []
        groups[name].append(index)
    return order, [groups[name] for name in order]
