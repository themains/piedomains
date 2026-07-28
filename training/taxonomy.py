#!/usr/bin/env python3
"""Map raw Shallalist categories onto the label space the model is trained on.

The rule that decides every case is a single question:

    **Is it visible in the page text?**

* Visible → it belongs in the label space.
* Not visible → excluded from the model entirely.
* Nothing to read → an *outcome* (:mod:`piedomains.outcomes`), not a category.

Applying it to the 39 shipped classes produces three changes, none of which need new
annotation — they rearrange labels we already have.

**Excluded, because a page does not say them.** `adv`, `tracker` and `spyware` are
properties of who runs a site and how it is monetised. A homepage selling handmade goods
reads identically whether or not the operator also runs trackers. These three are 4.1% of
the training set and sit behind a third of the evaluation errors: `amazon→adv`,
`etsy→spyware`, `duckduckgo→spyware`, `paypal→adv`. Cloudflare reaches the same
conclusion from the other direction — it runs these off domain age, reputation and DNS
behaviour, and never off content.

`redirector` is excluded for the adjacent reason: such a page has no content by
construction, so it is a fetch outcome rather than a topic.

**Split, because collapsing destroyed the labels.** `recreation` is 98%
travel-or-sports (278,346 domains of 283,587; then martialarts 2,132, restaurants 1,423,
humor 984, wellness 702). It was never a category, and no annotator could apply it
consistently — the reproducibility test fails outright. The sub-labels already exist in
the mirror; the original run threw them away.

**Merged, because they are one thing.** `porn`, `sex` and `models` compete for the same
pages. One `adult` label with its own threshold is both more learnable and more useful,
since the safety decision wants recall rather than a three-way split.
"""

from __future__ import annotations

__all__ = [
    "EXCLUDED",
    "MERGED",
    "SPLIT_PARENTS",
    "map_category",
    "target_classes",
]

#: Not visible in page text, so not learnable from it. `redirector` is here because a
#: redirector page has no content at all -- it belongs in `outcomes.py`.
EXCLUDED: frozenset[str] = frozenset({"adv", "tracker", "spyware", "redirector"})

#: Categories the original run dropped by name before any thresholding.
DROPPED_BY_NAME: frozenset[str] = frozenset({"chat", "hacking", "webtv"})

#: Parents whose children carry real, separable meaning and must not be collapsed.
#: Everything else collapses, because splitting it produces classes too small to learn.
SPLIT_PARENTS: frozenset[str] = frozenset({"recreation", "hobby"})

#: Children that must not inherit their parent's label.
#:
#: `finance/realestate` is promoted because every surveyed taxonomy -- IAB, Cloudflare,
#: WebOrganizer -- treats real estate as top-level.
#:
#: `sex/education` is redirected to `health` because sexual health content is not adult
#: content. Filters that conflate the two block exactly the resources people most need,
#: and Shallalist's own description of the category says it "can be misdetected as
#: porn". It is 166 domains, so this moves no metric -- it is correct rather than
#: material.
PROMOTED: dict[str, str] = {
    "finance/realestate": "realestate",
    "sex/education": "hospitals",
}

#: Distinct Shallalist categories describing one thing. Collapsing them removes a
#: three-way choice no annotator makes consistently, and lets the result be tuned for
#: recall as a single safety signal.
MERGED: dict[str, str] = {
    "porn": "adult",
    "sex": "adult",
    "models": "adult",
}


def map_category(raw: str) -> str | None:
    """Map one raw Shallalist category to its training label.

    Args:
        raw: A Shallalist category, possibly nested (``recreation/sports``).

    Returns:
        str | None: The label to train on, or ``None`` when the category is
        excluded from the model.

    Example:
        >>> map_category("recreation/sports")
        'recreation/sports'
        >>> map_category("finance/banking")
        'finance'
        >>> map_category("finance/realestate")
        'realestate'
        >>> map_category("porn")
        'adult'
        >>> map_category("spyware") is None
        True
    """
    if raw in PROMOTED:
        return PROMOTED[raw]

    parent = raw.split("/", 1)[0]
    if parent in EXCLUDED or parent in DROPPED_BY_NAME:
        return None
    if parent in MERGED:
        return MERGED[parent]
    if parent in SPLIT_PARENTS:
        return raw
    return parent


def target_classes(raw_categories: list[str]) -> list[str]:
    """Resolve a list of raw categories to the sorted set of training labels.

    Args:
        raw_categories: Raw Shallalist category names.

    Returns:
        list[str]: Sorted, de-duplicated training labels.

    Example:
        >>> target_classes(["porn", "sex", "spyware", "recreation/sports"])
        ['adult', 'recreation/sports']
    """
    mapped = {map_category(c) for c in raw_categories}
    return sorted(c for c in mapped if c is not None)
