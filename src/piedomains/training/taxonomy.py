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

**`redirector` is excluded for a different reason, and it is worth being precise about
which.** It is not a bad category -- it is a category whose *labels* have gone stale, and
those need opposite remedies.

Shallalist defines it as "sites that actively help to bypass url filters by accepting urls
via webform", and the domains bear that out: `hotspotshield.com`, `hidemyass.com`,
`anonymizer.com`, `4everproxy.com`. Curlie files 13 of the 14 Tranco-ranked ones under
`Computers`. A proxy site has a form, instructions and plenty of text, so it passes the
visibility rule comfortably. An earlier note in this file claimed it had "no content by
construction" -- that reads the name as an HTTP 301 and is simply wrong.

What disqualifies it is measured. Sampling 400 of them out of the corpus tarball: 46% are
under the token floor, 19% are parked, 7% are bot-walled, and 27% are usable -- of which
**six of twelve sampled were not proxy sites at all**, but a Chinese glassware
manufacturer, Japanese adult manga, an empty WordPress install and an article about
internet exchange points. Cheap TLDs get recycled, and by the 2022 scrape a large share of
these domains belonged to someone else entirely. Training on it would teach the model that
custom printed glassware means proxy.

`anonvpn` fails an even earlier test: it is not a list of websites. Of its 7,001 entries,
94% are bare IP addresses and 4% are residential reverse-DNS names -- Tor exit nodes and
open-proxy endpoints. **Exactly one has a fetched page.** This is a domain classifier, and
there is no homepage to read at `67.164.45.55`.

**Split, because collapsing destroyed the labels.** `recreation` is 98%
travel-or-sports (278,346 domains of 283,587; then martialarts 2,132, restaurants 1,423,
humor 984, wellness 702). It was never a category, and no annotator could apply it
consistently — the reproducibility test fails outright. The sub-labels already exist in
the mirror; the original run threw them away.

**Merged, because they are one thing.** `porn`, `sex` and `models` compete for the same
pages. One `adult` label with its own threshold is both more learnable and more useful,
since the safety decision wants recall rather than a three-way split.

**A second round of the same rule.** Measured on 4,673 held-out documents disjoint from
the v0.12 training set, three surviving splits ask about *delivery mechanism or legality*
rather than topic, and a page states none of them:

===================================  ==============================  =========================
split                                what it actually asks           cost
===================================  ==============================  =========================
`games-misc` / `games-online`        how you play the game           games-misc loses 34.4%
`radiotv` / `webradio`               how it is broadcast             webradio loses 31.8%
`downloads` / `warez`                whether the download is legal   warez loses 30.8%
===================================  ==============================  =========================

A page about a game rarely says whether you play it in a browser; a station's homepage
reads the same whether it also transmits over the air; and a warez site does not announce
its illegality -- that is a licensing fact about the files, not a property of the text.
Collapsing the three is worth **+0.023 accuracy and +0.023 macro-F1**, measured by
relabelling the shipped model's own predictions, which is a floor rather than an estimate:
it does not include the capacity the model currently spends failing to learn the
distinction.

`aggressive` is excluded outright on the evidence rather than the rule: 9 held-out
documents, recall 0.111, and the pages that *do* get labelled `aggressive` are politics
and humor. There is no support to learn it from and macro-F1 weights it the same as
`parked`'s 378.
"""

from __future__ import annotations

__all__ = [
    "EXCLUDED",
    "MERGED",
    "MERGED_PATHS",
    "SPLIT_PARENTS",
    "map_category",
    "target_classes",
]

#: Not visible in page text, so not learnable from it. `redirector` is here because a
#: redirector page has no content at all -- it belongs in `outcomes.py`.
#:
#: `aggressive` is here for a different reason: not that it is invisible, but that there
#: is too little of it to learn. 9 held-out documents, recall 0.111, and the pages that do
#: attract the label are politics and humor.
EXCLUDED: frozenset[str] = frozenset(
    {"adv", "tracker", "spyware", "redirector", "aggressive"}
)

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
#: `sex/education` is redirected to `education` rather than inheriting `adult`. It is
#: instructional content, and filters that conflate sexual health with adult content
#: block exactly the resources people most need -- Shallalist's own description of the
#: category concedes it "can be misdetected as porn". It is 166 domains, so this moves
#: no metric; it is simply correct.
PROMOTED: dict[str, str] = {
    "finance/realestate": "realestate",
    "sex/education": "education",
}

#: Distinct Shallalist categories describing one thing. Collapsing them removes a
#: three-way choice no annotator makes consistently, and lets the result be tuned for
#: recall as a single safety signal.
MERGED: dict[str, str] = {
    "porn": "adult",
    "sex": "adult",
    "models": "adult",
    # A web-only station and a broadcast one produce the same homepage. The split asks
    # how the signal reaches you, which the page does not say.
    "webradio": "radiotv",
    # Warez is downloads the licence did not permit. That is a fact about the files, not
    # about the text, and the page does not volunteer it.
    "warez": "downloads",
}

#: Merges of full nested paths, for children under :data:`SPLIT_PARENTS` that would
#: otherwise keep their own labels. Applied before the parent rules.
#:
#: Splitting games by whether they are played online asks about delivery, not subject --
#: and it is the single most expensive distinction left in the taxonomy, taking 34.4% of
#: `hobby/games-misc` with it. The result is `games`: the `hobby/` prefix is Shallalist's
#: filing structure, not part of the answer.
MERGED_PATHS: dict[str, str] = {
    "hobby/games-misc": "games",
    "hobby/games-online": "games",
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
        'sports'
        >>> map_category("finance/banking")
        'finance'
        >>> map_category("finance/realestate")
        'realestate'
        >>> map_category("porn")
        'adult'
        >>> map_category("hobby/games-online")
        'games'
        >>> map_category("webradio")
        'radiotv'
        >>> map_category("spyware") is None
        True
        >>> map_category("aggressive") is None
        True
    """
    if raw in MERGED_PATHS:
        return MERGED_PATHS[raw]
    if raw in PROMOTED:
        return PROMOTED[raw]

    parent = raw.split("/", 1)[0]
    if parent in EXCLUDED or parent in DROPPED_BY_NAME:
        return None
    if parent in MERGED:
        return MERGED[parent]
    if parent in SPLIT_PARENTS:
        # The child alone. `hobby` and `recreation` are Shallalist's filing structure, not
        # part of the answer: a page is about cooking or about sports, and prefixing that
        # with the drawer it was found in tells a caller nothing. The children are unique
        # across the whole label space, so nothing is lost by dropping the parent.
        return raw.split("/", 1)[1]
    return parent


def target_classes(raw_categories: list[str]) -> list[str]:
    """Resolve a list of raw categories to the sorted set of training labels.

    Args:
        raw_categories: Raw Shallalist category names.

    Returns:
        list[str]: Sorted, de-duplicated training labels.

    Example:
        >>> target_classes(["porn", "sex", "spyware", "recreation/sports"])
        ['adult', 'sports']
    """
    mapped = {map_category(c) for c in raw_categories}
    return sorted(c for c in mapped if c is not None)
