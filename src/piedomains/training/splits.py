#!/usr/bin/env python3
"""Decide which split a domain belongs to, from the domain alone.

**Why this exists.** Three times in this project a model was scored on data it had trained
on, and each time the number looked plausible enough to be believed for a while:

* image-only read **0.768** where the honest figure was 0.429
* image-only read **0.706** where the honest figure was 0.429
* text read **0.9357** where the honest figure was about 0.71 -- 79% of the "held-out"
  domains were in the text model's training set

All three trace to the same two lines. ``prepare_text.py`` shuffled its list of documents;
``prepare_images.py`` shuffled its list of screenshots. Different lists, so the same domain
landed in ``train`` on one side and ``test`` on the other. Re-preparing reshuffled
everything again, so a checkpoint's splits quietly stopped matching the files that shared
their name.

The response had been to add detectors -- ``--respect-splits``, ``align_splits``,
``refuse_if_leaky``, ``refuse_if_seen``, ``--assume-disjoint``, ``train_domains.json``,
``--exclude`` -- 43 references across 8 files, all guarding a mistake that should not have
been possible to make.

**Making the assignment a pure function of the domain removes the mistake instead of
detecting it.** There is no seed, no shuffle, no ordering, and no coordination between
scripts: text preparation, screenshot preparation and an evaluation written years later all
compute the same answer for the same domain. A domain cannot be in two splits, because
there is nowhere for a second opinion to come from.

**What it costs, stated rather than discovered later.** Hashing does not stratify by class,
so each class gets multinomial variance instead of an exact 80/10/10. `military`, at 115
documents, gets 11.5 +/- 3.2 test documents rather than exactly 11. That is the ordinary
trade: stratifying would require the preparers to agree on class counts before assigning,
which is coordination -- the thing being removed.
"""

from __future__ import annotations

import hashlib

__all__ = ["SPLITS", "TRAIN_PERCENT", "VAL_PERCENT", "is_held_out", "split_of"]

#: Split names, in the order the preparers write them.
SPLITS: tuple[str, str, str] = ("train", "val", "test")

#: Bucket boundaries, out of 100. Everything above :data:`VAL_PERCENT` is test.
TRAIN_PERCENT = 80
VAL_PERCENT = 90


def split_of(domain: str) -> str:
    """Return which split a domain belongs to.

    A pure function: the same domain always yields the same split, in every script, in
    every run, on every machine. Case and surrounding whitespace are normalised first so
    that ``CNN.com`` and ``cnn.com`` cannot disagree.

    SHA-256 rather than :func:`hash` because Python salts string hashing per process --
    ``hash("cnn.com")`` differs between runs, which would reintroduce exactly the
    instability this removes.

    Args:
        domain: The domain name.

    Returns:
        str: ``"train"``, ``"val"`` or ``"test"``.

    Example:
        >>> split_of("cnn.com") == split_of("  CNN.COM  ")
        True
        >>> split_of("cnn.com") in ("train", "val", "test")
        True
    """
    digest = hashlib.sha256(domain.strip().lower().encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < TRAIN_PERCENT:
        return "train"
    if bucket < VAL_PERCENT:
        return "val"
    return "test"


def is_held_out(domain: str) -> bool:
    """Report whether a domain is safe to evaluate on.

    The question every leak check used to ask, now answerable from the domain alone
    without consulting a manifest, a split file, or a checkpoint.

    Args:
        domain: The domain name.

    Returns:
        bool: True when the domain is not in the training split.

    Example:
        >>> is_held_out("cnn.com") == (split_of("cnn.com") != "train")
        True
    """
    return split_of(domain) != "train"
