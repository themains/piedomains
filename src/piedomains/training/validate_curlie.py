#!/usr/bin/env python3
r"""Score the text model against Curlie's human labels on popular domains.

**Why this exists.** Every accuracy number in this project so far comes from Shallalist —
the same source the model trains on, with the same selection bias (59% of Shallalist is
blockable categories, `porn` alone is 53%) and the same 2022 vintage. A held-out split of
a biased corpus measures how well the model learned that corpus, not whether it works.
`tests/eval/labels.csv` is independent but is 71 domains I labelled myself.

Curlie is a different thing: human editors, curated for a browsable directory rather than
a blocklist, and released as a public dataset with the Homepage2Vec paper. Crossed with
Tranco it gives ~38k *popular* domains carrying labels no part of this project produced.

**The comparison is coarse, deliberately.** Our 47 classes are mapped down to Curlie's 14,
never the reverse: coarsening a prediction is lossy but honest, whereas splitting a gold
label into guesses invents ground truth. Any prediction with no Curlie equivalent is
reported separately rather than counted as wrong.

**Read the result as agreement, not accuracy.** Curlie's top level is not a topic
taxonomy — it files google.com, facebook.com and youtube.com under `Computers`, because
the directory path is `Computers/Internet/...`. So a `socialnet` prediction for
facebook.com is right by our taxonomy and scores as `Computers` here only because the
mapping says so. Disagreement is as likely to be a taxonomy mismatch as a model error,
which is exactly why the per-class table matters more than the headline.

Usage::

    uv run python training/validate_curlie.py --paired /tmp/curlie/paired.json \\
        --limit 200 --max-rank 100000 --out /tmp/curlie_agreement.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: Our 44 classes to Curlie's 14. Only mappings a Curlie editor would plausibly agree
#: with; everything else is left out and reported as unmappable rather than scored.
#:
#: The awkward ones, stated rather than hidden: Curlie has no Adult branch in this release,
#: so `adult`, `dating`, `drugs`, `weapons` and `gamble` have no target and are excluded.
#: `parked` and `unavailable` are excluded too, and for a better reason -- a domain with no
#: site behind it has no Curlie topic to be right or wrong about. `searchengines`, `webmail`, `socialnet`, `isp`, `hosting` and
#: friends go to `Computers` because that is where the directory puts them -- a site-type
#: judgement, not a topic one.
TO_CURLIE: dict[str, str] = {
    "news": "News",
    "radiotv": "News",
    "movies": "Arts",
    "music": "Arts",
    "shopping": "Shopping",
    "finance": "Business",
    "jobsearch": "Business",
    "realestate": "Business",
    "automobile": "Recreation",
    "travel": "Recreation",
    "humor": "Recreation",
    "restaurants": "Recreation",
    "wellness": "Health",
    "sports": "Sports",
    "hospitals": "Health",
    "education": "Reference",
    "library": "Reference",
    "science": "Science",
    "government": "Society",
    "politics": "Society",
    "religion": "Society",
    "military": "Society",
    "forum": "Society",
    "fortunetelling": "Society",
    "cooking": "Home",
    "gardening": "Home",
    "pets": "Home",
    "homestyle": "Home",
    "games": "Games",
    "searchengines": "Computers",
    "webmail": "Computers",
    "socialnet": "Computers",
    "isp": "Computers",
    "imagehosting": "Computers",
    "downloads": "Computers",
    "urlshortener": "Computers",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Validate against Curlie labels")
    parser.add_argument(
        "--paired", required=True, help="JSON list of [domain, curlie_class, rank]"
    )
    parser.add_argument("--limit", type=int, default=200, help="Domains to score")
    parser.add_argument("--max-rank", type=int, default=100_000, help="Tranco cut-off")
    parser.add_argument("--cache-dir", default="cache/curlie")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="", help="Write the report as JSON")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Training-domain manifest to exclude from the sample, repeatable. "
            "Accepts a prepared JSONL split or a checkpoint's train_domains.json"
        ),
    )
    return parser


def collapse_legacy(label: str) -> str:
    """Map a label from an older checkpoint onto the current label space.

    A 47-class checkpoint still emits `games-misc`, `webradio` and `warez`, none of
    which are keys in :data:`TO_CURLIE` any more. Without this they would score as
    *unmappable* rather than as `games`, `radiotv` and `downloads` -- which would
    quietly penalise the older model for a taxonomy change rather than measure it, and the
    whole point of running this is to compare the two fairly.

    Delegates to :func:`piedomains.labels.project`, which the serving path also uses. A
    second hand-rolled copy is precisely how this function came to disagree with the
    taxonomy once already: it handled the merges but not the split-parent flattening, so
    `cooking`, `sports`, `travel`, `pets` and five more silently became unmappable and
    dropped out of the benchmark while it went on printing a number.

    Args:
        label: A predicted label, possibly from a superseded label space.

    Returns:
        str: The equivalent label in the current space, unchanged if already current.
        Excluded classes map to themselves, so they are reported as unmappable rather
        than vanishing.
    """
    from ..labels import project

    names, _ = project([label])
    return names[0] if names else label


def load_excluded(paths: list[str]) -> set[str]:
    """Read domains that must not appear in the sample.

    Curlie is only independent evidence for a model that never trained on it. Two
    checkpoints being compared have different training sets, so the sample has to exclude
    the union -- otherwise the comparison silently favours whichever model saw more of it.

    Args:
        paths: Prepared JSONL splits, or checkpoint ``train_domains.json`` files.

    Returns:
        set[str]: Every domain to withhold.
    """
    excluded: set[str] = set()
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        if path.endswith(".jsonl"):
            excluded |= {
                json.loads(line)["domain"] for line in text.splitlines() if line
            }
        else:
            excluded |= set(json.loads(text))
    return excluded


def sample(
    paired: list[list],
    limit: int,
    max_rank: int,
    seed: int,
    excluded: set[str] | None = None,
) -> list[tuple[str, str, int]]:
    """Take a class-balanced sample of popular domains.

    Stratifying matters more than usual here: Curlie's English subset is 24% ``Business``
    and 0.8% ``News``, so a uniform sample would score mostly one class and macro-F1 would
    be dominated by classes with two or three examples.

    Args:
        paired: ``[domain, curlie_class, rank]`` triples.
        limit: Total domains to return.
        max_rank: Only consider domains at least this popular.
        seed: Sampling seed.
        excluded: Domains to withhold, typically the union of the training sets of every
            checkpoint that will be scored.

    Returns:
        list[tuple[str, str, int]]: The sampled triples.
    """
    rng = random.Random(seed)  # noqa: S311 -- sampling, not security
    withheld = excluded or set()
    by_class: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for domain, label, rank in paired:
        if rank <= max_rank and domain not in withheld:
            by_class[label].append((domain, label, rank))

    per_class = max(1, limit // max(1, len(by_class)))
    chosen: list[tuple[str, str, int]] = []
    for label in sorted(by_class):
        pool = by_class[label]
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])
    rng.shuffle(chosen)
    return chosen[:limit]


def main(argv: list[str] | None = None) -> int:
    """Classify the sample and report agreement with Curlie.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status.
    """
    args = build_parser().parse_args(argv)
    from ..api import DomainClassifier

    paired = json.loads(Path(args.paired).read_text(encoding="utf-8"))
    excluded = load_excluded(args.exclude)
    if excluded:
        print(f"excluding {len(excluded):,} training domains from the sample")
    chosen = sample(paired, args.limit, args.max_rank, args.seed, excluded)
    print(f"scoring {len(chosen)} domains (Tranco <= {args.max_rank:,})")
    print("class balance:", dict(Counter(c for _, c, _ in chosen)))

    gold = {d: c for d, c, _ in chosen}
    run = DomainClassifier(cache_dir=args.cache_dir).classify_by_text(
        [d for d, _, _ in chosen]
    )

    rows = []
    for result in run["results"]:
        predicted = result.get("category")
        rows.append(
            {
                "domain": result["domain"],
                "curlie": gold.get(result["domain"]),
                "predicted": predicted,
                "collapsed": collapse_legacy(predicted) if predicted else None,
                "mapped": TO_CURLIE.get(collapse_legacy(predicted))
                if predicted
                else None,
                "confidence": result.get("confidence"),
                "status": result.get("status"),
                "error_code": result.get("error_code"),
            }
        )

    scored = [r for r in rows if r["mapped"] and r["curlie"]]
    unmappable = [r for r in rows if r["predicted"] and not r["mapped"]]
    failed = [r for r in rows if not r["predicted"]]
    hits = [r for r in scored if r["mapped"] == r["curlie"]]

    print(f"\n{'':22s} {'n':>5s}")
    print(f"{'requested':22s} {len(rows):>5d}")
    print(f"{'classified':22s} {len(rows) - len(failed):>5d}")
    print(f"{'fetch/infer failed':22s} {len(failed):>5d}")
    print(f"{'predicted, unmappable':22s} {len(unmappable):>5d}")
    print(f"{'scored against Curlie':22s} {len(scored):>5d}")
    if scored:
        print(f"\nagreement: {len(hits)}/{len(scored)} = {len(hits) / len(scored):.3f}")

    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in scored:
        per_class[r["curlie"]][1] += 1
        if r["mapped"] == r["curlie"]:
            per_class[r["curlie"]][0] += 1
    print(f"\n{'curlie class':16s} {'hit':>4s} {'n':>4s}  rate")
    for label in sorted(per_class):
        hit, total = per_class[label]
        print(f"{label:16s} {hit:>4d} {total:>4d}  {hit / total:.2f}")

    if unmappable:
        print("\nmost common unmappable predictions:")
        for name, count in Counter(r["predicted"] for r in unmappable).most_common(8):
            print(f"  {name:24s} {count}")

    report = {
        "requested": len(rows),
        "scored": len(scored),
        "agreement": (len(hits) / len(scored)) if scored else None,
        "unmappable": len(unmappable),
        "failed": len(failed),
        "per_class": {k: {"hit": v[0], "n": v[1]} for k, v in per_class.items()},
        "rows": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
