#!/usr/bin/env python3
r"""Turn the scraped HTML corpus into a labelled, split text dataset.

**The corpus is read straight out of the tarball.** ``shallalist_all.tar.gz`` is
18 GB compressed and expands past 47 GB, which does not fit on a normal disk
alongside everything else; streaming it costs nothing and removes the
constraint entirely. A directory tree is still accepted for a corpus that has
already been unpacked.

**Labels are not in the tarball.** It is flat -- ``shallalist_all/<domain>.html``
with no category anywhere in the path, and ``screenshot-index.tab`` carries only
``index,full_domain``. The original run fetched ``<category>/domains`` lists
from the ``cbuijs/shallalist`` GitHub mirror, and **that repository no longer
exists** (Shallalist was discontinued in 2022). ``Azothyran/ShallalistMirror``
is a surviving copy with all 74 category directories intact, and is what this
script reads. Fetched lists are cached, so the network is hit once.

Reproducing the original filtering matters: the shipped model has 39 classes,
not the 73 in shallalist_cats.txt, because training dropped `chat`, `hacking`
and `webtv` explicitly and then dropped every category with fewer than 100
documents (notebooks/04_train_model.ipynb).

Usage:
    python training/prepare_text.py --corpus data/corpus/shallalist_all.tar.gz \\
        --out data/prepared
    python training/prepare_text.py --corpus data/shallalist --out data/prepared \\
        --min-docs 100 --min-tokens 6
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from ..blocking import detect_block, looks_parked, looks_unavailable
from .splits import SPLITS, split_of
from .taxonomy import DROPPED_BY_NAME, map_category

#: Surviving Shallalist mirror. The one the original notebooks used
#: (``cbuijs/shallalist``) 404s -- the taxonomy was discontinued in 2022.
LABEL_MIRROR = "https://raw.githubusercontent.com/Azothyran/ShallalistMirror/master"


def load_category_map(cats_file: Path, cache: Path) -> dict[str, str]:
    """Build the ``domain -> category`` mapping the corpus is labelled with.

    A domain listed under more than one category is dropped rather than assigned
    arbitrarily: with one label per document, an ambiguous domain is a coin
    flip that shows up as noise in exactly the categories that overlap most.

    Args:
        cats_file: ``shallalist_cats.txt``, one category per line.
        cache: Where to keep the fetched lists so the mirror is hit once.

    Returns:
        dict[str, str]: Domain to category name.
    """
    import requests

    cache.mkdir(parents=True, exist_ok=True)
    categories = [c.strip() for c in cats_file.read_text().splitlines() if c.strip()]

    owners: dict[str, set[str]] = {}
    for category in categories:
        if category in DROPPED_BY_NAME:
            continue
        local = cache / f"{category.replace('/', '__')}.txt"
        if not local.exists():
            response = requests.get(f"{LABEL_MIRROR}/{category}/domains", timeout=60)
            if not response.ok:
                sys.stderr.write(f"  no domain list for {category}\n")
                local.write_text("")
            else:
                local.write_text(response.text)
        for line in local.read_text().splitlines():
            domain = line.strip().lower()
            if domain:
                owners.setdefault(domain, set()).add(category)

    ambiguous = sum(1 for cats in owners.values() if len(cats) > 1)
    if ambiguous:
        sys.stderr.write(f"  dropped {ambiguous} domains listed in >1 category\n")
    return {d: next(iter(c)) for d, c in owners.items() if len(c) == 1}


def iter_tar_documents(
    archive: Path, labels: dict[str, str]
) -> Iterator[tuple[str, str, str]]:
    """Stream ``<domain>.html`` members out of the corpus tarball.

    Streaming rather than unpacking: the archive expands past 47 GB, and every
    document is read exactly once anyway.

    Args:
        archive: Path to ``shallalist_all.tar.gz``.
        labels: Domain to category mapping; unlabelled members are skipped.

    Yields:
        tuple[str, str, str]: ``(category, domain, html)`` per labelled document.
    """
    with tarfile.open(archive, "r|gz") as tar:  # streaming mode
        for member in tar:
            if not member.isfile() or not member.name.endswith(".html"):
                continue
            domain = Path(member.name).stem.lower()
            category = labels.get(domain)
            if category is None:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            yield category, domain, handle.read().decode("utf-8", errors="ignore")


def iter_documents(corpus: Path) -> Iterator[tuple[str, str, Path]]:
    """Walk a ``<category>/<domain>.html`` tree.

    Args:
        corpus: Root of the unpacked corpus.

    Yields:
        tuple[str, str, Path]: ``(category, domain, html_path)`` per document.
    """
    for category_dir in sorted(p for p in corpus.iterdir() if p.is_dir()):
        category = category_dir.name
        for html_path in sorted(category_dir.rglob("*.html")):
            yield category, html_path.stem, html_path


def extract(html_path: Path) -> str:
    """Extract cleaned page text from one document.

    Uses the serving cleaner so the training and inference pipelines cannot
    drift. The domain prefix is deliberately NOT added here: deduplication has
    to run on the page text alone, or boilerplate shared across domains looks
    unique because each copy carries a different prefix.

    Args:
        html_path: File to read.

    Returns:
        str: Cleaned text, or ``""`` if the file could not be read.
    """
    from ..text_processor import TextProcessor

    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return TextProcessor.process_html_to_text(html)


def model_input(domain: str, text: str) -> str:
    """Build the training input, matching the original run.

    Args:
        domain: Domain the document belongs to.
        text: Cleaned page text.

    Returns:
        str: ``"<domain-stem> <text>"``, as in notebooks/04_train_model.ipynb.
    """
    stem = domain.rsplit(".", 1)[0] if "." in domain else domain
    return f"{stem} {text}"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Turn the HTML corpus into labelled text splits"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="shallalist_all.tar.gz (streamed), or an unpacked <category>/*.html tree",
    )
    parser.add_argument(
        "--cats",
        default="data/labels/shallalist_cats.txt",
        help="Category list, for labelling a tarball corpus",
    )
    parser.add_argument(
        "--label-cache",
        default="data/labels/domains",
        help="Where fetched per-category domain lists are cached",
    )
    parser.add_argument("--out", required=True, help="Output directory for JSONL")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=6,
        help="Drop documents with fewer tokens (training used >5)",
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=100,
        help="Drop categories with fewer documents (training used 100)",
    )
    parser.add_argument(
        "--raw-categories",
        action="store_true",
        help="Use Shallalist categories verbatim, skipping the taxonomy mapping. "
        "Reproduces the pre-0.9 label set",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Cap documents per class. Balances the long tail and makes the "
        "corpus trainable in hours rather than days; 0 keeps everything",
    )
    parser.add_argument(
        "--limit", type=int, help="Stop after N documents (for a smoke run)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the prepared dataset.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code.
    """
    args = build_parser().parse_args(argv)
    corpus = Path(args.corpus)
    if not corpus.exists():
        sys.stderr.write(f"corpus not found: {corpus}\n")
        return 1

    from ..text_processor import TextProcessor

    records: list[dict[str, str]] = []
    seen_text: Counter[str] = Counter()
    kept_per_class: Counter[str] = Counter()
    blocked_per_class: Counter[str] = Counter()
    parked_from: Counter[str] = Counter()
    unavailable_from: Counter[str] = Counter()
    scanned = 0

    if corpus.is_dir():
        documents = (
            (category, domain, None, path)
            for category, domain, path in iter_documents(corpus)
        )
    else:
        cats_file = Path(args.cats)
        if not cats_file.exists():
            sys.stderr.write(
                f"{cats_file} not found -- run download_corpus.py --set labels\n"
            )
            return 1
        print(f"loading labels from {LABEL_MIRROR}")
        raw_labels = load_category_map(cats_file, Path(args.label_cache))
        labels = {}
        excluded = 0
        for domain, category in raw_labels.items():
            mapped = category if args.raw_categories else map_category(category)
            if mapped is None:
                excluded += 1
                continue
            labels[domain] = mapped
        if excluded:
            print(f"excluded {excluded:,} domains in classes not visible in page text")
        print(f"labelled domains: {len(labels)}")
        documents = (
            (category, domain, html, None)
            for category, domain, html in iter_tar_documents(corpus, labels)
        )

    for category, domain, html, path in documents:
        if category in DROPPED_BY_NAME:
            continue
        scanned += 1
        if scanned % 20000 == 0:
            print(f"  scanned {scanned}, kept {len(records)}", flush=True)
        # Check the cap before extracting: text extraction dominates the run,
        # and doing it for a class that is already full is pure waste.
        if args.max_per_class and kept_per_class[category] >= args.max_per_class:
            continue
        # Refuse anti-bot interstitials before they become training data. The corpus is
        # a 2022 scrape and roughly 5% of it is challenge pages, but they are not spread
        # evenly: 29.9% of `drugs` documents are challenge pages, because pharmacy sites
        # are heavily bot-protected. The model therefore learned `drugs` as the label for
        # "I cannot read this page", which is why zappos.com, newlook.com and
        # suicidepreventionlifeline.org all came back as drugs at low confidence.
        #
        # This mattered less under the old cleaner, which deduplicated a challenge page
        # down to a handful of generic tokens. With term frequency preserved, a short
        # repetitive interstitial is a strong, consistent signal for whatever label the
        # domain happened to carry.
        raw = html if html is not None else path.read_text(errors="ignore")  # pyright: ignore[reportOptionalMemberAccess]
        verdict = detect_block(raw, domain=domain)
        if verdict.blocked:
            blocked_per_class[category] += 1
            continue

        text = (
            TextProcessor.process_html_to_text(html)
            if html is not None
            else extract(path)  # pyright: ignore[reportArgumentType]
        )
        if len(text.split()) < args.min_tokens:
            continue

        # A parking placeholder is labelled `parked`, not whatever the domain used to
        # sell. 7.9% of this corpus is parking pages and they concentrate hard: 42% of
        # `drugs`, 23% of `webmail`, 18% of `downloads`, because expired domains in
        # those niches get parked. Left alone, the model learns that a for-sale template
        # *means* drugs -- which is exactly why zappos.com and newlook.com came back as
        # drugs on an independent test set.
        if looks_parked(text):
            parked_from[category] += 1
            category = "parked"
        # The other way a domain serves bytes without being a site: an autoindex, a
        # registrar's "coming soon", a suspended account, a 404. Checked after parking
        # because a for-sale page is the more specific answer, and some say both.
        elif looks_unavailable(text):
            unavailable_from[category] += 1
            category = "unavailable"
        kept_per_class[category] += 1
        records.append({"domain": domain, "category": category, "text": text})
        seen_text[text] += 1
        if args.limit and len(records) >= args.limit:
            break

    # Training blanked boilerplate that appeared under more than one domain,
    # because identical navigation chrome carries no category signal. This must
    # happen before the domain prefix is added, or every copy looks unique.
    records = [r for r in records if seen_text[r["text"]] == 1]
    for record in records:
        record["text"] = model_input(record["domain"], record["text"])

    counts = Counter(r["category"] for r in records)
    keep = {c for c, n in counts.items() if n >= args.min_docs}
    dropped = sorted(set(counts) - keep)
    records = [r for r in records if r["category"] in keep]

    # The split is a pure function of the domain, so this preparer and the screenshot
    # preparer cannot disagree about where a domain belongs -- which they did, three
    # times, each producing a plausible wrong number. See splits.py.
    splits: dict[str, list[dict]] = {name: [] for name in SPLITS}
    for record in records:
        splits[split_of(record["domain"])].append(record)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    labels = sorted(keep)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    if blocked_per_class:
        worst = ", ".join(f"{c} {n}" for c, n in blocked_per_class.most_common(5))
        print(
            f"refused {sum(blocked_per_class.values()):,} anti-bot interstitials "
            f"(worst: {worst})"
        )
    if unavailable_from:
        worst = ", ".join(f"{c} {n}" for c, n in unavailable_from.most_common(5))
        print(
            f"relabelled {sum(unavailable_from.values()):,} no-site placeholders as "
            f"`unavailable` (taken from: {worst})"
        )
    if parked_from:
        worst = ", ".join(f"{c} {n}" for c, n in parked_from.most_common(5))
        print(
            f"relabelled {sum(parked_from.values()):,} parking placeholders as `parked` "
            f"(taken from: {worst})"
        )
    print(f"documents kept: {len(records)}")
    print(f"categories: {len(labels)}")
    if dropped:
        print(f"dropped for <{args.min_docs} docs: {', '.join(dropped)}")
    for name, rows in splits.items():
        print(f"  {name:5s} {len(rows):7d}")
    print(f"wrote {out_dir}/[train|val|test].jsonl and labels.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
