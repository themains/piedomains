#!/usr/bin/env python3
r"""Re-label a prepared corpus with an LLM, and keep the receipts.

**Why this exists.** The `drugs` class is 71.3% label noise. Shallalist defines it as
"Sites offering drugs or explain how to make drugs (legal and non legal). Covers tabacco
as well as viagra and similar substances", and most of the domains on that list were
online pill-pushers that expired years ago and are now SEO spam or parked pages. They kept
the label. So the class learned *"this page looks like a store"* rather than *"this page
is about drugs"*, and it now competes with retail: `walmart.com` scores drugs 0.56 against
shopping 0.14. That is a labelling failure, and no amount of additional data fixes it --
there are only ~672 `drugs` documents corpus-wide.

**This never produces evaluation gold.** If a model is ever scored against labels this
module wrote, the number means nothing. `tests/eval/labels.csv` stays hand-checked.

**Why a second pass and not a hook in prepare_text.py.** That script streams an 18 GB
tarball and takes hours; this has to be re-runnable in minutes. Everything prepare_text
does before its own relabel step -- `detect_block`, the token floor, `looks_parked`,
dedup, `--min-docs` -- is work the LLM must not pay to re-derive. Sending parked pages to
a language model when a regex already caught 42% of the class is spending money to
reproduce a regex.

**The verdicts file is both the audit trail and the cache.** One append-only JSONL, keyed
by ``(domain, prompt_hash)``, greppable by hand. A re-run costs nothing; an abort halfway
leaves a resumable file. `prompt_hash` covers the category list, the definitions, the
template version and the truncation length, so editing the prompt correctly invalidates
every stored verdict -- otherwise the corpus gets corrected by a prompt that no longer
exists.

**It does not know what a drug is.** No class name appears anywhere below.
``--only-classes`` picks the input, ``--allowed-labels`` defines the output vocabulary.
Splitting a class later is a config change and a re-run, not a code change.

Usage::

    uv run python -m piedomains.training.relabel \\
        --data data/prepared-text --out data/prepared-relabelled \\
        --only-classes drugs --verdicts data/relabel/drugs-verdicts.jsonl \\
        --provider anthropic --model claude-sonnet-5 \\
        --budget-usd 5 --mode drop --report reports/relabel-drugs.json

    # inspect what a finished pass would do, without spending anything
    uv run python -m piedomains.training.relabel ... --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .splits import SPLITS, split_of

#: Bumped whenever the prompt's wording or response schema changes. It feeds
#: ``prompt_hash``, so bumping it invalidates every cached verdict -- which is the point.
TEMPLATE_VERSION = "1"

#: What the model may say when a page does not support any confident answer. Kept distinct
#: from a malformed reply: a refusal is information, a parse failure is not.
ABSTAIN = "unclear"

#: Substance taxonomy recorded alongside the label. This is *measurement*, not a class
#: list -- it exists so the decision to split a category can be made from counts rather
#: than from guesswork, without committing the corpus to names that may not survive.
SUBSTANCE_TYPES = ("pharma", "recreational", "tobacco", "none", ABSTAIN)


class BudgetExceededError(RuntimeError):
    """Raised when the next request would take the run past its spend cap."""


@dataclass(frozen=True)
class Verdict:
    """One model opinion about one document.

    ``raw_label`` is stored verbatim, before validation and before any mapping. That is
    deliberate: if the class list changes later, verdicts can be re-projected onto the new
    vocabulary for free, whereas storing only the mapped name destroys that option.
    """

    domain: str
    split: str
    old_category: str
    raw_label: str | None
    category: str | None
    substance_type: str
    confidence: float
    reasoning: str
    evidence: str
    status: str
    truncated: bool
    model: str
    prompt_hash: str


def load_definitions(dict_file: Path, allowed: list[str]) -> dict[str, str]:
    r"""Map each allowed class to a one-line gloss.

    A model handed 44 bare names resolves ``shopping`` against ``automobile`` by its own
    priors, and this vocabulary is deliberately not mutually exclusive -- so the glosses
    are what stop it inventing a taxonomy. Raw Shallalist names are projected through
    ``taxonomy.map_category`` so a merged class (``porn`` -> ``adult``) still gets text.

    Args:
        dict_file: Shallalist's ``<category>\\t<description>`` file.
        allowed: The class names to describe.

    Returns:
        dict[str, str]: Class name to description, for whichever classes have one.
    """
    from .taxonomy import map_category

    permitted = set(allowed)
    found: dict[str, str] = {}
    if not dict_file.exists():
        return found
    for line in dict_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in line:
            continue
        raw, description = line.split("\t", 1)
        mapped = map_category(raw.strip())
        if mapped in permitted and mapped not in found:
            found[mapped] = " ".join(description.split())
    return found


def read_corpus(data: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Read every split of a prepared corpus and its label vocabulary.

    Args:
        data: A ``prepare_text.py`` output directory.

    Returns:
        tuple[dict[str, list[dict[str, Any]]], list[str]]: Rows by split name, and the
        corpus's own label list.

    Raises:
        SystemExit: If the directory is not a prepared corpus.
    """
    from .train_text import read_jsonl

    labels_file = data / "labels.json"
    if not labels_file.exists():
        raise SystemExit(f"{labels_file} not found -- run prepare_text.py first")
    rows = {name: read_jsonl(data / f"{name}.jsonl") for name in SPLITS}
    return rows, json.loads(labels_file.read_text(encoding="utf-8"))


def assert_splits(rows: dict[str, list[dict[str, Any]]]) -> None:
    """Refuse to proceed if any document sits in the wrong split.

    The split is a pure function of the domain, so this is cheap and total. It is checked
    rather than assumed because three separate times in this project a plausible accuracy
    number turned out to have come from a split violation.

    Args:
        rows: Rows by split name.

    Raises:
        SystemExit: If any domain does not hash to the split holding it.
    """
    for name, records in rows.items():
        for record in records:
            if split_of(record["domain"]) != name:
                raise SystemExit(
                    f"{record['domain']} is in {name} but hashes to "
                    f"{split_of(record['domain'])} -- refusing to write a corpus whose "
                    "splits cannot be recomputed"
                )


def select(
    rows: dict[str, list[dict[str, Any]]],
    only: set[str] | None,
    limit: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Choose which documents to ask about.

    Args:
        rows: Rows by split name.
        only: Restrict to these current categories; ``None`` means every document.
        limit: Cap on the number selected; ``0`` means no cap.
        seed: Seed for the cap's sampling, so a capped run is reproducible.

    Returns:
        list[tuple[str, dict[str, Any]]]: ``(split, record)`` pairs.
    """
    chosen = [
        (name, record)
        for name in SPLITS
        for record in rows[name]
        if only is None or record["category"] in only
    ]
    if limit and len(chosen) > limit:
        random.Random(seed).shuffle(chosen)  # noqa: S311 -- sampling, not security
        chosen = chosen[:limit]
    chosen.sort(key=lambda pair: (pair[0], pair[1]["domain"]))
    return chosen


def prompt_hash(
    categories: list[str], definitions: dict[str, str], max_chars: int
) -> str:
    """Fingerprint everything that can change an answer.

    Args:
        categories: The allowed class list, in the order the prompt presents it.
        definitions: Class glosses included in the prompt.
        max_chars: Truncation length applied to each document.

    Returns:
        str: A short hex digest used as part of the cache key.
    """
    payload = json.dumps(
        {
            "template": TEMPLATE_VERSION,
            "categories": categories,
            "definitions": definitions,
            "max_chars": max_chars,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def batches(
    items: list[tuple[str, dict[str, Any]]], size: int
) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    """Split the work into request-sized chunks.

    Args:
        items: ``(split, record)`` pairs.
        size: Documents per request.

    Yields:
        list[tuple[str, dict[str, Any]]]: One chunk.
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _with_retry(call: Any, attempts: int = 3) -> Any:
    """Retry a request on transient failures only.

    Authentication errors are re-raised immediately: retrying a bad key just burns wall
    clock and makes the real problem harder to see.

    Args:
        call: Zero-argument callable issuing the request.
        attempts: Maximum tries.

    Returns:
        Any: Whatever ``call`` returns.

    Raises:
        Exception: The last error, if every attempt fails.
        last: Re-raised when the retry loop somehow exits without a result.
    """
    delay = 2.0
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # litellm raises provider-specific types
            text = f"{type(exc).__name__}: {exc}".lower()
            if any(k in text for k in ("auth", "api key", "permission", "forbidden")):
                raise
            transient = any(
                k in text
                for k in ("rate", "timeout", "overload", "503", "502", "500", "529")
            )
            last = exc
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(delay + random.random())  # noqa: S311 -- retry jitter
            delay *= 2
    raise last if last else RuntimeError("retry loop exited without a result")


def ask(
    batch: list[tuple[str, dict[str, Any]]],
    categories: list[str],
    definitions: dict[str, str],
    params: dict[str, Any],
    max_chars: int,
    fingerprint: str,
) -> tuple[list[Verdict], float]:
    """Send one batch and turn the reply into verdicts.

    Replies are matched to requests **by domain, never by position**: the parser makes no
    ordering guarantee, and index-matching would silently write one document's verdict
    onto another.

    Args:
        batch: ``(split, record)`` pairs to ask about.
        categories: Allowed class names.
        definitions: Class glosses for the prompt.
        params: litellm keyword arguments.
        max_chars: Per-document truncation length.
        fingerprint: The prompt hash to stamp on each verdict.

    Returns:
        tuple[list[Verdict], float]: One verdict per requested document, and the request's
        cost in USD.
    """
    import litellm

    from ..llm.prompts import get_batch_prompt
    from ..llm.response_parser import parse_batch_response

    documents = [
        {"domain": record["domain"], "content": record["text"]} for _, record in batch
    ]
    prompt = get_batch_prompt(
        documents,
        categories,
        definitions=definitions,
        extra_fields={
            "evidence": "a short verbatim quote from the page that justifies the label",
            "substance_type": (
                "one of pharma, recreational, tobacco, none -- what kind of substance the "
                f"site is about, or none if it is not about any; {ABSTAIN} if unsure"
            ),
        },
        max_chars=max_chars,
        abstain=ABSTAIN,
    )
    response = _with_retry(
        lambda: litellm.completion(
            messages=[{"role": "user", "content": prompt}], **params
        )
    )
    cost = 0.0
    try:
        cost = float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:  # pricing gaps must not abort a run
        cost = 0.0

    text = response.choices[0].message.content or ""
    try:
        parsed = parse_batch_response(text, allowed=categories)
    except ValueError:
        parsed = []

    by_domain: dict[str, dict[str, Any]] = {}
    requested = {record["domain"] for _, record in batch}
    for item in parsed:
        raw = item.get("raw_response") or {}
        domain = str(raw.get("domain") or item.get("domain") or "").strip().lower()
        if domain in requested:
            by_domain[domain] = item

    model = str(params.get("model", ""))
    verdicts: list[Verdict] = []
    for split_name, record in batch:
        domain = record["domain"]
        item = by_domain.get(domain)
        if item is None:
            verdicts.append(
                _verdict(
                    record,
                    split_name,
                    None,
                    ABSTAIN,
                    0.0,
                    "",
                    "",
                    "missing",
                    max_chars,
                    model,
                    fingerprint,
                )
            )
            continue
        raw = item.get("raw_response") or {}
        label = item.get("category")
        status = "ok"
        if label == ABSTAIN:
            status, label = "abstain", None
        elif not item.get("valid", True):
            status = "invalid_label"
        substance = str(raw.get("substance_type", ABSTAIN)).strip().lower()
        verdicts.append(
            _verdict(
                record,
                split_name,
                item.get("category"),
                substance if substance in SUBSTANCE_TYPES else ABSTAIN,
                float(item.get("confidence", 0.0)),
                str(item.get("reasoning", "")),
                str(raw.get("evidence", "")),
                status,
                max_chars,
                model,
                fingerprint,
                accepted=label if status == "ok" else None,
            )
        )
    return verdicts, cost


def _verdict(
    record: dict[str, Any],
    split_name: str,
    raw_label: str | None,
    substance: str,
    confidence: float,
    reasoning: str,
    evidence: str,
    status: str,
    max_chars: int,
    model: str,
    fingerprint: str,
    accepted: str | None = None,
) -> Verdict:
    """Build a verdict record.

    Args:
        record: The corpus row.
        split_name: Split holding the row.
        raw_label: What the model said, verbatim.
        substance: Recorded substance type.
        confidence: Model confidence.
        reasoning: Model rationale -- a post-hoc rationalisation, not evidence.
        evidence: Verbatim quote from the page.
        status: Verdict status.
        max_chars: Truncation length, to record whether this document was cut.
        model: Model identifier.
        fingerprint: Prompt hash.
        accepted: The validated label, if the verdict is usable.

    Returns:
        Verdict: The record.
    """
    return Verdict(
        domain=record["domain"],
        split=split_name,
        old_category=record["category"],
        raw_label=raw_label,
        category=accepted,
        substance_type=substance,
        confidence=confidence,
        reasoning=reasoning,
        evidence=evidence,
        status=status,
        truncated=len(record["text"]) > max_chars,
        model=model,
        prompt_hash=fingerprint,
    )


def load_verdicts(path: Path, fingerprint: str) -> dict[str, Verdict]:
    """Load cached verdicts written under the current prompt.

    Verdicts stamped with a different ``prompt_hash`` are ignored rather than deleted:
    they stay in the file as history, but they cannot answer today's question.

    Args:
        path: The verdicts JSONL.
        fingerprint: Only verdicts carrying this prompt hash are returned.

    Returns:
        dict[str, Verdict]: Verdicts by domain.
    """
    if not path.exists():
        return {}
    found: dict[str, Verdict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if data.get("prompt_hash") == fingerprint:
            found[data["domain"]] = Verdict(**data)
    return found


def append_verdicts(path: Path, verdicts: list[Verdict]) -> None:
    """Append verdicts and flush, so an abort leaves a resumable file.

    Args:
        path: The verdicts JSONL.
        verdicts: Records to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for verdict in verdicts:
            handle.write(json.dumps(asdict(verdict)) + "\n")
        handle.flush()


def write_manifest(path: Path, params: dict[str, Any], fingerprint: str) -> None:
    """Record what produced a verdicts file.

    A verdicts file without a manifest is uninterpretable a year later -- models get
    deprecated and prompts get edited.

    Args:
        path: The verdicts JSONL; the manifest is written beside it.
        params: litellm keyword arguments used.
        fingerprint: The prompt hash.
    """
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- provenance only
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        revision = ""
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "model": params.get("model"),
                "temperature": params.get("temperature"),
                "prompt_hash": fingerprint,
                "template_version": TEMPLATE_VERSION,
                "git_revision": revision,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def apply(
    rows: dict[str, list[dict[str, Any]]],
    verdicts: dict[str, Verdict],
    min_confidence: float,
    mode: str,
) -> tuple[dict[str, list[dict[str, Any]]], Counter, Counter]:
    """Apply verdicts to the corpus.

    ``drop`` deletes documents the model disagrees with rather than moving them. Moving
    them would import the model's taxonomy into classes that are not being audited --
    ``shopping`` already has tens of thousands of unused documents, so redistributing a
    few hundred noisy ones is all risk and no gain.

    A document is left exactly as it was whenever the verdict is not usable: an abstention,
    an invalid label, a missing reply, or confidence below the floor. Silence is not
    evidence of a wrong label.

    Args:
        rows: Rows by split name.
        verdicts: Verdicts by domain.
        min_confidence: Verdicts below this are ignored.
        mode: ``drop``, ``status-only`` or ``correct``.

    Returns:
        tuple[dict[str, list[dict[str, Any]]], Counter, Counter]: The corrected corpus, a
        ``"old -> new"`` counter, and a counter of why verdicts were not acted on.
    """
    moves: Counter = Counter()
    skipped: Counter = Counter()
    corrected: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}

    for name in SPLITS:
        for record in rows[name]:
            verdict = verdicts.get(record["domain"])
            if verdict is None:
                corrected[name].append(record)
                continue
            if verdict.status != "ok" or verdict.category is None:
                skipped[verdict.status] += 1
                corrected[name].append(record)
                continue
            if verdict.confidence < min_confidence:
                skipped["low_confidence"] += 1
                corrected[name].append(record)
                continue
            if verdict.category == record["category"]:
                moves["kept"] += 1
                corrected[name].append(record)
                continue

            # Free information either way: how many the parked/unavailable heuristics
            # missed says whether blocking.py needs work.
            if verdict.category in ("parked", "unavailable"):
                skipped["heuristic_missed"] += 1
            moves[f"{record['category']} -> {verdict.category}"] += 1
            if mode == "correct" or (
                mode == "status-only" and verdict.category in ("parked", "unavailable")
            ):
                corrected[name].append({**record, "category": verdict.category})
            # mode == "drop", or a status-only move to a non-status class: document goes.
    return corrected, moves, skipped


def write_corpus(
    out: Path, rows: dict[str, list[dict[str, Any]]], min_docs: int
) -> list[str]:
    """Write the corrected corpus, re-applying the minimum-documents floor.

    Args:
        out: Destination directory.
        rows: Corrected rows by split name.
        min_docs: Classes with fewer training documents than this are dropped.

    Returns:
        list[str]: The surviving label vocabulary.

    Raises:
        SystemExit: If a document would be written into a split it does not hash to.
    """
    # Before the min-docs filter, not after: a class dropped here would otherwise carry
    # its mis-split documents out of the check entirely.
    assert_splits(rows)

    counts = Counter(record["category"] for record in rows["train"])
    keep = {name for name, n in counts.items() if n >= min_docs}
    dropped = sorted(set(counts) - keep)
    if dropped:
        print(
            "\nclasses falling below --min-docs and being dropped: "
            + ", ".join(f"{name} (n={counts[name]})" for name in dropped)
        )

    out.mkdir(parents=True, exist_ok=True)
    surviving: set[str] = set()
    for name in SPLITS:
        kept = [r for r in rows[name] if r["category"] in keep]
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for record in kept:
                if split_of(record["domain"]) != name:
                    raise SystemExit(
                        f"refusing to write {record['domain']} into {name}: it hashes to "
                        f"{split_of(record['domain'])}"
                    )
                handle.write(json.dumps(record) + "\n")
        surviving.update(r["category"] for r in kept)
        print(f"  {name:5} {len(kept):,}")
    labels = sorted(surviving)
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return labels


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Re-label a prepared corpus with an LLM"
    )
    parser.add_argument(
        "--data", required=True, help="prepare_text.py output directory"
    )
    parser.add_argument("--out", default="", help="Where to write the corrected corpus")
    parser.add_argument(
        "--verdicts", required=True, help="Verdict JSONL: audit and cache"
    )
    parser.add_argument(
        "--only-classes",
        default="",
        help="Comma-separated current categories to audit; empty means all",
    )
    parser.add_argument(
        "--allowed-labels",
        default="",
        help="JSON list of permitted output labels; defaults to the corpus's labels.json",
    )
    parser.add_argument(
        "--definitions",
        default="data/labels/shallalist_dict.txt",
        help="Category glosses to include in the prompt",
    )
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap documents asked about"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("drop", "status-only", "correct"),
        default="drop",
        help="What to do with a disagreement",
    )
    parser.add_argument("--min-docs", type=int, default=100)
    parser.add_argument("--report", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Spend nothing: apply cached verdicts only, and write no corpus",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a re-labelling pass.

    Returns:
        int: Process exit status; non-zero if the budget stopped the run.

    Args:
        argv: Command-line arguments.

    Raises:
        BudgetExceededError: Caught internally; the run stops and reports what it spent.
    """
    args = build_parser().parse_args(argv)

    rows, corpus_labels = read_corpus(Path(args.data))
    assert_splits(rows)
    allowed = (
        json.loads(Path(args.allowed_labels).read_text(encoding="utf-8"))
        if args.allowed_labels
        else corpus_labels
    )
    definitions = load_definitions(Path(args.definitions), allowed)
    fingerprint = prompt_hash(allowed, definitions, args.max_chars)

    only = {c.strip() for c in args.only_classes.split(",") if c.strip()} or None
    chosen = select(rows, only, args.limit, args.seed)
    verdicts_path = Path(args.verdicts)
    cached = load_verdicts(verdicts_path, fingerprint)
    pending = [pair for pair in chosen if pair[1]["domain"] not in cached]

    print(f"corpus {args.data}: {sum(len(v) for v in rows.values()):,} documents")
    print(f"selected {len(chosen):,}; {len(cached):,} cached, {len(pending):,} to ask")
    print(f"prompt {fingerprint}, {len(definitions)}/{len(allowed)} classes documented")

    spent = 0.0
    status = 0
    if pending and not args.dry_run:
        from ..llm.config import LLMConfig

        config = LLMConfig(
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            categories=allowed,
        )
        params = config.to_litellm_params()
        write_manifest(verdicts_path, params, fingerprint)
        try:
            for index, batch in enumerate(batches(pending, args.batch_size), start=1):
                if spent >= args.budget_usd:
                    raise BudgetExceededError(
                        f"spent ${spent:.2f} of ${args.budget_usd:.2f}; stopping before "
                        f"batch {index}"
                    )
                produced, cost = ask(
                    batch, allowed, definitions, params, args.max_chars, fingerprint
                )
                append_verdicts(verdicts_path, produced)
                spent += cost
                cached.update({v.domain: v for v in produced})
                print(
                    f"  batch {index}: {len(produced)} verdicts, ${spent:.3f} spent",
                    flush=True,
                )
        except BudgetExceededError as exc:
            print(f"\n{exc}", file=sys.stderr)
            status = 1

    corrected, moves, skipped = apply(rows, cached, args.min_confidence, args.mode)
    substances = Counter(v.substance_type for v in cached.values() if v.status == "ok")
    statuses = Counter(v.status for v in cached.values())

    print(f"\nverdict status: {dict(statuses)}")
    print(f"substance types: {dict(substances)}")
    print("\ntop label movements:")
    for name, n in moves.most_common(15):
        print(f"  {name:44} {n:5}")
    if skipped:
        print(f"\nnot acted on: {dict(skipped)}")

    labels: list[str] = []
    if args.out and not args.dry_run:
        print(f"\nwriting {args.out} (mode={args.mode}):")
        labels = write_corpus(Path(args.out), corrected, args.min_docs)
        print(f"  {len(labels)} classes")
    elif args.dry_run:
        counts = Counter(r["category"] for r in corrected["train"])
        thin = sorted(n for n in counts.items() if n[1] < args.min_docs)
        print(f"\ndry run: {len(counts)} classes would survive in train")
        if thin:
            print("  below --min-docs: " + ", ".join(f"{k} (n={v})" for k, v in thin))

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(
                {
                    "note": (
                        "Model reasoning is a post-hoc rationalisation, not evidence of "
                        "correctness. These labels must never be used as evaluation gold."
                    ),
                    "corpus": args.data,
                    "mode": args.mode,
                    "prompt_hash": fingerprint,
                    "model": args.model,
                    "selected": len(chosen),
                    "spent_usd": round(spent, 4),
                    "statuses": dict(statuses),
                    "substance_types": dict(substances),
                    "moves": dict(moves),
                    "not_acted_on": dict(skipped),
                    "labels": labels,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    return status


if __name__ == "__main__":
    sys.exit(main())
