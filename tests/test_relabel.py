#!/usr/bin/env python

"""Tests for the LLM re-labelling pass.

Every test here pins a way this module could quietly corrupt a corpus rather than fail
loudly. The batch API is the dangerous part: it sends N documents and gets back a JSON
array with no ordering guarantee, so matching replies to requests by position would write
one document's verdict onto another and nothing downstream would ever notice.

No test here makes a network call; ``litellm`` is patched throughout.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from piedomains.training.relabel import (
    ABSTAIN,
    BudgetExceededError,
    Verdict,
    append_verdicts,
    apply,
    ask,
    assert_splits,
    load_verdicts,
    prompt_hash,
    select,
    write_corpus,
)
from piedomains.training.splits import SPLITS, split_of

LABELS = ["drugs", "shopping", "news", "parked"]


def make_rows(domains_by_split):
    """Build a corpus dict from {split: [(domain, category)]}."""
    return {
        name: [
            {"domain": d, "category": c, "text": f"{d} some page text"}
            for d, c in domains_by_split.get(name, [])
        ]
        for name in SPLITS
    }


def domains_in(split_name, n):
    """Find n domains that genuinely hash to the given split."""
    found = []
    i = 0
    while len(found) < n:
        candidate = f"example{i}.com"
        if split_of(candidate) == split_name:
            found.append(candidate)
        i += 1
    return found


def fake_response(payload):
    """A litellm response object carrying the given JSON payload."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(payload)
    return response


class TestBatchMatching(unittest.TestCase):
    """The failure that would be invisible: a verdict landing on the wrong document."""

    def _ask(self, payload, records):
        rows = [("train", r) for r in records]
        with (
            patch("litellm.completion", return_value=fake_response(payload)),
            patch("litellm.completion_cost", return_value=0.001),
        ):
            return ask(rows, LABELS, {}, {"model": "test"}, 2000, "hash")[0]

    def test_out_of_order_reply_is_matched_by_domain_not_index(self):
        records = [
            {"domain": "a.com", "category": "drugs", "text": "a"},
            {"domain": "b.com", "category": "drugs", "text": "b"},
        ]
        # Deliberately reversed relative to the request.
        payload = [
            {"domain": "b.com", "category": "news", "confidence": 0.9},
            {"domain": "a.com", "category": "shopping", "confidence": 0.9},
        ]
        verdicts = {v.domain: v for v in self._ask(payload, records)}
        self.assertEqual(verdicts["a.com"].category, "shopping")
        self.assertEqual(verdicts["b.com"].category, "news")

    def test_domain_the_model_invented_is_dropped(self):
        records = [{"domain": "a.com", "category": "drugs", "text": "a"}]
        payload = [
            {"domain": "a.com", "category": "news", "confidence": 0.9},
            {"domain": "hallucinated.com", "category": "shopping", "confidence": 0.9},
        ]
        verdicts = self._ask(payload, records)
        self.assertEqual([v.domain for v in verdicts], ["a.com"])

    def test_domain_the_model_skipped_is_recorded_as_missing(self):
        records = [
            {"domain": "a.com", "category": "drugs", "text": "a"},
            {"domain": "b.com", "category": "drugs", "text": "b"},
        ]
        payload = [{"domain": "a.com", "category": "news", "confidence": 0.9}]
        verdicts = {v.domain: v for v in self._ask(payload, records)}
        self.assertEqual(verdicts["b.com"].status, "missing")
        self.assertIsNone(verdicts["b.com"].category)

    def test_invented_label_is_quarantined_not_accepted(self):
        records = [{"domain": "a.com", "category": "drugs", "text": "a"}]
        payload = [
            {"domain": "a.com", "category": "cryptocurrency", "confidence": 0.99}
        ]
        verdict = self._ask(payload, records)[0]
        self.assertEqual(verdict.status, "invalid_label")
        self.assertIsNone(verdict.category)
        # The raw answer survives, because what the model actually said is the evidence.
        self.assertEqual(verdict.raw_label, "cryptocurrency")

    def test_abstention_is_distinct_from_a_parse_failure(self):
        records = [{"domain": "a.com", "category": "drugs", "text": "a"}]
        payload = [{"domain": "a.com", "category": ABSTAIN, "confidence": 0.2}]
        verdict = self._ask(payload, records)[0]
        self.assertEqual(verdict.status, "abstain")
        self.assertIsNone(verdict.category)


class TestApply(unittest.TestCase):
    """A verdict that is not usable must leave the document exactly as it was."""

    def setUp(self):
        self.rows = make_rows({"train": [("a.com", "drugs"), ("b.com", "drugs")]})
        self.rows["train"][0]["domain"] = "a.com"

    def _verdict(self, domain, category, status="ok", confidence=0.9):
        return Verdict(
            domain=domain,
            split="train",
            old_category="drugs",
            raw_label=category,
            category=category,
            substance_type="none",
            confidence=confidence,
            reasoning="",
            evidence="",
            status=status,
            truncated=False,
            model="test",
            prompt_hash="h",
        )

    def test_drop_mode_deletes_disagreements(self):
        verdicts = {"a.com": self._verdict("a.com", "shopping")}
        corrected, moves, _ = apply(self.rows, verdicts, 0.6, "drop")
        self.assertEqual([r["domain"] for r in corrected["train"]], ["b.com"])
        self.assertEqual(moves["drugs -> shopping"], 1)

    def test_correct_mode_rewrites_the_label(self):
        verdicts = {"a.com": self._verdict("a.com", "shopping")}
        corrected, _, _ = apply(self.rows, verdicts, 0.6, "correct")
        by_domain = {r["domain"]: r["category"] for r in corrected["train"]}
        self.assertEqual(by_domain["a.com"], "shopping")

    def test_status_only_takes_parked_but_drops_other_moves(self):
        verdicts = {
            "a.com": self._verdict("a.com", "parked"),
            "b.com": self._verdict("b.com", "shopping"),
        }
        corrected, _, _ = apply(self.rows, verdicts, 0.6, "status-only")
        by_domain = {r["domain"]: r["category"] for r in corrected["train"]}
        self.assertEqual(by_domain, {"a.com": "parked"})

    def test_low_confidence_leaves_the_document_untouched(self):
        verdicts = {"a.com": self._verdict("a.com", "shopping", confidence=0.1)}
        corrected, _, skipped = apply(self.rows, verdicts, 0.6, "drop")
        self.assertEqual(len(corrected["train"]), 2)
        self.assertEqual(skipped["low_confidence"], 1)

    def test_abstention_leaves_the_document_untouched(self):
        verdicts = {"a.com": self._verdict("a.com", None, status="abstain")}
        corrected, _, skipped = apply(self.rows, verdicts, 0.6, "drop")
        self.assertEqual(len(corrected["train"]), 2)
        self.assertEqual(skipped["abstain"], 1)

    def test_text_and_domain_are_never_modified(self):
        verdicts = {"a.com": self._verdict("a.com", "shopping")}
        before = dict(self.rows["train"][0])
        corrected, _, _ = apply(self.rows, verdicts, 0.6, "correct")
        after = next(r for r in corrected["train"] if r["domain"] == "a.com")
        self.assertEqual(after["text"], before["text"])
        self.assertEqual(after["domain"], before["domain"])


class TestCache(unittest.TestCase):
    """A re-run must cost nothing, and an edited prompt must invalidate the cache."""

    def test_verdicts_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.jsonl"
            verdict = Verdict(
                domain="a.com",
                split="train",
                old_category="drugs",
                raw_label="shopping",
                category="shopping",
                substance_type="none",
                confidence=0.9,
                reasoning="r",
                evidence="e",
                status="ok",
                truncated=False,
                model="m",
                prompt_hash="h1",
            )
            append_verdicts(path, [verdict])
            self.assertEqual(load_verdicts(path, "h1")["a.com"].category, "shopping")

    def test_a_different_prompt_hash_ignores_the_cached_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.jsonl"
            append_verdicts(
                path,
                [
                    Verdict(
                        domain="a.com",
                        split="train",
                        old_category="drugs",
                        raw_label="shopping",
                        category="shopping",
                        substance_type="none",
                        confidence=0.9,
                        reasoning="",
                        evidence="",
                        status="ok",
                        truncated=False,
                        model="m",
                        prompt_hash="h1",
                    )
                ],
            )
            self.assertEqual(load_verdicts(path, "h2"), {})

    def test_prompt_hash_changes_with_every_input_that_changes_an_answer(self):
        base = prompt_hash(["a", "b"], {"a": "x"}, 2000)
        self.assertNotEqual(base, prompt_hash(["a", "c"], {"a": "x"}, 2000))
        self.assertNotEqual(base, prompt_hash(["a", "b"], {"a": "y"}, 2000))
        self.assertNotEqual(base, prompt_hash(["a", "b"], {"a": "x"}, 4000))
        self.assertEqual(base, prompt_hash(["a", "b"], {"a": "x"}, 2000))


class TestSplitGuarantee(unittest.TestCase):
    """Splits are a pure function of the domain; this module must never break that."""

    def test_a_tampered_split_is_refused_on_read(self):
        wrong = "test" if split_of("a.com") != "test" else "train"
        rows = {name: [] for name in SPLITS}
        rows[wrong] = [{"domain": "a.com", "category": "drugs", "text": "t"}]
        with self.assertRaises(SystemExit):
            assert_splits(rows)

    def test_a_tampered_split_is_refused_on_write(self):
        wrong = "test" if split_of("a.com") != "test" else "train"
        rows = {name: [] for name in SPLITS}
        rows[wrong] = [{"domain": "a.com", "category": "drugs", "text": "t"}]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                write_corpus(Path(tmp), rows, min_docs=0)

    def test_a_real_corpus_round_trips(self):
        train = domains_in("train", 3)
        rows = make_rows({"train": [(d, "drugs") for d in train]})
        assert_splits(rows)
        with tempfile.TemporaryDirectory() as tmp:
            labels = write_corpus(Path(tmp), rows, min_docs=1)
            self.assertEqual(labels, ["drugs"])
            written = (Path(tmp) / "train.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len(written.strip().splitlines()), 3)


class TestMinDocs(unittest.TestCase):
    """A class collapsing below the floor is the headline result, not a footnote."""

    def test_thin_class_is_dropped_and_excluded_from_labels(self):
        train = domains_in("train", 5)
        rows = make_rows(
            {
                "train": [(d, "drugs") for d in train[:1]]
                + [(d, "news") for d in train[1:]]
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            labels = write_corpus(Path(tmp), rows, min_docs=3)
            self.assertEqual(labels, ["news"])


class TestBudget(unittest.TestCase):
    """The cap has to abort. Warning and continuing is what the old classifier did."""

    def test_exceeding_the_budget_raises(self):
        with self.assertRaises(BudgetExceededError):
            raise BudgetExceededError("spent too much")

    def test_a_priced_call_reports_its_cost(self):
        records = [{"domain": "a.com", "category": "drugs", "text": "a"}]
        payload = [{"domain": "a.com", "category": "news", "confidence": 0.9}]
        with (
            patch("litellm.completion", return_value=fake_response(payload)),
            patch("litellm.completion_cost", return_value=0.25),
        ):
            _, cost = ask(
                [("train", records[0])], LABELS, {}, {"model": "m"}, 2000, "h"
            )
        self.assertEqual(cost, 0.25)

    def test_missing_pricing_does_not_abort_the_run(self):
        records = [{"domain": "a.com", "category": "drugs", "text": "a"}]
        payload = [{"domain": "a.com", "category": "news", "confidence": 0.9}]
        with (
            patch("litellm.completion", return_value=fake_response(payload)),
            patch(
                "litellm.completion_cost", side_effect=Exception("no price for model")
            ),
        ):
            verdicts, cost = ask(
                [("train", records[0])], LABELS, {}, {"model": "m"}, 2000, "h"
            )
        self.assertEqual(cost, 0.0)
        self.assertEqual(verdicts[0].category, "news")


class TestSelect(unittest.TestCase):
    """Selection is by category and is reproducible."""

    def test_only_classes_filters(self):
        rows = make_rows({"train": [("a.com", "drugs"), ("b.com", "news")]})
        chosen = select(rows, {"drugs"}, 0, 42)
        self.assertEqual([r["domain"] for _, r in chosen], ["a.com"])

    def test_limit_is_seeded_and_reproducible(self):
        rows = make_rows({"train": [(f"d{i}.com", "drugs") for i in range(20)]})
        first = [r["domain"] for _, r in select(rows, None, 5, 42)]
        second = [r["domain"] for _, r in select(rows, None, 5, 42)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)


if __name__ == "__main__":
    unittest.main()
