#!/usr/bin/env python

"""Tests for the outcome taxonomy and run reporting."""

import json
import socket
import unittest
from datetime import UTC, datetime, timedelta

from piedomains.outcomes import (
    ErrorCode,
    Stage,
    Status,
    annotate,
    build_report,
    classify_exception,
    failure,
)


class TestClassifyException(unittest.TestCase):
    """Exceptions map to stable, groupable codes."""

    def test_maps_by_exception_type(self):
        self.assertEqual(classify_exception(TimeoutError()), ErrorCode.TIMEOUT)
        self.assertEqual(classify_exception(socket.gaierror()), ErrorCode.DNS_ERROR)
        self.assertEqual(
            classify_exception(ConnectionResetError()), ErrorCode.CONNECTION_ERROR
        )
        self.assertEqual(
            classify_exception(FileNotFoundError()), ErrorCode.MISSING_INPUT_PATH
        )

    def test_maps_by_message_when_type_is_generic(self):
        cases = {
            "Page.goto: Timeout 30000ms exceeded": ErrorCode.TIMEOUT,
            "getaddrinfo failed: Name or service not known": ErrorCode.DNS_ERROR,
            "blocked by robots.txt": ErrorCode.ROBOTS_BLOCKED,
            "HTTP 429 Too Many Requests": ErrorCode.ARCHIVE_RATE_LIMITED,
            "no snapshot available near 20100101": ErrorCode.NO_ARCHIVE_SNAPSHOT,
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(classify_exception(RuntimeError(message)), expected)

    def test_unrecognised_falls_back_to_unknown(self):
        self.assertEqual(
            classify_exception(RuntimeError("something odd")), ErrorCode.UNKNOWN
        )

    def test_codes_are_plain_strings_for_json(self):
        """Codes must serialize as bare strings, not enum reprs."""
        payload = json.dumps({"error_code": ErrorCode.TIMEOUT})
        self.assertEqual(json.loads(payload)["error_code"], "timeout")


class TestFailureAndAnnotate(unittest.TestCase):
    """Row annotation fills in coherent outcome fields."""

    def test_failure_marks_retryable_for_transient_codes(self):
        row = failure(ErrorCode.TIMEOUT, Stage.FETCH, "timed out")
        self.assertEqual(row["status"], Status.FAILED.value)
        self.assertEqual(row["stage"], "fetch")
        self.assertEqual(row["error_code"], "timeout")
        self.assertTrue(row["retryable"])

    def test_failure_marks_permanent_codes_not_retryable(self):
        row = failure(ErrorCode.EMPTY_TEXT, Stage.INFER, "no text")
        self.assertFalse(row["retryable"])

    def test_annotate_infers_ok_from_a_category(self):
        row = annotate({"domain": "a.com", "category": "news", "confidence": 0.9})
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["error_code"])
        self.assertFalse(row["retryable"])

    def test_annotate_infers_failure_from_missing_category(self):
        row = annotate({"domain": "b.com", "category": None})
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error_code"], ErrorCode.UNKNOWN.value)

    def test_annotate_treats_an_error_as_failure_even_with_a_category(self):
        row = annotate({"domain": "c.com", "category": "news", "error": "partial"})
        self.assertEqual(row["status"], "failed")

    def test_annotate_is_idempotent(self):
        row = {"domain": "d.com", "category": None, "error": "x"}
        once = dict(annotate(row))
        twice = annotate(row)
        self.assertEqual(once, twice)


class TestBuildReport(unittest.TestCase):
    """The report aggregates failures and names what is missing."""

    def _rows(self):
        return [
            {"domain": "ok1.com", "category": "news", "confidence": 0.9},
            {"domain": "ok2.com", "category": "shopping", "confidence": 0.8},
            failure(ErrorCode.DNS_ERROR, Stage.FETCH, "dns") | {"domain": "dead.com"},
            failure(ErrorCode.TIMEOUT, Stage.FETCH, "slow") | {"domain": "slow.com"},
            failure(ErrorCode.EMPTY_TEXT, Stage.INFER, "blank")
            | {"domain": "blank.com"},
        ]

    def test_counts_and_missing_list(self):
        report = build_report(self._rows(), run_id="r1")
        self.assertEqual(report["total"], 5)
        self.assertEqual(report["classified"], 2)
        self.assertEqual(report["failed"], 3)
        self.assertEqual(report["missing"], ["dead.com", "slow.com", "blank.com"])

    def test_groups_by_reason_and_stage(self):
        report = build_report(self._rows(), run_id="r1")
        self.assertEqual(
            report["by_reason"], {"dns_error": 1, "timeout": 1, "empty_text": 1}
        )
        self.assertEqual(report["by_stage"], {"fetch": 2, "infer": 1})

    def test_elapsed_is_computed_from_the_start_time(self):
        started = datetime.now(UTC)
        finished = started + timedelta(milliseconds=1500)
        report = build_report(
            self._rows(), run_id="r1", started_at=started, finished_at=finished
        )
        self.assertEqual(report["elapsed_ms"], 1500)
        self.assertEqual(report["run_id"], "r1")

    def test_groups_successes_by_content_source(self):
        """Callers have to be able to see which rows came from the archive."""
        rows = [
            *self._rows(),
            {
                "domain": "etsy.com",
                "category": "shopping",
                "confidence": 0.7,
                "source": "archive",
            },
        ]
        report = build_report(rows, run_id="r1")
        self.assertEqual(report["by_source"], {"live": 2, "archive": 1})

    def test_failed_rows_have_no_source(self):
        """A row that produced nothing came from nowhere; do not count it."""
        report = build_report(self._rows(), run_id="r1")
        self.assertEqual(sum(report["by_source"].values()), report["classified"])

    def test_empty_input_reports_nothing_missing(self):
        report = build_report([], run_id="r1")
        self.assertEqual(report["total"], 0)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["missing"], [])

    def test_report_is_json_serialisable(self):
        report = build_report(self._rows(), run_id="r1")
        self.assertIn("dns_error", json.dumps(report))


class TestApiRunReport(unittest.TestCase):
    """classify() returns rows plus a report, with correct stage attribution."""

    def _run(self):
        from unittest.mock import patch

        from piedomains.api import DomainClassifier

        collection = {
            "domains": [
                {"domain": "good.com", "url": "good.com", "fetch_success": True},
                {
                    "domain": "dead.com",
                    "url": "dead.com",
                    "fetch_success": False,
                    "error": "getaddrinfo failed: Name or service not known",
                },
                {
                    "domain": "slow.com",
                    "url": "slow.com",
                    "fetch_success": False,
                    "error": "Page.goto: Timeout 30000ms exceeded",
                },
            ]
        }
        rows = [
            {"domain": "good.com", "category": "news", "confidence": 0.87},
            {"domain": "dead.com", "category": None},
            {"domain": "slow.com", "category": None},
            {
                "domain": "blank.com",
                "category": None,
                "error": "No meaningful text extracted",
            },
        ]
        classifier = DomainClassifier(cache_dir="/tmp/pd-report-test")
        with (
            patch.object(DomainClassifier, "collect_content", return_value=collection),
            patch.object(
                DomainClassifier, "classify_from_collection", return_value=rows
            ),
        ):
            return classifier.classify(
                ["good.com", "dead.com", "slow.com", "blank.com"]
            )

    def test_envelope_shape(self):
        run = self._run()
        self.assertIn("results", run)
        self.assertIn("report", run)

    def test_fetch_failures_are_attributed_to_the_fetch_stage(self):
        rows = {r["domain"]: r for r in self._run()["results"]}
        self.assertEqual(rows["dead.com"]["stage"], "fetch")
        self.assertEqual(rows["dead.com"]["error_code"], "dns_error")
        self.assertTrue(rows["dead.com"]["retryable"])
        self.assertEqual(rows["slow.com"]["error_code"], "timeout")

    def test_inference_failures_stay_at_the_infer_stage(self):
        rows = {r["domain"]: r for r in self._run()["results"]}
        self.assertEqual(rows["blank.com"]["stage"], "infer")
        self.assertEqual(rows["blank.com"]["error_code"], "empty_text")
        self.assertFalse(rows["blank.com"]["retryable"])

    def test_report_names_every_domain_that_produced_nothing(self):
        report = self._run()["report"]
        self.assertEqual(report["total"], 4)
        self.assertEqual(report["classified"], 1)
        self.assertEqual(report["failed"], 3)
        self.assertEqual(
            sorted(report["missing"]), ["blank.com", "dead.com", "slow.com"]
        )

    def test_report_carries_a_run_id(self):
        self.assertTrue(self._run()["report"]["run_id"])


class TestReconciliation(unittest.TestCase):
    """Every requested domain gets a row, even if the pipeline lost it."""

    def _run(self, requested, returned_domains):
        from unittest.mock import patch

        from piedomains.api import DomainClassifier

        collection = {
            "domains": [
                {"domain": d, "url": d, "fetch_success": True} for d in returned_domains
            ]
        }
        rows = [
            {"domain": d, "category": "news", "confidence": 0.9}
            for d in returned_domains
        ]
        classifier = DomainClassifier(cache_dir="/tmp/pd-reconcile-test")
        with (
            patch.object(DomainClassifier, "collect_content", return_value=collection),
            patch.object(
                DomainClassifier, "classify_from_collection", return_value=rows
            ),
        ):
            return classifier.classify(requested)

    def test_dropped_domains_are_reported_not_silently_lost(self):
        """A domain the pipeline never returns must still appear as failed.

        Regression: an eval run asked for 44 domains and got 33 rows back; the
        other 11 appeared in neither the results nor the report, because the
        report counted rows returned rather than domains requested.
        """
        run = self._run(["a.com", "b.com", "c.com"], ["a.com"])
        self.assertEqual(run["report"]["total"], 3)
        self.assertEqual(run["report"]["failed"], 2)
        self.assertEqual(sorted(run["report"]["missing"]), ["b.com", "c.com"])

    def test_results_preserve_requested_order(self):
        run = self._run(["z.com", "y.com", "x.com"], ["y.com"])
        self.assertEqual(
            [r["domain"] for r in run["results"]], ["z.com", "y.com", "x.com"]
        )

    def test_nothing_dropped_means_no_synthetic_rows(self):
        run = self._run(["a.com", "b.com"], ["a.com", "b.com"])
        self.assertEqual(run["report"]["total"], 2)
        self.assertEqual(run["report"]["failed"], 0)
        self.assertEqual(run["report"]["missing"], [])


if __name__ == "__main__":
    unittest.main()
