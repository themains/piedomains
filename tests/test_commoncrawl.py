#!/usr/bin/env python

"""Tests for the Common Crawl source.

The index server is the reason this module is shaped the way it is: four of five probes
during development returned 502, 504 or an HTML error page. So the tests that must always
pass are offline, and the live one skips rather than fails when the index does not answer
-- otherwise this file goes red on someone else's machine for reasons unrelated to their
change.

The WARC fixture is built at test time by ``warcio``'s own writer. That tests the parsing
against a real record without checking a binary into the repository, and without trusting
a hand-written byte string to be a valid WARC.
"""

import io
import unittest
from unittest.mock import patch

import pytest

from piedomains import commoncrawl
from piedomains.outcomes import ErrorCode


def warc_bytes(url="https://example.com/", body=b"<html><body>hello</body></html>"):
    """One gzipped WARC response record, exactly as a byte-range fetch would return."""
    from warcio.statusandheaders import StatusAndHeaders
    from warcio.warcwriter import WARCWriter

    buffer = io.BytesIO()
    writer = WARCWriter(buffer, gzip=True)
    writer.write_record(
        writer.create_warc_record(
            url,
            "response",
            payload=io.BytesIO(body),
            http_headers=StatusAndHeaders(
                "200 OK", [("Content-Type", "text/html")], protocol="HTTP/1.1"
            ),
        )
    )
    return buffer.getvalue()


class _Response:
    """Minimal stand-in for a requests response."""

    def __init__(self, content=b"", text="", status_code=200):
        self.content = content
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        import json

        return json.loads(self.text)


class TestRecordParsing(unittest.TestCase):
    """warcio does the parsing; this pins that we hand it the right thing."""

    def test_a_real_warc_record_round_trips(self):
        body = b"<html><head><title>Fixture</title></head><body>content</body></html>"
        raw = warc_bytes(body=body)
        with patch("requests.get", return_value=_Response(content=raw)):
            html = commoncrawl.fetch_record(
                {"filename": "x.warc.gz", "offset": 0, "length": len(raw)}
            )
        self.assertEqual(html, body.decode())

    def test_a_truncated_record_returns_none_rather_than_raising(self):
        # A byte range that lands mid-record must not take the run down with it.
        raw = warc_bytes()[:40]
        with patch("requests.get", return_value=_Response(content=raw)):
            html = commoncrawl.fetch_record(
                {"filename": "x.warc.gz", "offset": 0, "length": 40}
            )
        self.assertIsNone(html)

    def test_a_failed_range_request_returns_none(self):
        with patch("requests.get", side_effect=OSError("connection reset")):
            self.assertIsNone(
                commoncrawl.fetch_record(
                    {"filename": "x.warc.gz", "offset": 0, "length": 10}
                )
            )


class TestRecordSelection(unittest.TestCase):
    """Which index rows are worth fetching."""

    def test_prefers_the_site_root_over_an_article(self):
        root = {
            "url": "https://example.com/",
            "status": "200",
            "mime": "text/html",
            "filename": "f",
            "offset": "0",
            "length": "1",
        }
        article = dict(root, url="https://example.com/news/some-story")
        self.assertTrue(commoncrawl._is_usable(root, "example.com"))
        # "Classify this domain" does not mean "classify a random article on it".
        self.assertFalse(commoncrawl._is_usable(article, "example.com"))

    def test_rejects_non_html_and_non_200(self):
        base = {
            "url": "https://example.com/",
            "status": "200",
            "mime": "text/html",
            "filename": "f",
            "offset": "0",
            "length": "1",
        }
        self.assertFalse(
            commoncrawl._is_usable(dict(base, status="404"), "example.com")
        )
        self.assertFalse(
            commoncrawl._is_usable(dict(base, mime="text/plain"), "example.com")
        )
        # This is how a robots.txt row sneaks in -- it is 200 but not html.
        robots = dict(base, url="https://example.com/robots.txt", mime="text/plain")
        self.assertFalse(commoncrawl._is_usable(robots, "example.com"))

    def test_rejects_a_row_missing_its_byte_range(self):
        self.assertFalse(
            commoncrawl._is_usable(
                {"url": "https://example.com/", "status": "200", "mime": "text/html"},
                "example.com",
            )
        )


class TestIndexResilience(unittest.TestCase):
    """The index is unreliable, and that is the module's central assumption."""

    def test_a_502_is_retried_then_gives_up_quietly(self):
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            return _Response(status_code=502)

        with patch("requests.get", side_effect=flaky), patch("time.sleep"):
            self.assertIsNone(
                commoncrawl._query_index("CC-MAIN-2026-25", "example.com", timeout=1)
            )
        # Retried rather than failing on the first 502, which is what makes the feature
        # work at all against this server.
        self.assertEqual(len(calls), commoncrawl.INDEX_RETRIES)

    def test_a_502_then_success_still_finds_the_record(self):
        row = (
            '{"url": "https://example.com/", "status": "200", "mime": "text/html",'
            ' "filename": "f.warc.gz", "offset": "0", "length": "10",'
            ' "timestamp": "20260101000000"}'
        )
        responses = [_Response(status_code=504), _Response(text=row)]
        with patch("requests.get", side_effect=responses), patch("time.sleep"):
            record = commoncrawl._query_index(
                "CC-MAIN-2026-25", "example.com", timeout=1
            )
        self.assertIsNotNone(record)
        self.assertEqual(record["timestamp"], "20260101000000")

    def test_an_html_error_page_is_not_mistaken_for_records(self):
        # The index returns an nginx error page rather than JSON often enough to matter.
        with patch("requests.get", return_value=_Response(text="<html>502</html>")):
            self.assertIsNone(
                commoncrawl._query_index("CC-MAIN-2026-25", "example.com", timeout=1)
            )

    def test_no_record_maps_to_the_shared_error_code(self):
        # Same string the R sibling uses, so results aggregate across the two.
        self.assertEqual(commoncrawl.NO_RECORD, ErrorCode.NO_ARCHIVE_SNAPSHOT.value)
        self.assertEqual(commoncrawl.NO_RECORD, "no_archive_snapshot")


class TestCrawlSelection(unittest.TestCase):
    """Picking which crawl to ask."""

    def setUp(self):
        commoncrawl._crawls = [
            {"id": "CC-MAIN-2026-30"},
            {"id": "CC-MAIN-2024-10"},
            {"id": "CC-MAIN-2020-05"},
        ]

    def tearDown(self):
        commoncrawl._crawls = None

    def test_without_a_date_the_order_is_as_published(self):
        self.assertEqual(commoncrawl.crawls_for()[0], "CC-MAIN-2026-30")

    def test_with_a_date_the_nearest_crawl_comes_first(self):
        self.assertEqual(commoncrawl.crawls_for(near="20200101")[0], "CC-MAIN-2020-05")
        self.assertEqual(commoncrawl.crawls_for(near="20240201")[0], "CC-MAIN-2024-10")

    def test_a_year_end_date_picks_the_year_end_crawl(self):
        # The case a month*4 approximation gets wrong: 31 December is ISO week 1 of the
        # next year, not week 48. Ranked by that approximation a week-48 crawl came first
        # and the genuinely closest one could fall outside the max_crawls window.
        commoncrawl._crawls = [
            {"id": "CC-MAIN-2025-52"},
            {"id": "CC-MAIN-2025-48"},
            {"id": "CC-MAIN-2025-30"},
        ]
        self.assertEqual(commoncrawl.crawls_for(near="20251231")[0], "CC-MAIN-2025-52")

    def test_ranking_across_a_year_boundary_uses_real_distance(self):
        # The case the `year * 53 + week` fix still got wrong. 31 December 2025 is ISO
        # 2026-W01, so pseudo-week arithmetic scored both of these 3 away and the stable
        # sort kept the newest-first order -- picking the January crawl. In real days
        # they are 19 and 16 away, so the December one is closer.
        commoncrawl._crawls = [{"id": "CC-MAIN-2026-04"}, {"id": "CC-MAIN-2025-51"}]
        self.assertEqual(commoncrawl.crawls_for(near="20251231")[0], "CC-MAIN-2025-51")

    def test_a_malformed_week_sorts_last_rather_than_raising(self):
        # 2025 has 52 ISO weeks, so week 53 is not a date. Rank it last; do not guess.
        commoncrawl._crawls = [{"id": "CC-MAIN-2025-53"}, {"id": "CC-MAIN-2025-40"}]
        self.assertEqual(commoncrawl.crawls_for(near="20251001")[0], "CC-MAIN-2025-40")

    def test_an_unparseable_date_leaves_the_order_alone(self):
        self.assertEqual(
            commoncrawl.crawls_for(near="not-a-date")[0], "CC-MAIN-2026-30"
        )
        self.assertEqual(commoncrawl.crawls_for(near="20261301")[0], "CC-MAIN-2026-30")

    def test_an_unreachable_crawl_list_is_empty_not_an_error(self):
        commoncrawl._crawls = None
        with patch("requests.get", side_effect=OSError("down")):
            self.assertEqual(commoncrawl.crawls_for(), [])


@pytest.mark.integration
@pytest.mark.live
class TestLive(unittest.TestCase):
    """Skips rather than fails: the index answered 1 of 5 probes during development."""

    def test_a_real_lookup_if_the_index_is_up(self):
        record = commoncrawl.find_record("bbc.com", crawl="CC-MAIN-2026-25", timeout=45)
        if record is None:
            self.skipTest("Common Crawl index did not answer")
        self.assertIn("filename", record)
        html = commoncrawl.fetch_record(record, timeout=90)
        self.assertTrue(html and len(html) > 500)


if __name__ == "__main__":
    unittest.main()
