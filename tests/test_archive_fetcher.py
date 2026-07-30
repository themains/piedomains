#!/usr/bin/env python

"""Offline tests for the wayback-backed ArchiveFetcher.

These mock the `wayback` client so closest-snapshot selection, status
filtering, error mapping and cache-key separation are all covered without
touching the network. Live coverage lives in test_integration.py behind the
`archive` marker.
"""

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from piedomains.fetchers import ArchiveFetcher
from piedomains.outcomes import ErrorCode


def record(ts: datetime, url="http://cnn.com/", status=200):
    """Build a stand-in CdxRecord."""
    return SimpleNamespace(
        timestamp=ts, original=url, status_code=status, mimetype="text/html"
    )


class TestClosestRecordSelection(unittest.TestCase):
    """The capture nearest the target date wins."""

    def test_picks_the_nearest_capture(self):
        target = datetime(2010, 1, 1, tzinfo=UTC)
        candidates = [
            record(target - timedelta(days=200)),
            record(target + timedelta(hours=4)),  # nearest
            record(target + timedelta(days=90)),
        ]
        fetcher = ArchiveFetcher("20100101")
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.return_value.__enter__.return_value.search.return_value = iter(
                candidates
            )
            best = fetcher.find_closest_record("cnn.com")
        self.assertEqual(best.timestamp, target + timedelta(hours=4))

    def test_search_filters_to_status_200(self):
        """An archived 404 must never be offered as content."""
        fetcher = ArchiveFetcher("20100101")
        with patch.object(fetcher, "_client") as mock_client:
            search = mock_client.return_value.__enter__.return_value.search
            search.return_value = iter([record(datetime(2010, 1, 1, tzinfo=UTC))])
            fetcher.find_closest_record("cnn.com")
        self.assertEqual(search.call_args.kwargs["filter_field"], "statuscode:200")

    def test_no_captures_returns_none(self):
        fetcher = ArchiveFetcher("20100101")
        with patch.object(fetcher, "_client") as mock_client:
            mock_client.return_value.__enter__.return_value.search.return_value = iter(
                []
            )
            self.assertIsNone(fetcher.find_closest_record("nowhere.invalid"))


class TestPlaybackUrls(unittest.TestCase):
    """Text uses id_; screenshots use if_."""

    def test_iframe_url_uses_if_suffix(self):
        fetcher = ArchiveFetcher("20100101")
        url = fetcher.iframe_url(record(datetime(2010, 1, 1, 4, 17, 27, tzinfo=UTC)))
        self.assertEqual(
            url, "https://web.archive.org/web/20100101041727if_/http://cnn.com/"
        )

    def test_iframe_url_is_not_id_mode(self):
        """id_ would render without CSS or images — wrong for a screenshot."""
        fetcher = ArchiveFetcher("20100101")
        self.assertNotIn(
            "id_", fetcher.iframe_url(record(datetime(2010, 1, 1, tzinfo=UTC)))
        )


class TestFailureMapping(unittest.TestCase):
    """Failures carry stable, groupable codes."""

    def _fetch(self, side_effect=None, html_result=None):
        fetcher = ArchiveFetcher("20100101")
        target = "piedomains.fetchers.ArchiveFetcher._fetch_html"
        with patch(target, side_effect=side_effect, return_value=html_result):
            return asyncio.run(fetcher.fetch_single("https://cnn.com"))

    def test_missing_snapshot_is_reported_not_guessed(self):
        result = self._fetch(html_result=(None, "", "no 200 capture near 20100101"))
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, ErrorCode.NO_ARCHIVE_SNAPSHOT.value)
        self.assertIn("20100101", result.error)

    def test_rate_limit_maps_to_its_own_code(self):
        from wayback.exceptions import RateLimitError

        result = self._fetch(side_effect=RateLimitError("429", 60))
        self.assertEqual(result.error_code, ErrorCode.ARCHIVE_RATE_LIMITED.value)

    def test_robots_block_maps_to_robots_blocked(self):
        from wayback.exceptions import BlockedByRobotsError

        result = self._fetch(side_effect=BlockedByRobotsError("blocked"))
        self.assertEqual(result.error_code, ErrorCode.ROBOTS_BLOCKED.value)


class TestSuccessfulFetch(unittest.TestCase):
    """A good fetch reports the capture it actually used."""

    def test_reports_realized_timestamp_and_extracts_text(self):
        ts = datetime(2010, 1, 1, 4, 17, 27, tzinfo=UTC)
        # A realistic amount of copy: the archive path now applies the same 30-token
        # floor as the live path, and the old three-word fixture is exactly the stub
        # that floor exists to reject.
        html = (
            "<html><head><title>CNN</title></head><body>"
            "breaking news today london paris tokyo markets sports weather travel finance health science culture music films books politics economy business technology europe asia america africa research university hospital government election climate energy transport housing education justice defence agriculture"
            "</body></html>"
        )
        fetcher = ArchiveFetcher("20100101")
        with patch.object(
            ArchiveFetcher, "_fetch_html", return_value=(record(ts), html, "")
        ):
            result = asyncio.run(fetcher.fetch_single("https://cnn.com"))

        self.assertTrue(result.success)
        # The realized capture, not the requested date
        self.assertEqual(result.snapshot_timestamp, "20100101041727")
        self.assertNotEqual(result.snapshot_timestamp[:8], "")
        self.assertEqual(result.title, "CNN")
        self.assertEqual(result.html, html)

    def test_no_screenshot_requested_means_no_browser(self):
        ts = datetime(2010, 1, 1, tzinfo=UTC)
        fetcher = ArchiveFetcher("20100101")
        with (
            patch.object(
                ArchiveFetcher, "_fetch_html", return_value=(record(ts), "<html/>", "")
            ),
            patch.object(ArchiveFetcher, "_screenshot") as shot,
        ):
            asyncio.run(fetcher.fetch_single("https://cnn.com"))
        shot.assert_not_called()


class TestCacheKeySeparation(unittest.TestCase):
    """Different archive dates must not overwrite each other."""

    def test_stem_includes_the_archive_date(self):
        from piedomains.data_collector import DataCollector

        live = DataCollector(cache_dir="/tmp/pd-key-test")
        old = DataCollector(cache_dir="/tmp/pd-key-test", archive_date="20050101")
        new = DataCollector(cache_dir="/tmp/pd-key-test", archive_date="20150101")

        self.assertEqual(live._cache_stem("cnn.com"), "cnn.com")
        self.assertEqual(old._cache_stem("cnn.com"), "cnn.com@20050101")
        self.assertEqual(new._cache_stem("cnn.com"), "cnn.com@20150101")
        self.assertNotEqual(old._cache_stem("cnn.com"), new._cache_stem("cnn.com"))
        self.assertNotEqual(live._cache_stem("cnn.com"), old._cache_stem("cnn.com"))


class TestScreenshotReporting(unittest.TestCase):
    """A failed screenshot must not be advertised as a usable path."""

    def _collect(self, screenshot_path):
        import tempfile
        from unittest.mock import MagicMock

        from piedomains.data_collector import DataCollector
        from piedomains.fetchers import FetchResult

        collector = DataCollector(cache_dir=tempfile.mkdtemp(), archive_date="20100101")
        collector.fetcher = MagicMock()
        collector.fetcher.fetch_both.return_value = FetchResult(
            url="https://cnn.com",
            success=True,
            html="<html><body>news</body></html>",
            text="news",
            screenshot_path=screenshot_path,
            snapshot_timestamp="20100101041727",
        )
        return collector._collect_single_domain("cnn.com", use_cache=False)

    def test_captured_screenshot_is_reported(self):
        row = self._collect("/tmp/whatever/cnn.com@20100101.png")
        self.assertIsNotNone(row["image_path"])

    def test_failed_screenshot_reports_none_not_a_dead_path(self):
        row = self._collect("")
        self.assertTrue(row["fetch_success"])
        self.assertIsNone(row["image_path"])

    def test_snapshot_timestamp_is_surfaced_in_collection(self):
        row = self._collect("/tmp/whatever/cnn.com@20100101.png")
        self.assertEqual(row["snapshot_timestamp"], "20100101041727")


class TestArchivedThinContent(unittest.TestCase):
    """An archived stub is a failure, not a document.

    The live path has refused thin pages since 0.7.0, but archived text is fetched as raw
    HTML rather than rendered, so it never reached that check. ``sapphirecasino.com`` came
    back as a success from a 114-byte capture holding four characters of text -- a blank
    screenshot and an empty document, presented as data and then trained on.
    """

    def _fetch(self, html):
        import asyncio
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        from piedomains.fetchers import ArchiveFetcher

        record = MagicMock()
        record.timestamp = datetime(2025, 8, 7, 16, 39, 49, tzinfo=UTC)
        record.raw_url = "http://web.archive.org/web/20250807163949id_/http://x.com/"

        fetcher = ArchiveFetcher("20250807")
        with patch.object(fetcher, "_fetch_html", return_value=(record, html, "")):
            return asyncio.run(fetcher.fetch_single("x.com"))

    def test_a_stub_capture_is_refused(self):
        result = self._fetch("<html><body>Coming soon</body></html>")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "thin_content")
        self.assertIn("below the 30-word floor", result.error)

    def test_a_real_capture_still_succeeds(self):
        # Distinct words, because the cleaner deduplicates: 60 repetitions of one word
        # survive as a single token, so the floor is really 30 *distinct* tokens.
        result = self._fetch(
            "<html><body>breaking news today london paris tokyo markets sports weather travel finance health science culture music films books politics economy business technology europe asia america africa research university hospital government election climate energy transport housing education justice defence agriculture</body></html>"
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.snapshot_timestamp, "20250807163949")


if __name__ == "__main__":
    unittest.main()
