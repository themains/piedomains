#!/usr/bin/env python

"""Offline tests for recovering unfetchable pages from archive.org.

Six of the 44 domains in `tests/eval/labels.csv` serve an anti-bot interstitial
instead of content, and no amount of header tuning gets past them -- DataDome
and Cloudflare fingerprint headless Chromium itself. archive.org already holds
those pages, so the fallback fetches them there instead of evading anyone.

Two properties matter enough to pin down here:

* a capture from years ago is *not* a stand-in for the live page, and
* archive.org being rate-limited says nothing about the domain, so it must not
  harden into a terminal `cannot_classify` the caller cannot act on.
"""

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from piedomains.fetchers import ArchiveFetcher, FetchResult, PlaywrightFetcher
from piedomains.outcomes import RETRYABLE, ErrorCode


def record(ts: datetime, url="https://etsy.com/"):
    """Build a stand-in CdxRecord.

    Args:
        ts: Capture timestamp.
        url: Original URL the capture is of.

    Returns:
        SimpleNamespace: An object shaped like a ``CdxRecord``.
    """
    return SimpleNamespace(
        timestamp=ts, original=url, status_code=200, mimetype="text/html"
    )


def blocked(domain="etsy.com") -> FetchResult:
    """Build the FetchResult a detected bot wall produces.

    Args:
        domain: Domain that was blocked.

    Returns:
        FetchResult: A failed, bot-blocked result.
    """
    return FetchResult(
        url=f"https://{domain}/",
        success=False,
        error="datadome challenge page served instead of content",
        error_code=ErrorCode.BOT_BLOCKED.value,
    )


def run(coro):
    """Run a coroutine to completion.

    Args:
        coro: The coroutine to run.

    Returns:
        Any: Whatever the coroutine returned.
    """
    return asyncio.run(coro)


class TestFallbackTriggers(unittest.TestCase):
    """Which live-fetch verdicts the archive is asked to answer."""

    def test_success_is_left_alone(self):
        ok = FetchResult(url="https://cnn.com/", success=True, text="real content")
        out = run(PlaywrightFetcher()._archive_fallback(ok))
        self.assertIs(out, ok)
        self.assertEqual(out.source, "live")

    def test_thin_content_does_not_trigger_it(self):
        """A page that loaded but said little is not an acquisition failure."""
        thin = FetchResult(
            url="https://x.com/",
            success=False,
            error_code=ErrorCode.THIN_CONTENT.value,
        )
        with patch.object(ArchiveFetcher, "fetch_single") as archive:
            out = run(PlaywrightFetcher()._archive_fallback(thin))
        archive.assert_not_called()
        self.assertIs(out, thin)

    def test_timeout_triggers_it(self):
        """monster.com timed out rather than being walled; same missing page."""
        timed_out = FetchResult(
            url="https://monster.com/",
            success=False,
            error_code=ErrorCode.TIMEOUT.value,
        )
        recovered = FetchResult(
            url="https://web.archive.org/...",
            success=True,
            text="jobs hiring employers resume",
            snapshot_timestamp="20260726153506",
            source="archive",
        )
        with patch.object(ArchiveFetcher, "fetch_single", return_value=recovered):
            out = run(PlaywrightFetcher()._archive_fallback(timed_out))
        self.assertTrue(out.success)
        self.assertEqual(out.source, "archive")

    def test_disabled_by_config(self):
        fetcher = PlaywrightFetcher()
        with (
            patch.dict(fetcher.config._config, {"archive_fallback": False}),
            patch.object(ArchiveFetcher, "fetch_single") as archive,
        ):
            out = run(fetcher._archive_fallback(blocked()))
        archive.assert_not_called()
        self.assertEqual(out.error_code, ErrorCode.BOT_BLOCKED.value)


class TestRecovery(unittest.TestCase):
    """What a successful recovery looks like to the caller."""

    def test_recovered_row_is_marked_as_archived(self):
        recovered = FetchResult(
            url="https://web.archive.org/web/20260727020309id_/https://etsy.com/",
            success=True,
            text="handmade vintage shop sellers",
            snapshot_timestamp="20260727020309",
            source="archive",
        )
        with patch.object(ArchiveFetcher, "fetch_single", return_value=recovered):
            out = run(PlaywrightFetcher()._archive_fallback(blocked()))

        self.assertTrue(out.success)
        self.assertEqual(out.source, "archive")
        self.assertEqual(out.snapshot_timestamp, "20260727020309")
        # Keyed on what was asked for, not the playback URL, or the cache key
        # and the result row would disagree about which domain this is.
        self.assertEqual(out.url, "https://etsy.com/")

    def test_fallback_exception_leaves_the_block_verdict(self):
        with patch.object(
            ArchiveFetcher, "fetch_single", side_effect=RuntimeError("boom")
        ):
            out = run(PlaywrightFetcher()._archive_fallback(blocked()))
        self.assertFalse(out.success)
        self.assertEqual(out.error_code, ErrorCode.BOT_BLOCKED.value)


class TestTerminalVersusRetryable(unittest.TestCase):
    """Only a definitive "the archive lacks this" is allowed to be terminal."""

    def test_no_snapshot_becomes_cannot_classify(self):
        empty = FetchResult(
            url="https://sciencemag.org/",
            success=False,
            error="no 200 capture near 20260726",
            error_code=ErrorCode.NO_ARCHIVE_SNAPSHOT.value,
        )
        with patch.object(ArchiveFetcher, "fetch_single", return_value=empty):
            out = run(PlaywrightFetcher()._archive_fallback(blocked("sciencemag.org")))
        self.assertEqual(out.error_code, ErrorCode.CANNOT_CLASSIFY.value)
        self.assertIn("no 200 capture", out.error)

    def test_rate_limiting_stays_retryable(self):
        """archive.org throttling us says nothing about the domain."""
        throttled = FetchResult(
            url="https://etsy.com/",
            success=False,
            error="archive.org rate limited",
            error_code=ErrorCode.ARCHIVE_RATE_LIMITED.value,
        )
        with patch.object(ArchiveFetcher, "fetch_single", return_value=throttled):
            out = run(PlaywrightFetcher()._archive_fallback(blocked()))
        self.assertEqual(out.error_code, ErrorCode.BOT_BLOCKED.value)
        self.assertIn(ErrorCode(out.error_code), RETRYABLE)


class TestStalenessBound(unittest.TestCase):
    """How old a capture may be and still stand in for the live page."""

    def test_recent_capture_is_accepted(self):
        target = datetime(2026, 7, 26, tzinfo=UTC)
        fetcher = ArchiveFetcher("20260726", max_age_days=365)
        with (
            patch.object(
                fetcher,
                "find_closest_record",
                return_value=record(target - timedelta(days=2)),
            ),
            patch.object(fetcher, "_client") as client,
        ):
            memento = client.return_value.__enter__.return_value.get_memento
            memento.return_value = SimpleNamespace(
                text="<html>shop</html>", close=lambda: None
            )
            rec, html, why = fetcher._fetch_html("etsy.com")
        self.assertIsNotNone(rec)
        self.assertEqual(why, "")

    def test_stale_capture_is_refused(self):
        """sciencemag.org's nearest capture is from 2021 -- not today's site."""
        target = datetime(2026, 7, 26, tzinfo=UTC)
        fetcher = ArchiveFetcher("20260726", max_age_days=365)
        with patch.object(
            fetcher,
            "find_closest_record",
            return_value=record(target - timedelta(days=1790)),
        ):
            rec, html, why = fetcher._fetch_html("sciencemag.org")
        self.assertIsNone(rec)
        self.assertEqual(html, "")
        self.assertIn("older than the 365-day limit", why)

    def test_no_bound_accepts_any_capture(self):
        """A deliberate historical request wants the old capture."""
        target = datetime(2010, 1, 1, tzinfo=UTC)
        fetcher = ArchiveFetcher("20100101")
        self.assertIsNone(fetcher.max_age)
        with (
            patch.object(
                fetcher,
                "find_closest_record",
                return_value=record(target - timedelta(days=300)),
            ),
            patch.object(fetcher, "_client") as client,
        ):
            memento = client.return_value.__enter__.return_value.get_memento
            memento.return_value = SimpleNamespace(
                text="<html>2009 cnn</html>", close=lambda: None
            )
            rec, _, why = fetcher._fetch_html("cnn.com")
        self.assertIsNotNone(rec)
        self.assertEqual(why, "")


if __name__ == "__main__":
    unittest.main()
