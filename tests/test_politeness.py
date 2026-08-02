#!/usr/bin/env python

"""Tests for robots.txt compliance and per-host throttling.

The live fetch path had none of this: robots.txt was never fetched, there was no delay
between requests to one host, and the user-agent impersonated Chrome while Chromium
launched with an automation-hiding flag. ``ErrorCode.ROBOTS_BLOCKED`` existed in the
taxonomy from the start and was only ever produced by the archive path.

Every test but the last patches ``aiohttp`` and reaches no network at all. The last one
runs a stdlib server on loopback, because the property it checks -- that a disallowed URL
is never requested -- can only be shown by a server that can report what it was asked for.
"""

import asyncio
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from unittest.mock import patch

from piedomains.politeness import HostThrottle, RobotsCache, user_agent_token

AGENT = "piedomains/0.13 (+https://github.com/themains/piedomains)"


def run(coro):
    """Run a coroutine to completion."""
    return asyncio.run(coro)


def cache_with(body, reachable=True):
    """A RobotsCache whose download step returns a fixed body."""
    cache = RobotsCache(user_agent=AGENT)

    async def fake(origin):
        return body, reachable

    cache._download = fake  # type: ignore[method-assign]
    return cache


class TestUserAgentToken(unittest.TestCase):
    """robots.txt groups name a product token, not a whole UA string."""

    def test_extracts_the_product_token(self):
        self.assertEqual(user_agent_token(AGENT), "piedomains")
        self.assertEqual(user_agent_token("rdomains/0.5.0 (+url)"), "rdomains")
        self.assertEqual(user_agent_token("bare"), "bare")


class TestRobotsRules(unittest.TestCase):
    """Parsing is protego's job; these pin the behaviour we depend on."""

    def test_disallow_is_honoured(self):
        cache = cache_with("User-agent: *\nDisallow: /private\n")
        self.assertFalse(run(cache.allowed("https://x.com/private/page")))
        self.assertTrue(run(cache.allowed("https://x.com/public")))

    def test_longest_match_wins(self):
        cache = cache_with("User-agent: *\nDisallow: /a\nAllow: /a/ok\n")
        self.assertFalse(run(cache.allowed("https://x.com/a/no")))
        self.assertTrue(run(cache.allowed("https://x.com/a/ok")))

    def test_a_group_naming_us_beats_the_wildcard(self):
        cache = cache_with(
            "User-agent: *\nDisallow:\n\nUser-agent: piedomains\nDisallow: /\n"
        )
        self.assertFalse(run(cache.allowed("https://x.com/anything")))

    def test_crawl_delay_is_read(self):
        cache = cache_with("User-agent: *\nCrawl-delay: 5\n")
        self.assertEqual(run(cache.crawl_delay("https://x.com/")), 5.0)

    def test_no_robots_file_allows_everything(self):
        cache = cache_with("")
        self.assertTrue(run(cache.allowed("https://x.com/anything")))

    def test_results_are_cached_per_origin(self):
        cache = RobotsCache(user_agent=AGENT)
        calls = []

        async def fake(origin):
            calls.append(origin)
            return "User-agent: *\nDisallow: /no\n", True

        cache._download = fake  # type: ignore[method-assign]
        run(cache.allowed("https://x.com/a"))
        run(cache.allowed("https://x.com/b"))
        run(cache.allowed("https://other.com/a"))
        self.assertEqual(calls, ["https://x.com", "https://other.com"])


class TestFailureDirection(unittest.TestCase):
    """Which way to fail is the whole design, and it differs by cause."""

    def test_a_server_error_fails_closed(self):
        # The host is there and broken. We do not know what it permits, so we do not go.
        cache = cache_with(None, reachable=False)
        self.assertFalse(run(cache.allowed("https://x.com/")))

    def test_an_unreachable_host_fails_open(self):
        # There is no host. Failing closed here would report `robots_blocked` -- that the
        # host refused us -- when the truth is a DNS failure the fetch will report
        # accurately. Two integration tests caught exactly this.
        cache = RobotsCache(user_agent=AGENT)

        async def boom(origin):
            raise OSError("Name or service not known")

        cache._download = boom  # type: ignore[method-assign]
        # _download's own handler converts the exception; call the real one.
        real = RobotsCache(user_agent=AGENT)
        with patch("aiohttp.ClientSession", side_effect=OSError("no such host")):
            self.assertTrue(run(real.allowed("https://no-such-host.invalid/")))

    def test_an_unparseable_body_is_not_permission(self):
        cache = RobotsCache(user_agent=AGENT)

        async def fake(origin):
            return "User-agent: *\nDisallow: /\n", True

        cache._download = fake  # type: ignore[method-assign]
        with patch("protego.Protego.parse", side_effect=ValueError("bad")):
            self.assertFalse(run(cache.allowed("https://x.com/")))


class TestHostThrottle(unittest.TestCase):
    """max_parallel bounds browser contexts, not navigations. This bounds navigations."""

    def test_requests_to_one_host_are_spaced(self):
        throttle = HostThrottle(delay=0.3, jitter=0.0)

        async def two():
            import time

            await throttle.wait("https://x.com/a")
            start = time.monotonic()
            await throttle.wait("https://x.com/b")
            return time.monotonic() - start

        self.assertGreaterEqual(run(two()), 0.3)

    def test_different_hosts_do_not_wait_on_each_other(self):
        throttle = HostThrottle(delay=0.5, jitter=0.0)

        async def two():
            import time

            await throttle.wait("https://a.com/")
            start = time.monotonic()
            await throttle.wait("https://b.com/")
            return time.monotonic() - start

        self.assertLess(run(two()), 0.3)

    def test_the_semaphore_binds_to_the_running_loop(self):
        # Every public entry point in fetchers.py creates a fresh event loop, so state
        # built eagerly in __init__ would be bound to a dead one.
        throttle = HostThrottle(delay=0.0, jitter=0.0)

        async def once():
            async with throttle.limit():
                return True

        self.assertTrue(run(once()))
        self.assertTrue(asyncio.run(once()))


class _RobotsServer(BaseHTTPRequestHandler):
    """Serves a robots.txt that disallows everything, and records every path asked for."""

    hits: ClassVar[list] = []

    def do_HEAD(self):
        self._respond(body=False)

    def do_GET(self):
        self._respond(body=True)

    def _respond(self, *, body):
        type(self).hits.append(f"{self.command} {self.path}")
        if self.path == "/robots.txt":
            payload = b"User-agent: *\nDisallow: /\n"
            content_type = "text/plain"
        else:
            payload = b"<html><body>" + b"page content " * 20 + b"</body></html>"
            content_type = "text/html"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if body:
            self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestRobotsDecidesBeforeAnythingIsRequested(unittest.TestCase):
    """A disallowed URL must never be requested -- not even by the content preflight.

    The preflight in ``_validate_url_security`` used to run first, so a host whose robots
    file disallowed ``/`` was sent a HEAD, and possibly a ranged GET, before the fetcher
    reported ``robots_blocked``. Asking robots for permission after taking the thing it
    guards is not compliance, and only a server that records its requests can tell the
    difference -- the return value was already correct.
    """

    def setUp(self):
        _RobotsServer.hits = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _RobotsServer)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_a_disallowed_url_is_never_requested(self):
        from piedomains.config import configure, get_config
        from piedomains.fetchers import PlaywrightFetcher
        from piedomains.outcomes import ErrorCode

        original = get_config().get("check_addresses", True)
        try:
            # The address guard would refuse a loopback literal long before robots got a
            # say, which would make this pass for the wrong reason.
            configure(check_addresses=False, obey_robots=True)
            result = run(
                PlaywrightFetcher().fetch_single(f"http://127.0.0.1:{self.port}/page")
            )
        finally:
            configure(check_addresses=original)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, ErrorCode.ROBOTS_BLOCKED.value)
        # The assertion that carries the fix: robots.txt was read, and nothing else was
        # touched. No browser launched either -- /page appears in no form.
        self.assertEqual(_RobotsServer.hits, ["GET /robots.txt"])


if __name__ == "__main__":
    unittest.main()
