#!/usr/bin/env python

"""Tests for the address guard.

Three layers, none of which touch the public network:

1. the predicate, pure -- the table mirrors the R sibling's, because the two packages are
   meant to refuse the same things;
2. ``check_url`` with an injected resolver, so the real code runs and only the syscall is
   stubbed;
3. a stdlib HTTP server on loopback, which is the only layer that proves the guard refuses
   *before anything reaches the network* -- the assertion is that the server recorded no
   hits at all.
"""

import asyncio
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from piedomains.netsafety import AddressGuard, check_url, is_global_address
from piedomains.outcomes import RETRYABLE, ErrorCode


def run(coro):
    """Run a coroutine to completion."""
    return asyncio.run(coro)


class TestPredicate(unittest.TestCase):
    """`is_global_address` is the whole guard; everything else is plumbing."""

    def test_rejects_addresses_that_are_not_public(self):
        for text in (
            "127.0.0.1",  # loopback
            "10.1.2.3",  # RFC1918
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",  # cloud metadata
            "100.64.0.1",  # CGNAT
            "198.18.0.1",  # benchmarking
            "192.0.2.1",  # TEST-NET
            "0.0.0.0",  # noqa: S104 -- an address under test, not a bind
            "224.0.0.1",  # multicast -- is_global alone reports True for this
            "240.0.0.1",
            "255.255.255.255",
            "::1",
            "fe80::1",
            "fc00::1",
            "ff02::1",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_global_address(text))

    def test_accepts_public_addresses(self):
        for text in ("8.8.8.8", "1.1.1.1", "93.184.216.34", "2001:4860:4860::8888"):
            with self.subTest(text=text):
                self.assertTrue(is_global_address(text))

    def test_unwraps_ipv4_mapped_ipv6(self):
        # The step people forget, and the reason we do it by hand: `is_global` reports
        # False for the whole ::ffff:0:0/96 block on 3.11 and 3.12, and delegates to the
        # embedded address from 3.13. This assertion failed on CI and passed locally
        # until the unwrap became explicit.
        self.assertFalse(is_global_address("::ffff:10.0.0.1"))
        self.assertFalse(is_global_address("::ffff:127.0.0.1"))
        self.assertTrue(is_global_address("::ffff:8.8.8.8"))

    def test_other_ipv4_in_ipv6_encodings_are_refused_outright(self):
        # 6to4 and Teredo can carry a private address too, but every version reports the
        # whole block non-global, so they need no unwrapping -- pinned here so a future
        # stdlib change that makes them global does not pass silently.
        for text in (
            "2002:7f00:0001::",  # 6to4 wrapping 127.0.0.1
            "2002:0808:0808::",  # 6to4 wrapping 8.8.8.8 -- refused as well
            "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # Teredo
        ):
            with self.subTest(text=text):
                self.assertFalse(is_global_address(text))

    def test_garbage_is_never_permission(self):
        for text in ("", "   ", "not-an-ip", "999.1.1.1", "1.2.3"):
            with self.subTest(text=text):
                self.assertFalse(is_global_address(text))


class TestCheckUrl(unittest.TestCase):
    """The real code, with only DNS stubbed."""

    @staticmethod
    def guard(addresses, **kwargs):
        return AddressGuard(resolver=lambda host: list(addresses), **kwargs)

    def test_a_public_host_is_allowed(self):
        self.assertIsNone(
            run(check_url("https://example.com", guard=self.guard(["93.184.216.34"])))
        )

    def test_a_host_resolving_privately_is_refused(self):
        self.assertEqual(
            run(check_url("https://example.com", guard=self.guard(["10.0.0.1"]))),
            ErrorCode.PRIVATE_ADDRESS.value,
        )

    def test_any_private_address_refuses_even_with_a_public_one(self):
        # We do not choose which address the browser connects to, so "one of them was
        # global" proves nothing about the connection that actually happens.
        self.assertEqual(
            run(
                check_url(
                    "https://example.com",
                    guard=self.guard(["93.184.216.34", "10.0.0.1"]),
                )
            ),
            ErrorCode.PRIVATE_ADDRESS.value,
        )

    def test_no_addresses_is_a_dns_error(self):
        self.assertEqual(
            run(check_url("https://example.com", guard=self.guard([]))),
            ErrorCode.DNS_ERROR.value,
        )

    def test_a_failing_resolver_fails_closed(self):
        def boom(host):
            raise OSError("resolver down")

        guard = AddressGuard(resolver=boom)
        self.assertIsNotNone(run(check_url("https://example.com", guard=guard)))

    def test_hosts_a_domain_classifier_has_nothing_to_say_about(self):
        guard = self.guard(["93.184.216.34"])
        for url in (
            "ftp://example.com",
            "https://1.2.3.4",  # bare IP literal
            "https://[::1]",
            "https://secret.onion",
            "https://box.local",
            "https://localhost",  # single label
            "http://2130706433/",  # decimal-encoded 127.0.0.1, also single label
        ):
            with self.subTest(url=url):
                self.assertEqual(
                    run(check_url(url, guard=guard)), ErrorCode.INVALID_DOMAIN.value
                )

    def test_allow_hosts_exempts_only_the_named_host(self):
        guard = AddressGuard(
            allow_hosts=frozenset({"internal.example"}),
            resolver=lambda host: ["10.0.0.1"],
        )
        self.assertIsNone(run(check_url("https://internal.example/x", guard=guard)))
        # An exemption does not extend to anything else that resolves privately.
        self.assertEqual(
            run(check_url("https://other.example/x", guard=guard)),
            ErrorCode.PRIVATE_ADDRESS.value,
        )

    def test_verdicts_are_cached_per_host(self):
        calls = []

        def counting(host):
            calls.append(host)
            return ["93.184.216.34"]

        guard = AddressGuard(resolver=counting)
        run(check_url("https://example.com/a", guard=guard))
        run(check_url("https://example.com/b", guard=guard))
        run(check_url("https://other.com/a", guard=guard))
        self.assertEqual(calls, ["example.com", "other.com"])


class TestErrorCode(unittest.TestCase):
    """The code is shared verbatim with the R sibling, so results aggregate."""

    def test_private_address_is_not_retryable(self):
        # It is a property of a DNS record, not a transient condition -- and repeated
        # retries are exactly what a rebinding attacker needs for a favourable roll.
        self.assertEqual(ErrorCode.PRIVATE_ADDRESS.value, "private_address")
        self.assertNotIn(ErrorCode.PRIVATE_ADDRESS, RETRYABLE)


class _Recorder(BaseHTTPRequestHandler):
    """A server that records every path it is asked for."""

    hits: ClassVar[list] = []

    def do_GET(self):
        type(self).hits.append(self.path)
        body = b"<html><body>" + b"real page content " * 20 + b"</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestNothingReachesTheNetwork(unittest.TestCase):
    """The layer a mock cannot fake: a real socket that was never connected to."""

    def setUp(self):
        _Recorder.hits = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_a_refused_url_never_reaches_the_server(self):
        from piedomains.fetchers import PlaywrightFetcher

        result = run(
            PlaywrightFetcher().fetch_single(f"http://127.0.0.1:{self.port}/page")
        )
        self.assertFalse(result.success)
        # A bare IP is refused as invalid_domain before the address check even runs.
        self.assertEqual(result.error_code, ErrorCode.INVALID_DOMAIN.value)
        # The assertion that matters: no browser launched, no preflight fired, and the
        # robots.txt fetch never happened either.
        self.assertEqual(_Recorder.hits, [])

    def test_the_check_can_be_switched_off_for_an_intranet_corpus(self):
        from piedomains.config import configure, get_config
        from piedomains.fetchers import PlaywrightFetcher
        from tests.conftest import browser_available

        # The only test here that needs a real browser: switching the guard off means the
        # fetch proceeds, and proceeding is a page load. CI installs no browsers, so this
        # skips there. Its sibling above -- the one that proves nothing reaches the
        # network -- needs no browser and does run everywhere, which is the right way
        # round: the security assertion is the one that must never be skipped.
        if not browser_available():
            self.skipTest("Playwright browsers not available")

        original = get_config().get("check_addresses", True)
        try:
            configure(check_addresses=False, obey_robots=False)
            run(PlaywrightFetcher().fetch_single(f"http://127.0.0.1:{self.port}/page"))
            self.assertIn("/page", _Recorder.hits)
        finally:
            configure(check_addresses=original, obey_robots=True)


if __name__ == "__main__":
    unittest.main()
