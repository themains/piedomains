#!/usr/bin/env python

"""Tests for anti-bot interstitial detection.

The fixtures below are shaped after pages this project actually received. The
false-positive cases matter as much as the true positives: an earlier version of
this detector flagged reddit, walmart, tinyurl, quora and bankofamerica as
blocked purely because they embed a vendor's script, which would have silently
discarded good classifications.
"""

import unittest

from piedomains.blocking import detect_block, is_thin

# ~1470-byte DataDome interstitial, as served for etsy/monster/reuters.
DATADOME = (
    '<html lang="en"><head><title>etsy.com</title></head><body>'
    "<script>var dd={'host':'geo.captcha-delivery.com'}</script>"
    '<script src="https://ct.captcha-delivery.com/c.js"></script></body></html>'
)

CLOUDFLARE_CHALLENGE = (
    "<html><head><title>Just a moment...</title></head><body>"
    "<script>window._cf_chl_opt={cvId:'3'}</script></body></html>"
)

AKAMAI_DENIED = (
    "<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD><BODY>Access Denied "
    'You don\'t have permission to access "http://www.mayoclinic.org/" on this '
    "server. Reference #18.a9d02e17</BODY></HTML>"
)

# A real page that merely *embeds* a vendor widget — must NOT be flagged.
HEALTHY_WITH_TURNSTILE = (
    "<html><head><title>URL Shortener, Branded Short Links</title></head><body>"
    '<script src="https://challenges.cloudflare.com/turnstile/v0/api.js"></script>'
    + "<p>real content about shortening links</p>" * 400
    + "</body></html>"
)

HEALTHY_WITH_RECAPTCHA = (
    "<html><head><title>Reddit - The heart of the internet</title></head><body>"
    '<div class="g-recaptcha"></div>'
    + "<p>communities posts comments discussion</p>" * 400
    + "</body></html>"
)

# Legal boilerplate containing denial-ish wording on a large healthy page.
HEALTHY_WITH_LEGAL_TEXT = (
    "<html><head><title>Bank of America - Banking, Credit Cards</title></head><body>"
    + "<p>checking savings mortgage loans credit cards investing</p>" * 500
    + "<p>Unauthorized users do not have permission to access this system.</p>"
    "</body></html>"
)


class TestTruePositives(unittest.TestCase):
    """Real interstitials must be caught."""

    def test_datadome_stub(self):
        d = detect_block(DATADOME, domain="etsy.com")
        self.assertTrue(d.blocked)
        self.assertEqual(d.vendor, "datadome")

    def test_cloudflare_challenge(self):
        d = detect_block(CLOUDFLARE_CHALLENGE, domain="indeed.com")
        self.assertTrue(d.blocked)
        self.assertEqual(d.vendor, "cloudflare")

    def test_akamai_access_denied(self):
        self.assertTrue(detect_block(AKAMAI_DENIED, domain="mayoclinic.org"))

    def test_blocking_http_status(self):
        for status in (401, 403, 429, 503):
            with self.subTest(status=status):
                d = detect_block("<html>whatever</html>", status=status)
                self.assertTrue(d.blocked)
                self.assertEqual(d.vendor, "http")


class TestFalsePositives(unittest.TestCase):
    """Embedding a vendor's script is not being blocked by it."""

    def test_turnstile_widget_on_a_real_page(self):
        d = detect_block(HEALTHY_WITH_TURNSTILE, domain="tinyurl.com")
        self.assertFalse(d.blocked, f"flagged as {d.vendor}: {d.reason}")

    def test_recaptcha_widget_on_a_real_page(self):
        d = detect_block(HEALTHY_WITH_RECAPTCHA, domain="reddit.com")
        self.assertFalse(d.blocked, f"flagged as {d.vendor}: {d.reason}")

    def test_denial_wording_in_legal_copy(self):
        d = detect_block(HEALTHY_WITH_LEGAL_TEXT, domain="bankofamerica.com")
        self.assertFalse(d.blocked, f"flagged as {d.vendor}: {d.reason}")

    def test_ordinary_page(self):
        html = "<html><head><title>CNN</title></head><body>news</body></html>"
        self.assertFalse(detect_block(html, domain="cnn.com"))

    def test_empty_response_is_not_a_block(self):
        """Nothing fetched is a fetch failure, not a block."""
        self.assertFalse(detect_block("", domain="x.com"))


class TestThinContent(unittest.TestCase):
    """The token floor below which a label is not meaningful."""

    def test_below_floor(self):
        self.assertTrue(is_thin("one two three"))

    def test_above_floor(self):
        self.assertFalse(is_thin(" ".join(str(i) for i in range(50))))

    def test_threshold_is_configurable(self):
        text = " ".join(["word"] * 10)
        self.assertTrue(is_thin(text, min_tokens=20))
        self.assertFalse(is_thin(text, min_tokens=5))


if __name__ == "__main__":
    unittest.main()
