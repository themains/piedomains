#!/usr/bin/env python

"""Tests for anti-bot interstitial detection.

The fixtures below are shaped after pages this project actually received. The
false-positive cases matter as much as the true positives: an earlier version of
this detector flagged reddit, walmart, tinyurl, quora and bankofamerica as
blocked purely because they embed a vendor's script, which would have silently
discarded good classifications.
"""

import unittest

from piedomains.blocking import detect_block, is_thin, looks_parked, looks_unavailable

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


class TestRefusingToLabelThinPages(unittest.TestCase):
    """Below the token floor the model returns its prior, not a reading.

    On empty input the shipped text model outputs recreation 0.31, shopping
    0.21, porn 0.19 — which is where results like ``facebook.com -> porn`` come
    from. Refusing is the honest answer.
    """

    def _classifier(self):
        from piedomains.text import TextClassifier

        return TextClassifier.__new__(TextClassifier)

    def test_thin_text_is_refused_rather_than_labelled(self):
        from piedomains.outcomes import ErrorCode

        row: dict = {"domain": "facebook.com", "category": None, "error": None}
        self._classifier()._score_or_refuse(row, "facebook.com", "log in sign up")

        self.assertIsNone(row["category"])
        self.assertEqual(row["error_code"], ErrorCode.THIN_CONTENT.value)
        self.assertIn("below the 30-token floor", row["error"])

    def test_empty_text_keeps_its_own_code(self):
        from piedomains.outcomes import ErrorCode

        row: dict = {"domain": "x.com", "category": None, "error": None}
        self._classifier()._score_or_refuse(row, "x.com", "   ")
        self.assertEqual(row["error_code"], ErrorCode.EMPTY_TEXT.value)

    def test_substantial_text_is_scored(self):
        row: dict = {"domain": "cnn.com", "category": None, "error": None}
        text = " ".join(["news politics world business"] * 20)

        classifier = self._classifier()
        classifier._model_input = lambda domain, t: f"{domain} {t}"  # pyright: ignore
        classifier._predict_text = lambda _: {  # pyright: ignore
            "text_label": "news",
            "text_prob": 0.87,
            "text_domain_probs": {},
        }
        classifier._score_or_refuse(row, "cnn.com", text)

        self.assertEqual(row["category"], "news")
        self.assertIsNone(row["error"])


class TestUnavailable(unittest.TestCase):
    """A domain can serve bytes without being a website.

    Distinct from parking, which is specifically *for sale*. These are the other ways:
    a server autoindex, a registrar placeholder, a suspended account, a bare 404. 321 of
    them sat in the training corpus wearing the label of whatever the domain used to be.
    """

    def test_server_artifacts(self):
        for text in (
            "index of / name last modified size description cgi-bin/ 2019-11-19 20:02",
            "index of / name last modified size cgi-bin proudly served by litespeed",
            "object not found! the requested url was not found on this server.",
        ):
            with self.subTest(text=text[:32]):
                self.assertTrue(looks_unavailable(text))

    def test_placeholders_including_non_english(self):
        for text in (
            "namebright.com - next generation domain registration el7z.com is coming soon",
            "account suspended account for domain matbay.com.au has been suspended",
            "this domain is registered at dynadot.com . website coming soon. pokerisland.net",
            "nrl-clan.de diese webpraesenz befindet sich noch im aufbau. bitte versuchen",
        ):
            with self.subTest(text=text[:32]):
                self.assertTrue(looks_unavailable(text))

    def test_length_guard_protects_real_sites(self):
        """Live sites say "coming soon" in passing, and the guard is the whole defence.

        Sampled at a 100-word limit the rule swept in a riding school; a 319-word breeder
        site announcing puppies, an 877-word games portal and a 592-word lottery shop all
        say it too. Precision matters more than recall, because a false positive relabels
        a real page out of its real class.
        """
        breeder = "our puppies are coming soon " + (
            "kennel breeder show champion pedigree litter whelped sire dam " * 12
        )
        self.assertGreater(len(breeder.split()), 60)
        self.assertFalse(looks_unavailable(breeder))

    def test_parking_is_not_swallowed(self):
        """A for-sale page is parked; that is the more specific answer."""
        for_sale = "this domain is for sale contact the owner to inquire"
        self.assertTrue(looks_parked(for_sale))
        self.assertFalse(looks_unavailable(for_sale))


if __name__ == "__main__":
    unittest.main()
