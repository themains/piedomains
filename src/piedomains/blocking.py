"""Detect anti-bot interstitials served in place of real page content.

Without this, a challenge page is classified as though it were the site. The
signatures here were taken from pages this project actually received, not from
documentation:

- ``etsy.com``, ``monster.com`` and ``reuters.com`` return a ~1470-byte DataDome
  interstitial that loads ``captcha-delivery.com`` and whose only visible text
  is the domain name.
- ``indeed.com`` and ``sciencemag.org`` return Cloudflare's ``Just a moment...``
  challenge, carrying ``cf_chl_opt``.
- ``mayoclinic.org`` returns a 296-byte Akamai "Access Denied … Reference #".

Changing the user-agent does not help — these systems fingerprint headless
Chromium itself — so detection plus an archive.org fallback is the honest
response rather than an evasion arms race.

**Markers are tiered, because presence of a vendor is not a block.** ``reddit``,
``walmart``, ``tinyurl`` and ``quora`` all serve real pages (76-677 tokens, real
titles) while embedding reCAPTCHA, PerimeterX or a Cloudflare Turnstile widget.
Treating those as blocks would silently discard good classifications, so a weak
marker only counts when the page also looks like an interstitial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["BlockDetection", "detect_block", "is_thin"]

#: Unambiguous: these appear only on an actual challenge/denial page.
_STRONG_MARKERS: dict[str, tuple[str, ...]] = {
    "datadome": ("captcha-delivery.com", "geo.captcha-delivery"),
    "cloudflare": ("cf_chl_opt", "cf-browser-verification", "cf-please-wait"),
    "imperva": ("incapsula incident id",),
    "akamai": ("reference #18.",),
}

#: Ambiguous: the vendor's script is present, but plenty of healthy pages embed
#: these. Only a block when corroborated by :func:`_looks_like_interstitial`.
_WEAK_MARKERS: dict[str, tuple[str, ...]] = {
    "cloudflare": ("challenges.cloudflare.com",),
    "perimeterx": ("perimeterx", "px-captcha"),
    "datadome": ("datadome",),
    "recaptcha": ("g-recaptcha", "recaptcha/api.js"),
}

#: Titles that are themselves the tell.
_CHALLENGE_TITLES = frozenset(
    {
        "just a moment...",
        "attention required! | cloudflare",
        "access denied",
        "security check",
        "are you a robot?",
        "verifying you are human",
    }
)

#: HTTP statuses that usually mean "blocked", not "broken".
_BLOCK_STATUSES = frozenset({401, 403, 429, 503})

#: Deliberately narrow, and only applied to short pages: phrases like
#: "permission to access" appear in the legal copy of perfectly healthy sites
#: (bankofamerica.com, 491 tokens, matched an earlier looser pattern).
_DENIAL_TEXT = re.compile(
    r"(you don'?t have permission to access|access denied"
    r"|enable javascript and cookies to continue"
    r"|verify you are (?:a )?human)",
    re.IGNORECASE,
)

#: Above this size a page is substantive enough that a weak marker is almost
#: certainly an embedded widget rather than an interstitial.
_INTERSTITIAL_MAX_BYTES = 60_000

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class BlockDetection:
    """The outcome of inspecting a response for an anti-bot interstitial."""

    blocked: bool
    #: Vendor or heuristic that matched, for logging and reporting.
    vendor: str = ""
    #: Human-readable reason, suitable for the row's ``error`` field.
    reason: str = ""

    def __bool__(self) -> bool:
        """Allow ``if detect_block(...):``.

        Returns:
            bool: True when the response was a block.
        """
        return self.blocked


def _title_of(html: str) -> str:
    """Extract a page title, lowercased and stripped.

    Args:
        html: Raw response body.

    Returns:
        str: The title, or ``""`` when absent.
    """
    match = _TITLE_RE.search(html)
    return match.group(1).strip().lower() if match else ""


def _looks_like_interstitial(html: str, domain: str) -> bool:
    """Corroborate a weak marker: does this page look like a challenge?

    Args:
        html: Raw response body.
        domain: Domain requested.

    Returns:
        bool: True when the page is small, or titled like a challenge, or its
        only content is the domain name.
    """
    title = _title_of(html)
    if title in _CHALLENGE_TITLES:
        return True
    if domain and title == domain.strip().lower() and len(html) < 4000:
        return True
    return len(html) < _INTERSTITIAL_MAX_BYTES and bool(_DENIAL_TEXT.search(html))


def detect_block(
    html: str, *, status: int | None = None, domain: str = ""
) -> BlockDetection:
    """Decide whether a response is an anti-bot interstitial rather than content.

    Args:
        html: Raw response body.
        status: HTTP status, when the caller knows it.
        domain: Domain requested, used for the degenerate-title heuristic.

    Returns:
        BlockDetection: Whether this is a block, and which signature matched.
    """
    if not html:
        return BlockDetection(False)

    lowered = html.lower()

    for vendor, markers in _STRONG_MARKERS.items():
        for marker in markers:
            if marker in lowered:
                return BlockDetection(
                    True, vendor, f"{vendor} challenge page served instead of content"
                )

    title = _title_of(html)
    if title in _CHALLENGE_TITLES:
        return BlockDetection(
            True, "challenge_title", f"challenge page titled {title!r}"
        )

    if status in _BLOCK_STATUSES:
        return BlockDetection(True, "http", f"HTTP {status} — request was refused")

    if _looks_like_interstitial(html, domain):
        for vendor, markers in _WEAK_MARKERS.items():
            for marker in markers:
                if marker in lowered:
                    return BlockDetection(
                        True, vendor, f"{vendor} interstitial served instead of content"
                    )
        # Small page whose only content is its own domain name: DataDome's stub.
        if domain and len(html) < 4000 and title == domain.strip().lower():
            return BlockDetection(
                True, "stub", "stub page whose only content is the domain name"
            )
        if _DENIAL_TEXT.search(html):
            return BlockDetection(True, "access_denied", "access-denied page served")

    return BlockDetection(False)


def is_thin(text: str, *, min_tokens: int = 30) -> bool:
    """Report whether extracted text is too sparse to classify honestly.

    Args:
        text: Extracted page text.
        min_tokens: Token floor below which a label is not meaningful.

    Returns:
        bool: True when the text falls below the floor.
    """
    return len(text.split()) < min_tokens


#: Registrars and parking services whose landing pages are unmistakable.
PARKING_SERVICES: tuple[str, ...] = (
    "dan.com",
    "sedo.com",
    "hugedomains",
    "afternic",
    "bodis.com",
    "parkingcrew",
    "wc_landing",
    "domainmarket",
    "undeveloped.com",
    "namesilo.com",
)

#: Phrases a parking page uses about itself. Held to a length limit because a real
#: shop legitimately says "for sale" while running to thousands of words.
PARKING_PHRASES: tuple[str, ...] = (
    "is for sale",
    "buy this domain",
    "may be for sale",
    "domain for sale",
    "this domain is parked",
    "domain parking",
    "inquire about this domain",
    "domain name is for sale",
    "the domain you are looking for is for sale",
    # Found by sampling what the first version missed: orderwith.com, a parked page
    # labelled `drugs`, said "available for sale" rather than "is for sale".
    "available for sale",
    "available to purchase",
    "parked free of charge",
    "is available -- inquire",
)

#: A parked page is a placeholder. Above this it is a real site that happens to
#: mention a sale.
PARKED_MAX_WORDS = 250


def looks_parked(text: str) -> bool:
    """Report whether a page is a domain-parking placeholder rather than a site.

    **Why this is worth its own answer.** 7.9% of the training corpus turned out to be
    parking pages, and they are not spread evenly: 42% of the `drugs` class, 23% of
    `webmail`, 18% of `downloads`. Expired pharmacy domains get parked, so the model
    learned that a "this domain is for sale" template *means* drugs -- which is why
    zappos.com, newlook.com and suicidepreventionlifeline.org all came back as drugs.

    "Parked" is also the honest answer for such a domain. It is a fact about the domain
    that a caller can act on, and it is plainly stated in the page text, unlike the
    hosting and monetisation properties the taxonomy deliberately excludes.

    Args:
        text: Extracted page text, already lowercased by the cleaner.

    Returns:
        bool: True when the page is a parking placeholder.
    """
    lowered = text.lower()
    if any(service in lowered for service in PARKING_SERVICES):
        return True
    # A phrase alone is not enough: length is what separates a placeholder from a shop.
    if len(lowered.split()) > PARKED_MAX_WORDS:
        return False
    return any(phrase in lowered for phrase in PARKING_PHRASES)
