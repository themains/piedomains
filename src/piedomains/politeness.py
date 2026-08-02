#!/usr/bin/env python3
"""Honour robots.txt, and space out requests to one host.

**Why this did not exist.** The live fetch path never looked at robots.txt, never delayed
between requests to the same host, and sent a user-agent impersonating Chrome on macOS
while Chromium launched with ``--disable-blink-features=AutomationControlled``. The
project's own position, stated in ``blocking.py``, is that evasion is not the answer --
"detection plus an archive.org fallback is the honest response rather than an evasion arms
race" -- so the flag contradicted the package's stated ethics as well as its behaviour.

``ErrorCode.ROBOTS_BLOCKED`` has existed in ``outcomes.py`` since the taxonomy was written
and was only ever produced by the *archive* path, for archive.org's own refusals. The
vocabulary anticipated this; only the enforcement was missing.

**Nothing here is hand-rolled.** ``protego`` is Scrapy's robots.txt parser: pure Python,
maintained, and correct on the parts that are easy to get wrong -- longest-match
precedence, ``*``/``$`` wildcards, and ``Crawl-delay``. The sibling R package writes its own
only because R's ``robotstxt`` fetches with its own HTTP stack, bypassing every timeout and
guard; ``protego`` parses a string and leaves fetching to us, so that objection does not
apply.

The fetch uses ``aiohttp``, which is already a declared dependency and was imported
nowhere.

**Concurrency note.** Every public entry point in ``fetchers.py`` creates a fresh event
loop, so per-host locks must be created lazily inside the running loop rather than eagerly
in ``__init__`` -- a ``Lock`` bound to a dead loop raises when awaited from a new one.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

__all__ = ["HostThrottle", "RobotsCache", "user_agent_token"]

#: Cap the robots body. RFC 9309 says 500 KiB; more than that is not a robots file.
MAX_ROBOTS_BYTES = 512_000

#: How long a parsed robots.txt stays good.
ROBOTS_TTL_SECONDS = 86_400


def user_agent_token(user_agent: str) -> str:
    """Extract the product token a robots.txt group would name.

    Args:
        user_agent: Full user-agent string.

    Returns:
        str: The leading product token, lowercased, or the whole string if there is no
        ``/`` separator.
    """
    head = user_agent.split()[0] if user_agent.split() else user_agent
    return head.split("/")[0].lower()


@dataclass
class _Entry:
    """One origin's parsed rules."""

    rules: Any
    fetched_at: float
    reachable: bool


@dataclass
class RobotsCache:
    """Fetch, parse and cache robots.txt per origin.

    Fail-closed where we cannot know: a 5xx, a timeout or an unparseable body disallows.
    A 4xx allows, which is what RFC 9309 specifies and is overwhelmingly the common case
    (most hosts simply have no robots.txt).

    Attributes:
        user_agent: The full user-agent string we send.
        timeout: Seconds to wait for robots.txt itself.
        ttl: Seconds a parsed result stays good.
    """

    user_agent: str
    timeout: float = 10.0
    ttl: float = ROBOTS_TTL_SECONDS
    _cache: dict[str, _Entry] = field(default_factory=dict)

    async def _download(self, origin: str) -> tuple[str | None, bool]:
        """Fetch one robots.txt.

        Args:
            origin: Scheme and host, e.g. ``https://example.com``.

        Returns:
            tuple[str | None, bool]: The body (``None`` when there is none to parse) and
            whether the host answered at all.
        """
        import aiohttp

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {"User-Agent": self.user_agent}
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
                # allow_redirects=False: aiohttp follows by default, and a redirected
                # robots.txt is not worth a second place for a hostile host to send us.
                # RFC 9309 permits following up to five hops; treating a 3xx as "no
                # robots file" is the safer reading and costs almost nothing in practice.
                session.get(f"{origin}/robots.txt", allow_redirects=False) as response,
            ):
                if response.status >= 500:
                    return None, False
                if response.status >= 400:
                    # No robots.txt means everything is allowed.
                    return "", True
                body = await response.content.read(MAX_ROBOTS_BYTES + 1)
                if len(body) > MAX_ROBOTS_BYTES:
                    return None, False
                return body.decode("utf-8", errors="replace"), True
        except Exception as exc:
            # Fail *open* when the host itself is unreachable, and let the real fetch
            # report the accurate reason. Failing closed here would label an
            # unresolvable domain `robots_blocked`, which says the host refused us when
            # in fact there is no host. Genuine refusals -- a 5xx, an oversized or
            # unparseable body -- still fail closed above.
            logger.debug("robots.txt unreachable for %s: %s", origin, exc)
            return "", True

    async def rules_for(self, url: str) -> tuple[Any, bool]:
        """Return parsed rules for a URL's origin.

        Args:
            url: Any URL on the origin.

        Returns:
            tuple[Any, bool]: The parsed rules and whether the origin answered.
        """
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"

        hit = self._cache.get(origin)
        if hit is not None and (time.monotonic() - hit.fetched_at) < self.ttl:
            return hit.rules, hit.reachable

        body, reachable = await self._download(origin)
        rules: Any = None
        if body is not None:
            try:
                from protego import Protego

                rules = Protego.parse(body)
            except Exception as exc:  # an unparseable body is not permission
                logger.debug("robots.txt parse failed for %s: %s", origin, exc)
                reachable = False

        self._cache[origin] = _Entry(rules, time.monotonic(), reachable)
        return rules, reachable

    async def allowed(self, url: str) -> bool:
        """Whether we may fetch this URL.

        Args:
            url: The URL to check.

        Returns:
            bool: True when robots.txt permits it.
        """
        rules, reachable = await self.rules_for(url)
        if not reachable:
            return False
        if rules is None:
            return True
        return bool(rules.can_fetch(url, user_agent_token(self.user_agent)))

    async def crawl_delay(self, url: str) -> float | None:
        """The host's requested delay, if it states one.

        Args:
            url: Any URL on the origin.

        Returns:
            float | None: Seconds, or ``None`` when unspecified.
        """
        rules, _ = await self.rules_for(url)
        if rules is None:
            return None
        delay = rules.crawl_delay(user_agent_token(self.user_agent))
        return float(delay) if delay is not None else None


class HostThrottle:
    """Space out requests to the same host, and bound total concurrency.

    Two separate limits, because they answer different questions. The semaphore bounds how
    much work we do at once; the per-host lock plus delay bounds how hard we lean on any
    one server. ``max_parallel`` in this project bounds browser *contexts*, not
    navigations -- one page is opened per URL and all of them are gathered at once -- so
    without this a batch of 500 hits 500 pages simultaneously.

    Locks are created on first use so they bind to whatever loop is running, since every
    public entry point here builds a fresh one.
    """

    def __init__(
        self, delay: float = 1.0, max_concurrent: int = 8, jitter: float = 0.25
    ):
        """Set the limits.

        Args:
            delay: Minimum seconds between requests to one host.
            max_concurrent: Maximum simultaneous fetches across all hosts.
            jitter: Random extra delay, so a crawl does not arrive as a metronome.
        """
        self.delay = delay
        self.jitter = jitter
        self._max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    def _sem(self) -> asyncio.Semaphore:
        """The global semaphore, created inside the running loop.

        Returns:
            asyncio.Semaphore: The shared limiter.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def wait(self, url: str, delay: float | None = None) -> None:
        """Block until it is polite to request this URL.

        Args:
            url: The URL about to be fetched.
            delay: Override the configured delay, e.g. with a host's ``Crawl-delay``.
        """
        host = urlsplit(url).netloc.lower()
        wanted = max(self.delay, delay or 0.0)

        lock = self._locks.get(host)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[host] = lock

        async with lock:
            previous = self._last.get(host)
            now = time.monotonic()
            if previous is not None:
                remaining = wanted - (now - previous)
                if remaining > 0:
                    await asyncio.sleep(remaining + random.random() * self.jitter)  # noqa: S311
            self._last[host] = time.monotonic()

    def limit(self):
        """Context manager bounding total concurrency.

        Returns:
            asyncio.Semaphore: Usable with ``async with``.
        """
        return self._sem()
