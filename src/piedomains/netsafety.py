#!/usr/bin/env python3
"""Refuse to fetch addresses that are not on the public internet.

**Why this exists.** Nothing in this package checked where a domain actually resolved, so
a domain in a caller's list that pointed at ``127.0.0.1``, ``10.0.0.1`` or the cloud
metadata endpoint ``169.254.169.254`` was fetched. The sibling R package has had this
guard; this closes the gap.

**Nothing here is hand-rolled.** ``ipaddress`` is stdlib and already knows every range that
matters -- RFC1918, loopback, link-local, CGNAT, benchmarking, TEST-NET, IPv6 ULA and
link-local -- and it unwraps IPv4-mapped IPv6 (``::ffff:10.0.0.1``) without a manual step,
which is the step people forget. The R sibling spells these out only because R's equivalent
is a package rather than the standard library.

``is_global`` alone is not enough: multicast reports ``True``. The predicate is
``is_global and not is_multicast and not is_reserved``.

**The check resolves; it does not parse.** ``ipaddress.ip_address("2130706433")`` raises, so
a decimal-encoded address falls through to the hostname path where ``getaddrinfo`` turns it
into ``127.0.0.1`` and the address check catches it. A parse-only guard is bypassed by
``http://2130706433/``. Do not "optimise" the resolve away for hosts that look numeric.

**Every resolved address is checked, not the first.** Chromium resolves independently and
may prefer a different record than the one we looked at, so "the first address was global"
proves nothing about the connection that actually happens.

What this does **not** protect against, and each of these is real:

1. **Server-side redirects.** Chromium follows 3xx hops internally and Playwright does not
   create a ``Route`` for them, so a handler cannot veto one -- verified against a local
   server, which saw ``/evil`` and ``/landing`` while the route handler saw only ``/evil``.
   The hop's body never reaches text extraction, but the blind request is still issued, and
   for an internal endpoint with side effects a blind GET is already harm.
2. **DNS rebinding.** Two independent resolutions sit between our check and the browser's
   connect. This is not closable in-process without a local proxy the browser is pinned to.
3. **Split-horizon or round-robin DNS**, even with no attacker involved.
4. **Public hosts that proxy to internal resources.** An address says nothing about an open
   proxy at a global one.
5. **Service-worker requests**, which ``page.route`` does not intercept.

**The control that does cover all of these already exists, and you should use it.** Set
``proxy_server`` (or ``PIEDOMAINS_PROXY_SERVER``) to an egress proxy such as Stripe's
Smokescreen -- https://github.com/stripe/smokescreen -- which resolves every hostname and
refuses non-publicly-routable addresses. Because the browser issues a fresh ``CONNECT``
for every redirect hop and every subresource, the proxy re-checks each one, which closes
both the redirect gap and the rebinding window that this module cannot. Note that
Smokescreen has itself shipped deny-list bypasses (GHSA-qwrf-gfpj-qvj6,
GHSA-gcj7-j438-hjj2) -- which is the argument for using a maintained implementation rather
than writing another one, not against using a proxy.

This module is the cheap in-process layer that refuses the obvious cases before a browser
is even launched. It is not a substitute for the proxy, and should not be read as one.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .outcomes import ErrorCode

logger = logging.getLogger(__name__)

__all__ = ["AddressGuard", "check_url", "is_global_address"]

#: Hosts that are not on the public internet by construction.
HOST_DENY_SUFFIXES = (".onion", ".local", ".internal", ".localhost")

#: How long a per-host verdict stays good. Memory only -- a cache on disk becomes a
#: staler second source of truth about DNS, and hands anyone who can influence one entry
#: a bypass that outlives the run.
VERDICT_TTL_SECONDS = 300.0


def is_global_address(text: str) -> bool:
    """Whether a string is an IP address we may connect to.

    Args:
        text: An IP literal, in any form ``ipaddress`` accepts.

    Returns:
        bool: True only for globally routable unicast addresses. Anything unparseable is
        False, so a malformed address is never treated as permission.
    """
    try:
        address = ipaddress.ip_address(text.strip())
    except ValueError:
        return False
    return bool(
        address.is_global and not address.is_multicast and not address.is_reserved
    )


def _resolve(host: str) -> list[str]:
    """Resolve a hostname to every address it offers.

    Args:
        host: Hostname to resolve.

    Returns:
        list[str]: Address strings, possibly empty.
    """
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


@dataclass
class AddressGuard:
    """Decide whether a host may be contacted, with a short-lived verdict cache.

    Attributes:
        allow_hosts: Host names exempt from the address check. Matching is on the host
            string only: an exempt host does **not** exempt anything it redirects to.
        ttl: Seconds a verdict stays cached.
        timeout: Seconds to allow for one resolution.
        resolver: Injected in tests so nothing touches DNS.
    """

    allow_hosts: frozenset[str] = frozenset()
    ttl: float = VERDICT_TTL_SECONDS
    timeout: float = 5.0
    resolver: Callable[[str], list[str]] | None = None
    _verdicts: dict[str, tuple[str | None, float]] = field(default_factory=dict)

    async def check_host(self, host: str) -> str | None:
        """Whether we may contact this host.

        Args:
            host: Hostname or IP literal, without scheme or port.

        Returns:
            str | None: ``None`` when safe, otherwise an :class:`ErrorCode` value.
        """
        host = host.strip().lower().strip("[]")
        if not host:
            return ErrorCode.INVALID_DOMAIN.value
        if host in self.allow_hosts:
            return None

        # An IP literal needs no DNS at all, which is the cheap half of the route check.
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return None if is_global_address(host) else ErrorCode.PRIVATE_ADDRESS.value

        cached = self._verdicts.get(host)
        if cached is not None and time.monotonic() < cached[1]:
            return cached[0]

        resolve = self.resolver or _resolve
        try:
            addresses = await asyncio.wait_for(
                asyncio.to_thread(resolve, host), timeout=self.timeout
            )
        except Exception as exc:
            # Fail closed: if we cannot resolve it the browser almost certainly cannot
            # either, so refusing costs nothing real.
            logger.debug("address check could not resolve %s: %s", host, exc)
            verdict: str | None = ErrorCode.DNS_ERROR.value
            self._verdicts[host] = (verdict, time.monotonic() + self.ttl)
            return verdict

        if not addresses:
            verdict = ErrorCode.DNS_ERROR.value
        elif all(is_global_address(a) for a in addresses):
            verdict = None
        else:
            offending = next(a for a in addresses if not is_global_address(a))
            logger.warning("%s resolves to a non-global address (%s)", host, offending)
            verdict = ErrorCode.PRIVATE_ADDRESS.value

        self._verdicts[host] = (verdict, time.monotonic() + self.ttl)
        return verdict

    async def allows_url(self, url: str) -> str | None:
        """Address-only check for a URL, tolerating IP-literal hosts.

        Used for subresources, where a literal public address is unusual but legal and
        blocking it would break a page for no security gain.

        Args:
            url: The URL about to be requested.

        Returns:
            str | None: ``None`` when safe, otherwise an :class:`ErrorCode` value.
        """
        host = urlsplit(url).hostname
        if not host:
            return None
        return await self.check_host(host)


async def check_url(url: str, *, guard: AddressGuard) -> str | None:
    """Full entry-point check for a URL we are about to fetch.

    Stricter than :meth:`AddressGuard.allows_url`: a bare IP literal is refused outright,
    because a domain classifier has nothing to say about one.

    Args:
        url: The URL to check.
        guard: The guard holding the exemptions and the verdict cache.

    Returns:
        str | None: ``None`` when safe, otherwise an :class:`ErrorCode` value.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return ErrorCode.INVALID_DOMAIN.value

    host = (parts.hostname or "").strip().lower()
    if not host:
        return ErrorCode.INVALID_DOMAIN.value
    if host in guard.allow_hosts:
        return None
    if host.endswith(HOST_DENY_SUFFIXES) or "." not in host:
        return ErrorCode.INVALID_DOMAIN.value
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return ErrorCode.INVALID_DOMAIN.value

    return await guard.check_host(host)
