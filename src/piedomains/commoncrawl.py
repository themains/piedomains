#!/usr/bin/env python3
r"""Fetch a domain's page from Common Crawl.

A third content source beside the live fetch and archive.org, not a replacement for
either. The two archives fail in different places, so a domain missing from one can be
tried in the other, and the row records which answered.

**What Common Crawl is good at**: bulk. The crawl already happened, so reading from
``data.commoncrawl.org`` involves no robots.txt, no bot walls, no throttling and no SSRF
surface -- the entire politeness stack is irrelevant because we are reading S3, not
crawling. For thousands of domains that is decisive; ``capture_screenshots.py`` measured
0.017 domains/sec and ~33 days for this project's own corpus.

**What it is bad at**: being there. Coverage is a sample rather than a census. Probing
CC-MAIN-2026-30 for ``cnn.com`` returned its ``robots.txt`` and no homepage at all. Ask
for a specific domain at a specific date and archive.org is the better bet; ask for a lot
of domains and Common Crawl is.

**The index server is the binding constraint, and this module is shaped around it.** Four
of five probes during development returned 502, 504 or an HTML error page. A single
attempt makes the feature look broken at random, so every index call retries with
backoff, and a failure is reported as :data:`ErrorCode.NO_ARCHIVE_SNAPSHOT` rather than
raised.

**Nothing here parses WARC by hand.** ``warcio`` does that -- it is the maintained
library for the format, and the one-gzip-member-from-a-byte-range case has enough sharp
edges (record types, chunked transfer, content-length vs payload) to make a hand-rolled
split on ``\\r\\n\\r\\n`` a bug waiting to happen.

``cdx-toolkit`` was the obvious candidate for the index half and is **not** used:
``CDXFetcher(source="cc")`` enumerates all 126 crawl indexes on construction and timed out
at ten minutes in testing. Selecting a crawl from ``collinfo.json`` and querying that one
index is three HTTP calls and no wait.
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import date
from typing import Any

from .outcomes import ErrorCode

logger = logging.getLogger(__name__)

__all__ = ["CommonCrawlRecord", "crawls_for", "fetch_record", "find_record"]

#: Where the list of crawls lives. One call, cached for the process.
COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"

#: Where WARC files are served. Byte-range requests against this are the whole point.
DATA_URL = "https://data.commoncrawl.org"

#: Index lookups are flaky enough that a single attempt is not a fair test.
INDEX_RETRIES = 3

_crawls: list[dict[str, str]] | None = None


class CommonCrawlRecord(dict):
    """One index record: ``url``, ``timestamp``, ``filename``, ``offset``, ``length``."""


def crawls_for(near: str | None = None, timeout: float = 60.0) -> list[str]:
    """List crawl ids, most recent first, optionally ordered by closeness to a date.

    Args:
        near: ``YYYYMMDD``. When given, crawls are ordered by how close their own start
            date is to it, so the caller can try the nearest first.
        timeout: Seconds to wait for the crawl list.

    Returns:
        list[str]: Crawl ids such as ``CC-MAIN-2026-25``. Empty when the list is
        unreachable.
    """
    global _crawls
    if _crawls is None:
        import requests

        try:
            response = requests.get(COLLINFO_URL, timeout=timeout)
            response.raise_for_status()
            _crawls = response.json()
        except Exception as exc:
            logger.debug("could not list Common Crawl indexes: %s", exc)
            return []

    ids = [c["id"] for c in (_crawls or []) if "id" in c]
    if not near:
        return ids

    # Ids look like CC-MAIN-2026-25: ISO year and ISO week. Turn both that and the
    # requested date into real dates and compare in days.
    #
    # Two arithmetic shortcuts were tried here and both misordered crawls. `month * 4`
    # puts 31 December in week 48, ranking a week-48 crawl above the one that actually
    # covers the date. `year * 53 + week` then fails at the year boundary, because it
    # invents a 53rd week in years that have 52: against that same date it scores
    # CC-MAIN-2026-04 and CC-MAIN-2025-51 as equally distant when they are 19 and 16 days
    # away. With `max_crawls=3` either error is enough to return content weeks off, or to
    # push a usable capture out of the window entirely. Calendars do not linearise; ask
    # the calendar.
    try:
        target = date(int(near[:4]), int(near[4:6]), int(near[6:8]))
    except (ValueError, IndexError):
        logger.debug("could not read %r as YYYYMMDD; leaving crawls in order", near)
        return ids

    def distance(crawl_id: str) -> int:
        try:
            _, _, year, week = crawl_id.split("-")
            # Monday of that ISO week. fromisocalendar rejects week 53 in a 52-week year,
            # which is a malformed id rather than something to guess at.
            return abs((date.fromisocalendar(int(year), int(week), 1) - target).days)
        except (ValueError, TypeError):
            return 10**9

    return sorted(ids, key=distance)


def find_record(
    domain: str,
    *,
    crawl: str | None = None,
    near: str | None = None,
    timeout: float = 60.0,
    max_crawls: int = 3,
) -> CommonCrawlRecord | None:
    """Find an HTML capture of a domain in Common Crawl.

    Args:
        domain: Domain to look up.
        crawl: A specific crawl id. When given, only that crawl is searched.
        near: ``YYYYMMDD`` to search near. Crawls are tried nearest-first.
        timeout: Seconds per index call.
        max_crawls: How many crawls to try before giving up. Coverage is patchy, so
            trying only one is often a false negative.

    Returns:
        CommonCrawlRecord | None: The record, or ``None`` when nothing was found.
    """
    candidates = (
        [crawl] if crawl else crawls_for(near=near, timeout=timeout)[:max_crawls]
    )
    for candidate in candidates:
        if not candidate:
            continue
        record = _query_index(candidate, domain, timeout=timeout)
        if record is not None:
            return record
    return None


def _query_index(
    crawl: str, domain: str, *, timeout: float
) -> CommonCrawlRecord | None:
    """Ask one crawl's index for an HTML capture.

    Args:
        crawl: Crawl id.
        domain: Domain to look up.
        timeout: Seconds per attempt.

    Returns:
        CommonCrawlRecord | None: The first usable record, or ``None``.
    """
    import requests

    url = f"https://index.commoncrawl.org/{crawl}-index"
    params = {
        "url": domain,
        "matchType": "domain",
        "output": "json",
        "limit": 40,
        "filter": ["status:200", "mime:text/html"],
    }

    delay = 2.0
    for attempt in range(INDEX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            # The index answers 502/504 often enough that these are expected, not
            # exceptional. Retry rather than treating them as "no record".
            if response.status_code in (429, 502, 503, 504):
                logger.debug("index %s returned %s", crawl, response.status_code)
                if attempt == INDEX_RETRIES - 1:
                    return None
                time.sleep(delay)
                delay *= 2
                continue
            if response.status_code == 404:
                return None
            response.raise_for_status()

            for line in response.text.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                data = json.loads(line)
                if _is_usable(data, domain):
                    return CommonCrawlRecord(data)
            return None
        except Exception as exc:
            if attempt == INDEX_RETRIES - 1:
                logger.debug(
                    "Common Crawl index failed for %s in %s: %s", domain, crawl, exc
                )
                return None
            time.sleep(delay)
            delay *= 2
    return None


def _is_usable(data: dict[str, Any], domain: str) -> bool:
    """Whether an index record is a homepage-ish HTML capture we can use.

    Args:
        data: One index record.
        domain: The domain requested.

    Returns:
        bool: True when the record is worth fetching.
    """
    if data.get("status") != "200" or "html" not in str(data.get("mime", "")):
        return False
    if not all(k in data for k in ("filename", "offset", "length")):
        return False
    # Prefer the site root. The index carries every captured path, and a random article
    # is not what "classify this domain" means.
    url = str(data.get("url", ""))
    path = url.split(domain, 1)[-1] if domain in url else url
    return path in ("", "/", "/index.html") or path.startswith(("/?", "/#"))


def fetch_record(record: dict[str, Any], *, timeout: float = 120.0) -> str | None:
    """Fetch one WARC record's HTML by byte range.

    Args:
        record: An index record carrying ``filename``, ``offset`` and ``length``.
        timeout: Seconds to wait.

    Returns:
        str | None: The captured HTML, or ``None``.
    """
    import requests
    from warcio.archiveiterator import ArchiveIterator

    try:
        offset = int(record["offset"])
        length = int(record["length"])
        response = requests.get(
            f"{DATA_URL}/{record['filename']}",
            headers={"Range": f"bytes={offset}-{offset + length - 1}"},
            timeout=timeout,
        )
        response.raise_for_status()

        for entry in ArchiveIterator(io.BytesIO(response.content)):
            if entry.rec_type == "response":
                return entry.content_stream().read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("could not fetch Common Crawl record: %s", exc)
    return None


def fetch_page(
    domain: str,
    *,
    near: str | None = None,
    crawl: str | None = None,
    timeout: float = 120.0,
) -> tuple[str | None, str | None, str | None]:
    """Fetch a domain's page from Common Crawl.

    Args:
        domain: Domain to fetch.
        near: ``YYYYMMDD`` to search near.
        crawl: A specific crawl id.
        timeout: Seconds.

    Returns:
        tuple[str | None, str | None, str | None]: HTML, capture timestamp and the
        captured URL. All ``None`` when there is no usable record -- the caller should
        report :data:`ErrorCode.NO_ARCHIVE_SNAPSHOT`.
    """
    record = find_record(domain, crawl=crawl, near=near, timeout=min(timeout, 60.0))
    if record is None:
        return None, None, None
    html = fetch_record(record, timeout=timeout)
    if not html:
        return None, None, None
    return html, str(record.get("timestamp")), str(record.get("url"))


#: Re-exported so callers do not have to reach into `outcomes` for the one code this
#: module can produce.
NO_RECORD = ErrorCode.NO_ARCHIVE_SNAPSHOT.value
