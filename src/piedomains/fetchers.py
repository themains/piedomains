#!/usr/bin/env python3
"""Playwright-based page fetcher for content extraction.

Supports live content fetching and archive.org historical snapshots.
Unified pipeline for HTML, text extraction, and screenshots.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import Page, async_playwright
from wayback import (  # pyright: ignore[reportMissingImports]
    CdxRecord,
    Mode,
    WaybackClient,
    WaybackSession,
)
from wayback.exceptions import (  # pyright: ignore[reportMissingImports]
    BlockedByRobotsError,
    BlockedSiteError,
    MementoPlaybackError,
    NoMementoError,
    RateLimitError,
    WaybackException,
)

from .blocking import detect_block, is_thin
from .config import get_config
from .content_validation import ContentValidator
from .outcomes import ErrorCode
from .piedomains_logging import get_logger

logger = get_logger()

#: Live-fetch verdicts the archive can answer. A bot wall is the motivating
#: case, but a timeout or a refused connection leaves us equally without a page,
#: and archive.org is just as good an answer there.
_ARCHIVE_RECOVERABLE: frozenset[str] = frozenset(
    {
        ErrorCode.BOT_BLOCKED.value,
        ErrorCode.TIMEOUT.value,
        ErrorCode.CONNECTION_ERROR.value,
        ErrorCode.DNS_ERROR.value,
        ErrorCode.HTTP_ERROR.value,
    }
)


@dataclass
class FetchResult:
    """Result from a single fetch operation."""

    url: str
    success: bool
    html: str = ""
    text: str = ""  # Clean extracted text
    screenshot_path: str = ""
    title: str = ""
    meta_description: str = ""
    error: str = ""
    #: Stable reason this fetch failed, from :class:`piedomains.outcomes.ErrorCode`.
    error_code: str = ""
    #: For archive fetches, the capture actually used (``YYYYMMDDHHMMSS``). The
    #: requested date is not echoed here — this is what was really retrieved.
    snapshot_timestamp: str = ""
    #: Where the content came from: ``live`` or ``archive``. A live fetch that
    #: hit a bot wall and was recovered from archive.org reports ``archive``, so
    #: a caller can tell which rows are not the site as it stands today.
    source: str = "live"


class BaseFetcher:
    """Base class for content fetchers with security validation."""

    def __init__(self):
        """Initialize fetcher with content validator."""
        self.config = get_config()
        self.validator = ContentValidator(self.config)

    def _context_kwargs(self) -> dict:
        """Build the browser-context settings every capture path shares.

        Three call sites used to spell this out independently, which is how they drift:
        a setting added to one is silently absent from the others, and screenshots taken
        by the batch path stop matching those taken by the single path.

        Returns:
            dict: Keyword arguments for ``browser.new_context``.
        """
        return {
            "user_agent": self.config.user_agent,
            "viewport": self.config.get(
                "playwright_viewport", {"width": 1280, "height": 1024}
            ),
            "device_scale_factor": self.config.get("screenshot_scale", 1),
        }

    def _validate_url_security(
        self,
        url: str,
        *,
        force_fetch: bool = False,
        allow_content_types: list[str] | None = None,
        ignore_extensions: bool = False,
    ) -> tuple[bool, str]:
        """Validate URL security before fetching content.

        Args:
            url: URL to validate.
            force_fetch: Skip content-safety validation.
            allow_content_types: Override the allowed content types.
            ignore_extensions: Skip file-extension validation.

        Returns:
            tuple: (is_safe, error_message_or_warnings)
        """
        try:
            validation_result = self.validator.validate_url(
                url,
                force_fetch=force_fetch,
                allow_content_types=allow_content_types,
                ignore_extensions=ignore_extensions,
            )

            if not validation_result.is_safe:
                logger.warning(
                    f"Security validation failed for {url}: {validation_result.error_message}"
                )
                return False, validation_result.error_message

            if validation_result.warnings:
                warning_msg = "; ".join(validation_result.warnings)
                logger.info(f"Security warnings for {url}: {warning_msg}")

            if validation_result.sandbox_recommended:
                sandbox_cmd = self.validator.get_sandbox_command(url)
                logger.warning(
                    f"Sandbox execution recommended for {url}: {sandbox_cmd}"
                )

            return True, (
                "; ".join(validation_result.warnings)
                if validation_result.warnings
                else ""
            )

        except Exception as e:
            error_msg = f"Security validation error: {e}"
            logger.error(f"Validation error for {url}: {error_msg}")
            return False, error_msg

    def _parse_domain_name(self, url_or_domain: str) -> str:
        """Extract clean domain name from URL or domain string.

        Args:
            url_or_domain: URL or domain name

        Returns:
            str: Clean domain name
        """
        # Import here to avoid circular imports
        from .piedomain import Piedomain

        return Piedomain.parse_url_to_domain(url_or_domain)

    async def fetch_single(
        self, url: str, screenshot_path: str | None = None
    ) -> FetchResult:
        """Fetch HTML, text and (optionally) a screenshot for one URL.

        Args:
            url: URL to fetch.
            screenshot_path: Where to write the screenshot, if wanted.

        Returns:
            FetchResult: The fetch outcome.

        Raises:
            NotImplementedError: Always; subclasses provide the implementation.
        """
        raise NotImplementedError

    async def fetch_batch(
        self, urls: list[str], cache_dir: str = "cache"
    ) -> list[FetchResult]:
        """Fetch several URLs concurrently.

        Args:
            urls: URLs to fetch.
            cache_dir: Directory screenshots are written under.

        Returns:
            list[FetchResult]: One result per URL, in input order.

        Raises:
            NotImplementedError: Always; subclasses provide the implementation.
        """
        raise NotImplementedError

    async def _archive_fallback(
        self, result: FetchResult, screenshot_path: str | None = None
    ) -> FetchResult:
        """Recover a page the live fetch could not get, from archive.org.

        DataDome, Cloudflare and Akamai fingerprint headless Chromium itself, so
        no amount of header tuning gets past them. archive.org already holds
        these pages, which is the answer that does not involve evading anyone.
        The same applies to a domain that simply timed out.

        Captures older than ``archive_max_age_days`` are refused: a page from
        years ago is not a stand-in for the site as it stands today.

        Args:
            result: The live fetch outcome. Returned unchanged unless the live
                fetch failed for a reason the archive can answer.
            screenshot_path: Where to write the archived screenshot, if wanted.

        Returns:
            FetchResult: The archived content on success, otherwise ``result``
            with its verdict sharpened to say what the archive offered.
        """
        if result.error_code not in _ARCHIVE_RECOVERABLE:
            return result
        if not self.config.get("archive_fallback", True):
            return result

        domain = self._parse_domain_name(result.url)
        max_age = self.config.get("archive_max_age_days", 365)
        fetcher = ArchiveFetcher(
            datetime.now(UTC).strftime("%Y%m%d"), max_parallel=1, max_age_days=max_age
        )
        try:
            recovered = await fetcher.fetch_single(result.url, screenshot_path)
        except Exception as e:  # the block verdict is the fallback's fallback
            logger.warning(
                f"{domain}: archive.org fallback failed: {e}",
                extra={"domain": domain, "stage": "fetch"},
            )
            return result

        if not recovered.success:
            logger.warning(
                f"{domain}: live fetch failed and archive.org has nothing usable "
                f"({recovered.error})",
                extra={"domain": domain, "stage": "fetch", "source": "archive"},
            )
            result.error = f"{result.error}; {recovered.error}"
            # Only a definitive "the archive does not have this" is terminal.
            # archive.org being rate-limited or unreachable says nothing about
            # the domain, so that stays retryable rather than being frozen into
            # a verdict the caller cannot act on.
            if recovered.error_code == ErrorCode.NO_ARCHIVE_SNAPSHOT.value:
                result.error_code = ErrorCode.CANNOT_CLASSIFY.value
            return result

        logger.info(
            f"{domain}: live fetch failed, recovered from archive.org capture "
            f"{recovered.snapshot_timestamp}",
            extra={
                "domain": domain,
                "stage": "fetch",
                "source": "archive",
                "snapshot_timestamp": recovered.snapshot_timestamp,
            },
        )
        # Keep the caller's URL rather than the archive playback URL, so cache
        # keys and result rows stay keyed on the domain that was asked for.
        recovered.url = result.url
        return recovered

    def _politeness(self) -> tuple:
        """Robots cache and throttle for this fetcher, built on first use.

        Built lazily rather than in ``__init__`` because every public entry point creates
        a fresh event loop, and an ``asyncio.Lock`` bound to a dead loop raises when
        awaited from a new one.

        Returns:
            tuple: ``(RobotsCache, HostThrottle)``.
        """
        if getattr(self, "_politeness_state", None) is None:
            from .politeness import HostThrottle, RobotsCache

            config = self.config
            self._politeness_state = (
                RobotsCache(
                    user_agent=config.get("user_agent", "piedomains"),
                    timeout=float(config.get("http_timeout", 10)),
                ),
                HostThrottle(
                    delay=float(config.get("crawl_delay", 1.0)),
                    max_concurrent=int(config.get("max_concurrent_fetches", 8)),
                ),
            )
        return self._politeness_state

    async def _robots_allow(self, url: str) -> bool:
        """Whether robots.txt permits fetching this URL.

        Args:
            url: The URL about to be fetched.

        Returns:
            bool: True when permitted, or when robots compliance is switched off.
        """
        if not self.config.get("obey_robots", True):
            return True
        robots, _ = self._politeness()
        try:
            return await robots.allowed(url)
        except Exception:  # never let the robots check itself break a fetch
            return True

    async def _be_polite(self, url: str) -> None:
        """Wait until it is polite to request this URL.

        Args:
            url: The URL about to be fetched.
        """
        robots, throttle = self._politeness()
        delay = None
        if self.config.get("obey_robots", True):
            try:
                delay = await robots.crawl_delay(url)
                cap = float(self.config.get("max_crawl_delay", 30.0))
                if delay is not None and delay > cap:
                    delay = cap
            except Exception:
                delay = None
        await throttle.wait(url, delay)

    def cleanup(self) -> None:
        """Release any resources held by the fetcher.

        Browsers are opened and closed inside each ``async with
        async_playwright()`` block, so there is nothing left to release; this
        exists so callers can invoke it unconditionally.
        """
        return

    def fetch_both(self, url: str, output_path: str, **kwargs) -> FetchResult:
        """Synchronously fetch HTML, text and a screenshot for one URL.

        Defined on the base class so that both live and archive fetchers expose
        it — ``DataCollector`` calls this on whatever ``get_fetcher`` returned.

        Args:
            url: URL or bare domain to fetch.
            output_path: Where to write the screenshot.
            **kwargs: Accepted for call-site compatibility; unused.

        Returns:
            FetchResult: The fetch outcome.
        """
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.fetch_single(url, output_path))
        finally:
            loop.close()


class PlaywrightFetcher(BaseFetcher):
    """Unified Playwright fetcher for all content extraction."""

    def __init__(self, max_parallel: int = 4):
        """Initialize Playwright fetcher.

        Args:
            max_parallel: Maximum number of parallel browser contexts
        """
        super().__init__()
        self.max_parallel = max_parallel or self.config.get("max_parallel", 4)

    async def _configure_page(self, page: Page) -> None:
        """Configure page with security and performance settings."""
        # Block heavy resources
        blocked_resources = self.config.get(
            "block_resources", ["media", "video", "font", "websocket", "manifest"]
        )

        async def handle_route(route):
            if route.request.resource_type in blocked_resources:
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", handle_route)

        # Block known video/streaming domains and file extensions
        video_patterns = [
            "*youtube.com/*",
            "*youtube-nocookie.com/*",
            "*vimeo.com/*",
            "*dailymotion.com/*",
            "*twitch.tv/*",
            "*netflix.com/*",
            "*hulu.com/*",
            "*.mp4",
            "*.webm",
            "*.avi",
            "*.mov",
            "*.mkv",
            "*.flv",
            "*.wmv",
            "*.m4v",
            "*.mpg",
            "*.mpeg",
            "*.3gp",
        ]

        for pattern in video_patterns:
            await page.route(pattern, lambda route: route.abort())

        # Set reasonable headers
        await page.set_extra_http_headers(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )

    @staticmethod
    async def _content_after_navigation(page: Page, timeout: int) -> str:
        """Read page HTML, tolerating a redirect in flight.

        Args:
            page: The page to read.
            timeout: Navigation timeout in milliseconds.

        Returns:
            str: The page HTML, or ``""`` if it could not be read.

        Raises:
            Exception: Re-raised when the failure is not a navigation race, or
                when the retries are exhausted.
        """
        for attempt in range(3):
            try:
                return await page.content()
            except Exception as e:
                if "navigating" not in str(e).lower() or attempt == 2:
                    raise
                # Let the redirect land, then read the destination.
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state(
                        "domcontentloaded", timeout=min(timeout, 10000)
                    )
                await page.wait_for_timeout(500)
        return ""

    async def _capture_screenshot(self, page: Page, screenshot_path: str | None) -> str:
        """Write a screenshot of the loaded page, if one was asked for.

        Separate from text extraction because the two succeed independently: a
        page can render perfectly and yield almost no text, and that page is
        still the whole input to the screenshot-only classifier.

        Args:
            page: The loaded Playwright page.
            screenshot_path: Where to write the PNG, or ``None`` to skip.

        Returns:
            str: The path written, or ``""`` if none was.
        """
        if not screenshot_path:
            return ""
        Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(
            path=screenshot_path,
            full_page=False,  # Just viewport for consistency
            type="png",
        )
        logger.info(f"Screenshot saved to {screenshot_path}")
        return screenshot_path

    async def _extract_from_page(
        self, page: Page, url: str, screenshot_path: str | None = None
    ) -> FetchResult:
        """Extract all content from a loaded page.

        Args:
            page: The loaded Playwright page.
            url: URL being fetched, for logging.
            screenshot_path: Where to write the screenshot, if wanted.

        Returns:
            FetchResult: The extracted content.
        """
        result = FetchResult(url=url, success=False)

        try:
            timeout = self.config.get("playwright_timeout", 30000)
            # `networkidle` loses whole sites: theverge, stackoverflow and
            # weather.com all time out on it (3 of 10 popular sites tested), and
            # outlook.com yields 1 usable token against 414 with this strategy.
            # Wait for the DOM, then race a short network-quiet window -- capped,
            # because nytimes.com does yield more text once the network settles,
            # so the upside is kept without the 20s cliff.
            # Both fetch_single and fetch_batch navigate here, so one throttle covers
            # both. max_parallel bounds browser contexts, not navigations -- a page is
            # opened per URL and all are gathered at once -- so without this a batch of
            # 500 arrives simultaneously.
            await self._be_polite(url)
            async with self._politeness()[1].limit():
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            settle = self.config.get("settle_ms", 1500)
            quiet = self.config.get("network_quiet_ms", 3000)
            await page.wait_for_timeout(settle)
            # A chatty page never going quiet is fine -- we already have the DOM.
            with contextlib.suppress(Exception):
                await page.wait_for_load_state("networkidle", timeout=quiet)

            # Extract HTML. A site that redirects after DOMContentLoaded (e.g.
            # outlook.com -> login) can be mid-navigation here, and page.content()
            # raises "Unable to retrieve content because the page is navigating".
            # `networkidle` used to mask this by waiting for the redirect to
            # finish; wait explicitly instead of going back to it.
            result.html = await self._content_after_navigation(page, timeout)

            blocked = detect_block(result.html, domain=self._parse_domain_name(url))
            if blocked:
                result.success = False
                result.error = blocked.reason
                result.error_code = ErrorCode.BOT_BLOCKED.value
                logger.warning(
                    f"{url}: {blocked.reason}",
                    extra={
                        "domain": self._parse_domain_name(url),
                        "stage": "fetch",
                        "error_code": "bot_blocked",
                        "vendor": blocked.vendor,
                    },
                )
                return result

            # Extract text content using page evaluation
            result.text = await page.evaluate(
                """() => {
                // Remove script, style, and other non-content elements
                const elementsToRemove = document.querySelectorAll(
                    'script, style, noscript, iframe, object, embed, applet'
                );
                elementsToRemove.forEach(el => el.remove());

                // Get text content from body
                const body = document.body;
                if (!body) return '';

                // Get text but preserve some structure
                let textContent = body.innerText || body.textContent || '';

                // Clean up excessive whitespace
                textContent = textContent.replace(/\\n\\n\\n+/g, '\\n\\n');
                textContent = textContent.trim();

                return textContent;
            }"""
            )

            # A page whose entire rendered body is a handful of words did not
            # load as text; caching that makes the failure permanent, which is
            # how spotify.com came to sit in the cache with 8 usable tokens
            # against 292 on a live refetch.
            #
            # It is still a page, though, and it renders. Screenshot-only
            # classification exists precisely for callers with no text -- bot
            # walls, image-only landing pages, JS-heavy sites -- so failing the
            # whole fetch here made the image path useless on exactly the pages
            # it is for. `espn.com` renders 11 words and a complete homepage.
            #
            # So capture the screenshot first, then drop the text and HTML
            # rather than the result. The collector writes HTML only when
            # `result.html` is non-empty, so nothing thin is cached either way,
            # and the text classifier applies this same floor again before
            # scoring.
            floor = self.config.get("min_tokens", 30)
            thin = is_thin(result.text, min_tokens=floor)
            if thin:
                words = len(result.text.split())
                result.screenshot_path = await self._capture_screenshot(
                    page, screenshot_path
                )
                result.error = (
                    f"page rendered only {words} words, below the {floor}-word floor"
                )
                result.error_code = ErrorCode.THIN_CONTENT.value
                result.success = bool(result.screenshot_path)
                result.html = ""
                result.text = ""
                logger.warning(
                    f"{url}: {result.error}"
                    + (
                        "; keeping the screenshot for image classification"
                        if result.success
                        else ""
                    ),
                    extra={
                        "domain": self._parse_domain_name(url),
                        "stage": "fetch",
                        "error_code": ErrorCode.THIN_CONTENT.value,
                    },
                )
                return result

            # Extract metadata
            result.title = await page.title()

            # Get meta description
            meta_desc = await page.query_selector('meta[name="description"]')
            if meta_desc:
                desc_content = await meta_desc.get_attribute("content")
                result.meta_description = desc_content or ""

            # Also try og:description if regular description is empty
            if not result.meta_description:
                og_desc = await page.query_selector('meta[property="og:description"]')
                if og_desc:
                    og_content = await og_desc.get_attribute("content")
                    result.meta_description = og_content or ""

            result.screenshot_path = await self._capture_screenshot(
                page, screenshot_path
            )

            result.success = True
            logger.info(
                f"Successfully extracted content from {url} "
                f"(HTML: {len(result.html)} chars, Text: {len(result.text)} chars)"
            )

        except Exception as e:
            # Name it. Without this a Playwright navigation timeout reaches the
            # caller as `unknown`, which hides the single most common fetch
            # failure and stops the archive fallback from even being tried.
            from .outcomes import classify_exception

            result.error = str(e)
            result.error_code = classify_exception(e).value
            logger.error(
                f"Failed to extract content from {url}: {e}",
                extra={
                    "domain": self._parse_domain_name(url),
                    "stage": "fetch",
                    "error_code": result.error_code,
                },
            )

        return result

    async def fetch_single(
        self, url: str, screenshot_path: str | None = None
    ) -> FetchResult:
        """Fetch content from a single URL."""
        # Normalize URL - add https:// if no protocol
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        # Security validation
        is_safe, msg = self._validate_url_security(url)
        if not is_safe:
            return FetchResult(url=url, success=False, error=msg)

        if not await self._robots_allow(url):
            return FetchResult(
                url=url,
                success=False,
                error="disallowed by robots.txt",
                error_code=ErrorCode.ROBOTS_BLOCKED.value,
            )

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.config.get("playwright_headless", True),
                # No automation-hiding flag: blocking.py states this project detects bot
                # walls and falls back to archive.org rather than evading them.
                args=[],
            )

            context = await browser.new_context(
                **self._context_kwargs(),
                ignore_https_errors=False,
            )

            page = await context.new_page()

            # Configure security and performance
            await self._configure_page(page)

            # Extract everything
            result = await self._extract_from_page(page, url, screenshot_path)

            await context.close()
            await browser.close()

        # Outside the browser block on purpose: the fallback launches its own
        # browser for the archived screenshot, and nesting them is needless.
        return await self._archive_fallback(result, screenshot_path)

    async def fetch_batch(
        self, urls: list[str], cache_dir: str = "cache"
    ) -> list[FetchResult]:
        """Fetch multiple URLs in parallel."""
        # Normalize and validate all URLs first
        validated_urls = []
        results = []

        for url in urls:
            # Normalize URL - add https:// if no protocol
            if not url.startswith(("http://", "https://")):
                normalized_url = f"https://{url}"
            else:
                normalized_url = url

            is_safe, msg = self._validate_url_security(normalized_url)
            if is_safe and not await self._robots_allow(normalized_url):
                results.append(
                    FetchResult(
                        url=normalized_url,
                        success=False,
                        error="disallowed by robots.txt",
                        error_code=ErrorCode.ROBOTS_BLOCKED.value,
                    )
                )
            elif is_safe:
                validated_urls.append(normalized_url)
            else:
                logger.warning(f"Skipping unsafe URL {normalized_url}: {msg}")
                results.append(
                    FetchResult(url=normalized_url, success=False, error=msg)
                )

        if not validated_urls:
            return results

        logger.info(
            f"Starting batch fetch for {len(validated_urls)} URLs "
            f"with {self.max_parallel} parallel workers"
        )

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.config.get("playwright_headless", True),
                # No automation-hiding flag: blocking.py states this project detects bot
                # walls and falls back to archive.org rather than evading them.
                args=[],
            )

            # Create parallel contexts
            contexts = []
            for i in range(min(self.max_parallel, len(validated_urls))):
                context = await browser.new_context(
                    **self._context_kwargs(),
                    ignore_https_errors=False,
                )
                contexts.append(context)
                logger.debug(f"Created context {i + 1}/{self.max_parallel}")

            # Process URLs in parallel
            tasks = []
            screenshot_paths = []
            for i, url in enumerate(validated_urls):
                context = contexts[i % len(contexts)]
                page = await context.new_page()
                await self._configure_page(page)

                # Generate screenshot path
                domain = self._parse_domain_name(url)
                screenshot_path = f"{cache_dir}/images/{domain}.png"
                screenshot_paths.append(screenshot_path)

                # Create task
                task = self._extract_from_page(page, url, screenshot_path)
                tasks.append(task)
                logger.debug(f"Created task for {url}")

            # Execute all tasks
            logger.info(f"Executing {len(tasks)} parallel fetch tasks")
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            fetched: list[FetchResult] = []
            for i, result in enumerate(batch_results):
                if isinstance(result, BaseException):
                    url = validated_urls[i]
                    logger.error(f"Task failed for {url}: {result}")
                    fetched.append(
                        FetchResult(url=url, success=False, error=str(result))
                    )
                else:
                    fetched.append(result)

            # Cleanup
            logger.debug("Cleaning up browser contexts")
            for context in contexts:
                await context.close()
            await browser.close()

        # Bot-walled domains get one archive.org attempt each, after the live
        # browser is down. Serialized: archive.org is rate-limited, and the
        # blocked population is small by construction.
        for i, result in enumerate(fetched):
            if result.error_code == ErrorCode.BOT_BLOCKED.value:
                fetched[i] = await self._archive_fallback(result, screenshot_paths[i])
        results.extend(fetched)

        logger.info(
            f"Batch fetch complete. Success: {sum(r.success for r in results)}/{len(results)}"
        )
        return results

    # Synchronous wrapper methods for compatibility
    def fetch_html(self, url: str, **kwargs) -> tuple[bool, str, str]:
        """Sync wrapper for HTML fetching."""
        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.fetch_single(url))
            return result.success, result.html, result.error
        finally:
            loop.close()

    def fetch_content(self, url: str, **kwargs) -> FetchResult:
        """Sync wrapper for content fetching (alias for fetch_single)."""
        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.fetch_single(url))
            return result
        finally:
            loop.close()

    def fetch_screenshot(
        self, url: str, output_path: str, **kwargs
    ) -> tuple[bool, str]:
        """Sync wrapper for screenshot."""
        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.fetch_single(url, output_path))
            return result.success, result.error
        finally:
            loop.close()


class ArchiveFetcher(BaseFetcher):
    """Fetch historical snapshots from archive.org via the CDX + Memento APIs.

    Snapshot discovery, closest-date matching, rate limiting and backoff are
    delegated to the ``wayback`` library rather than hand-rolled. Two playback
    modes are used deliberately:

    * **text/HTML** uses ``Mode.original`` (the ``id_`` suffix) — the raw
      capture, with no injected Wayback JavaScript and no rewritten URLs, so no
      browser and no toolbar-stripping are needed.
    * **screenshots** use the ``if_`` suffix, which suppresses the toolbar but
      keeps asset references pointing at archived copies, so the page still
      renders as it did. ``id_`` would screenshot a page stripped of its CSS
      and images.

    Only captures with HTTP status 200 are considered; a domain whose only
    captures are redirects or error pages fails loudly rather than having an
    archived 404 classified as content.
    """

    def __init__(
        self,
        target_date: str | datetime,
        max_parallel: int | None = None,
        max_age_days: int | None = None,
    ):
        """Initialize archive fetcher.

        Args:
            target_date: Target date as 'YYYYMMDD' string or datetime object
            max_parallel: Maximum concurrent snapshot fetches
            max_age_days: Reject a capture further than this many days from
                ``target_date``. ``None`` (the default) accepts whatever the
                search window returns, which is what a deliberate historical
                request wants. The bot-wall fallback sets it, because a capture
                from years ago is not a stand-in for the live page.
        """
        super().__init__()
        if isinstance(target_date, datetime):
            self.target_date = target_date.strftime("%Y%m%d")
        else:
            self.target_date = target_date

        self.target = datetime.strptime(self.target_date, "%Y%m%d").replace(tzinfo=UTC)
        self.max_parallel = max_parallel or self.config.get("archive_max_parallel", 2)
        self.search_window = timedelta(days=self.config.get("archive_window_days", 365))
        self.max_age = timedelta(days=max_age_days) if max_age_days else None
        self._session_kwargs = {
            "retries": self.config.get("archive_retries", 3),
            "backoff": self.config.get("archive_backoff", 2),
            "search_calls_per_second": self.config.get("archive_search_rate", 1),
            "memento_calls_per_second": self.config.get("archive_memento_rate", 4),
        }

    def _client(self) -> WaybackClient:
        """Build a rate-limited Wayback client.

        Returns:
            WaybackClient: A client configured from the archive settings.
        """
        return WaybackClient(session=WaybackSession(**self._session_kwargs))

    def iframe_url(self, record: CdxRecord) -> str:
        """Build the ``if_`` playback URL used for screenshots.

        Args:
            record: The CDX record to render.

        Returns:
            str: A Wayback URL that renders without the toolbar but with
            archived assets intact.
        """
        stamp = record.timestamp.strftime("%Y%m%d%H%M%S")
        return f"https://web.archive.org/web/{stamp}if_/{record.original}"

    def find_closest_record(self, url: str) -> CdxRecord | None:
        """Find the status-200 capture nearest the target date.

        Args:
            url: URL or bare domain to look up.

        Returns:
            The nearest ``CdxRecord``, or ``None`` if the domain has no usable
            capture in the search window.
        """
        with self._client() as client:
            records = list(
                client.search(
                    url,
                    from_date=self.target - self.search_window,
                    to_date=self.target + self.search_window,
                    filter_field="statuscode:200",
                    collapse="timestamp:8",
                )
            )
        if not records:
            return None
        return min(records, key=lambda r: abs(r.timestamp - self.target))

    def _fetch_html(self, url: str) -> tuple[CdxRecord | None, str, str]:
        """Retrieve the raw archived HTML for a URL.

        Args:
            url: URL or bare domain to fetch.

        Returns:
            tuple: ``(record, html, error)``; ``record`` is ``None`` when no
            usable capture exists, and ``error`` then says why.
        """
        record = self.find_closest_record(url)
        if record is None:
            return None, "", f"no 200 capture near {self.target_date}"

        # Refuse a capture too far from the target rather than passing a page
        # from years ago off as the current site.
        if (
            self.max_age is not None
            and abs(record.timestamp - self.target) > self.max_age
        ):
            stamp = record.timestamp.strftime("%Y-%m-%d")
            return (
                None,
                "",
                f"nearest 200 capture is {stamp}, older than the "
                f"{self.max_age.days}-day limit",
            )

        with self._client() as client:
            memento = client.get_memento(record, mode=Mode.original)
            if memento is None:
                # CDX listed the capture but playback returned nothing; treat it
                # as "no usable snapshot" rather than dereferencing None.
                return (
                    None,
                    "",
                    f"capture {record.timestamp:%Y%m%d%H%M%S} would not play back",
                )
            try:
                return record, memento.text, ""
            finally:
                memento.close()

    async def _screenshot(self, record: CdxRecord, screenshot_path: str) -> str:
        """Render the archived page and write a screenshot.

        Args:
            record: The CDX record to render.
            screenshot_path: Where to write the PNG.

        Returns:
            str: The screenshot path on success, or ``""`` on failure.
        """
        timeout = self.config.get("playwright_timeout", 30000)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=self.config.get("playwright_headless", True)
                )
                try:
                    context = await browser.new_context(**self._context_kwargs())
                    page = await context.new_page()

                    # archive.org serves assets slowly; fonts and media are not
                    # needed for a layout screenshot and were the dominant cause
                    # of "waiting for fonts to load" screenshot timeouts.
                    async def _block_heavy(route):
                        if route.request.resource_type in {
                            "font",
                            "media",
                            "websocket",
                            "manifest",
                        }:
                            await route.abort()
                        else:
                            await route.continue_()

                    await page.route("**/*", _block_heavy)

                    await page.goto(
                        self.iframe_url(record),
                        wait_until="domcontentloaded",
                        timeout=timeout,
                    )
                    await page.wait_for_timeout(
                        self.config.get("archive_render_settle_ms", 1500)
                    )
                    Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                    await page.screenshot(
                        path=screenshot_path,
                        full_page=False,
                        type="png",
                        animations="disabled",
                        timeout=self.config.get("archive_screenshot_timeout", 15000),
                    )
                    return screenshot_path
                finally:
                    await browser.close()
        except Exception as e:  # screenshots are best-effort
            logger.warning(f"Archive screenshot failed for {record.original}: {e}")
            return ""

    async def fetch_single(
        self, url: str, screenshot_path: str | None = None
    ) -> FetchResult:
        """Fetch one archived page's HTML, text and optional screenshot.

        Args:
            url: URL or bare domain to fetch.
            screenshot_path: Where to write the screenshot, if wanted.

        Returns:
            FetchResult: The fetch outcome, carrying the realized
            ``snapshot_timestamp``.
        """
        from .outcomes import ErrorCode
        from .text_processor import TextProcessor

        result = FetchResult(url=url, success=False, source="archive")
        try:
            record, html, why = await asyncio.to_thread(self._fetch_html, url)
        except RateLimitError as e:
            result.error = f"archive.org rate limited: {e}"
            result.error_code = ErrorCode.ARCHIVE_RATE_LIMITED.value
            return result
        except (BlockedByRobotsError, BlockedSiteError) as e:
            result.error = f"archive.org blocked this URL: {e}"
            result.error_code = ErrorCode.ROBOTS_BLOCKED.value
            return result
        except (NoMementoError, MementoPlaybackError) as e:
            result.error = f"no usable capture near {self.target_date}: {e}"
            result.error_code = ErrorCode.NO_ARCHIVE_SNAPSHOT.value
            return result
        except WaybackException as e:
            result.error = f"archive.org error: {e}"
            result.error_code = ErrorCode.NO_ARCHIVE_SNAPSHOT.value
            return result

        if record is None:
            result.error = why
            result.error_code = ErrorCode.NO_ARCHIVE_SNAPSHOT.value
            return result

        result.html = html
        result.text = TextProcessor.process_html_to_text(html)
        result.snapshot_timestamp = record.timestamp.strftime("%Y%m%d%H%M%S")

        # The same floor the live path applies. It was missing here because archived text
        # is fetched as raw HTML rather than rendered, so it never passes through
        # `_extract_from_page` where the live check lives -- and a capture of a dead site
        # is often a stub. sapphirecasino.com came back as a "success" from a 114-byte
        # capture with 4 characters of text, which is a blank screenshot and a blank
        # document presented as data.
        #
        # Unlike the live path, this one still fails outright rather than keeping the
        # screenshot. The asymmetry is the point: a live page with no text usually
        # renders fine and is real input for the image model, whereas a 114-byte capture
        # has nothing to render -- the screenshot would be as empty as the text.
        floor = self.config.get("min_tokens", 30)
        if is_thin(result.text, min_tokens=floor):
            result.success = False
            result.error = (
                f"archived capture {result.snapshot_timestamp} has only "
                f"{len(result.text.split())} words, below the {floor}-word floor"
            )
            result.error_code = ErrorCode.THIN_CONTENT.value
            logger.warning(
                f"{url}: {result.error}",
                extra={
                    "domain": url,
                    "stage": "fetch",
                    "error_code": ErrorCode.THIN_CONTENT.value,
                    "snapshot_timestamp": result.snapshot_timestamp,
                },
            )
            return result

        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            result.title = soup.title.string.strip()
        meta = soup.find("meta", attrs={"name": "description"})
        if meta is not None:
            result.meta_description = str(meta.get("content") or "")

        if screenshot_path:
            result.screenshot_path = await self._screenshot(record, screenshot_path)

        result.success = True
        logger.info(
            f"Archived {url} from {result.snapshot_timestamp} "
            f"({len(html)} bytes, {len(result.text)} chars of text)",
            extra={"domain": url, "snapshot_timestamp": result.snapshot_timestamp},
        )
        return result

    async def fetch_batch(
        self, urls: list[str], cache_dir: str = "cache"
    ) -> list[FetchResult]:
        """Fetch several archived pages, bounded by ``max_parallel``.

        Args:
            urls: URLs or bare domains to fetch.
            cache_dir: Directory screenshots are written under.

        Returns:
            list[FetchResult]: One result per URL, in input order.
        """
        semaphore = asyncio.Semaphore(self.max_parallel)
        image_dir = Path(cache_dir) / "images"

        async def one(url: str) -> FetchResult:
            async with semaphore:
                domain = self._parse_domain_name(url)
                # Suffix with the requested date so snapshots of the same domain
                # at different dates do not overwrite one another. Must stay in
                # step with DataCollector._cache_stem.
                stem = f"{domain}@{self.target_date}"
                return await self.fetch_single(url, str(image_dir / f"{stem}.png"))

        results = await asyncio.gather(*(one(u) for u in urls), return_exceptions=True)
        final: list[FetchResult] = []
        for url, item in zip(urls, results, strict=True):
            if isinstance(item, BaseException):
                from .outcomes import classify_exception

                final.append(
                    FetchResult(
                        url=url,
                        success=False,
                        error=str(item),
                        error_code=classify_exception(item).value,
                    )
                )
            else:
                final.append(item)
        return final

    def fetch_html(self, url: str, **kwargs) -> tuple[bool, str, str]:
        """Synchronously fetch archived HTML for one URL.

        Args:
            url: URL or bare domain to fetch.
            **kwargs: Accepted for call-site compatibility; unused.

        Returns:
            tuple: ``(success, html, error)``.
        """
        result = self.fetch_both(url, "")
        return result.success, result.html, result.error


def get_fetcher(
    archive_date: str | datetime | None = None, max_parallel: int = 4
) -> BaseFetcher:
    """Factory function to get appropriate fetcher.

    Args:
        archive_date: If provided, returns ArchiveFetcher for this date.
                     If None, returns PlaywrightFetcher for current content.
        max_parallel: Maximum number of parallel browser contexts

    Returns:
        BaseFetcher: Appropriate fetcher instance
    """
    if archive_date:
        return ArchiveFetcher(archive_date, max_parallel)
    else:
        return PlaywrightFetcher(max_parallel)
