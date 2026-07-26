#!/usr/bin/env python3
"""Playwright-based page fetcher for content extraction.

Supports live content fetching and archive.org historical snapshots.
Unified pipeline for HTML, text extraction, and screenshots.
"""

import asyncio
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

from .config import get_config
from .content_validation import ContentValidator
from .piedomains_logging import get_logger

logger = get_logger()


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


class BaseFetcher:
    """Base class for content fetchers with security validation."""

    def __init__(self):
        """Initialize fetcher with content validator."""
        self.config = get_config()
        self.validator = ContentValidator(self.config)

    def _validate_url_security(
        self,
        url: str,
        *,
        force_fetch: bool = False,
        allow_content_types: list[str] | None = None,
        ignore_extensions: bool = False,
    ) -> tuple[bool, str]:
        """Validate URL security before fetching content.

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
            url_or_domain (str): URL or domain name

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

    async def _extract_from_page(
        self, page: Page, url: str, screenshot_path: str | None = None
    ) -> FetchResult:
        """Extract all content from a loaded page."""
        result = FetchResult(url=url, success=False)

        try:
            # Navigate with timeout
            timeout = self.config.get("playwright_timeout", 30000)
            await page.goto(url, wait_until="networkidle", timeout=timeout)

            # Extract HTML
            result.html = await page.content()

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

            # Take screenshot if path provided
            if screenshot_path:
                # Ensure directory exists
                Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(
                    path=screenshot_path,
                    full_page=False,  # Just viewport for consistency
                    type="png",
                )
                result.screenshot_path = screenshot_path
                logger.info(f"Screenshot saved to {screenshot_path}")

            result.success = True
            logger.info(
                f"Successfully extracted content from {url} "
                f"(HTML: {len(result.html)} chars, Text: {len(result.text)} chars)"
            )

        except Exception as e:
            result.error = str(e)
            logger.error(f"Failed to extract content from {url}: {e}")

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

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.config.get("playwright_headless", True),
                args=["--disable-blink-features=AutomationControlled"],
            )

            viewport = self.config.get(
                "playwright_viewport", {"width": 1280, "height": 1024}
            )
            context = await browser.new_context(
                user_agent=self.config.user_agent,
                viewport=viewport,
                ignore_https_errors=False,
            )

            page = await context.new_page()

            # Configure security and performance
            await self._configure_page(page)

            # Extract everything
            result = await self._extract_from_page(page, url, screenshot_path)

            await context.close()
            await browser.close()

        return result

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
            if is_safe:
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
                args=["--disable-blink-features=AutomationControlled"],
            )

            viewport = self.config.get(
                "playwright_viewport", {"width": 1280, "height": 1024}
            )

            # Create parallel contexts
            contexts = []
            for i in range(min(self.max_parallel, len(validated_urls))):
                context = await browser.new_context(
                    user_agent=self.config.user_agent,
                    viewport=viewport,
                    ignore_https_errors=False,
                )
                contexts.append(context)
                logger.debug(f"Created context {i + 1}/{self.max_parallel}")

            # Process URLs in parallel
            tasks = []
            for i, url in enumerate(validated_urls):
                context = contexts[i % len(contexts)]
                page = await context.new_page()
                await self._configure_page(page)

                # Generate screenshot path
                domain = self._parse_domain_name(url)
                screenshot_path = f"{cache_dir}/images/{domain}.png"

                # Create task
                task = self._extract_from_page(page, url, screenshot_path)
                tasks.append(task)
                logger.debug(f"Created task for {url}")

            # Execute all tasks
            logger.info(f"Executing {len(tasks)} parallel fetch tasks")
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(batch_results):
                if isinstance(result, BaseException):
                    url = validated_urls[i]
                    logger.error(f"Task failed for {url}: {result}")
                    results.append(
                        FetchResult(url=url, success=False, error=str(result))
                    )
                else:
                    results.append(result)

            # Cleanup
            logger.debug("Cleaning up browser contexts")
            for context in contexts:
                await context.close()
            await browser.close()

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

    def __init__(self, target_date: str | datetime, max_parallel: int | None = None):
        """Initialize archive fetcher.

        Args:
            target_date: Target date as 'YYYYMMDD' string or datetime object
            max_parallel: Maximum concurrent snapshot fetches
        """
        super().__init__()
        if isinstance(target_date, datetime):
            self.target_date = target_date.strftime("%Y%m%d")
        else:
            self.target_date = target_date

        self.target = datetime.strptime(self.target_date, "%Y%m%d").replace(tzinfo=UTC)
        self.max_parallel = max_parallel or self.config.get("archive_max_parallel", 2)
        self.search_window = timedelta(days=self.config.get("archive_window_days", 365))
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

    def _fetch_html(self, url: str) -> tuple[CdxRecord | None, str]:
        """Retrieve the raw archived HTML for a URL.

        Args:
            url: URL or bare domain to fetch.

        Returns:
            tuple: ``(record, html)``; ``record`` is ``None`` when no usable
            capture exists.
        """
        record = self.find_closest_record(url)
        if record is None:
            return None, ""
        with self._client() as client:
            memento = client.get_memento(record, mode=Mode.original)
            try:
                return record, memento.text
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
        viewport = self.config.get(
            "playwright_viewport", {"width": 1280, "height": 1024}
        )
        timeout = self.config.get("playwright_timeout", 30000)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=self.config.get("playwright_headless", True)
                )
                try:
                    context = await browser.new_context(
                        user_agent=self.config.user_agent, viewport=viewport
                    )
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

        result = FetchResult(url=url, success=False)
        try:
            record, html = await asyncio.to_thread(self._fetch_html, url)
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
            result.error = f"no 200 capture near {self.target_date}"
            result.error_code = ErrorCode.NO_ARCHIVE_SNAPSHOT.value
            return result

        result.html = html
        result.text = TextProcessor.process_html_to_text(html)
        result.snapshot_timestamp = record.timestamp.strftime("%Y%m%d%H%M%S")

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
