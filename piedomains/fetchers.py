#!/usr/bin/env python3
"""
Playwright-based page fetcher for content extraction.
Supports live content fetching and archive.org historical snapshots.
Unified pipeline for HTML, text extraction, and screenshots.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import aiohttp
import requests
from bs4 import BeautifulSoup
from playwright.async_api import Page, async_playwright

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
        allow_content_types: list[str] = None,
        ignore_extensions: bool = False,
    ) -> tuple[bool, str]:
        """
        Validate URL security before fetching content.

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
        """
        Extract clean domain name from URL or domain string.

        Args:
            url_or_domain (str): URL or domain name

        Returns:
            str: Clean domain name
        """
        # Import here to avoid circular imports
        from .piedomain import Piedomain

        return Piedomain.parse_url_to_domain(url_or_domain)


class PlaywrightFetcher(BaseFetcher):
    """Unified Playwright fetcher for all content extraction."""

    def __init__(self, max_parallel: int = 4):
        """
        Initialize Playwright fetcher.

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
                if isinstance(result, Exception):
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

    def fetch_both(self, url: str, output_path: str, **kwargs) -> FetchResult:
        """Sync wrapper for both HTML and screenshot."""
        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.fetch_single(url, output_path))
            return result
        finally:
            loop.close()


class ArchiveFetcher(BaseFetcher):
    """Fetcher for archive.org historical snapshots using Playwright."""

    def __init__(self, target_date: str | datetime, max_parallel: int = None):
        """
        Initialize archive fetcher.

        Args:
            target_date: Target date as 'YYYYMMDD' string or datetime object
            max_parallel: Maximum number of parallel browser contexts (default: 2 for archive.org)
        """
        super().__init__()
        if isinstance(target_date, datetime):
            self.target_date = target_date.strftime("%Y%m%d")
        else:
            self.target_date = target_date

        # Use archive-specific defaults
        self.max_parallel = max_parallel or self.config.get("archive_max_parallel", 2)
        self.cdx_rate_limit = self.config.get("archive_cdx_rate_limit", 1.0)
        self.page_delay = self.config.get("archive_page_delay", 0.5)
        self.retry_on_429 = self.config.get("archive_retry_on_429", True)
        self.wait_time_429 = self.config.get("archive_429_wait_time", 60)

    def _find_closest_snapshot(self, url: str) -> str | None:
        """Find the closest archived snapshot to the target date (sync version)."""
        try:
            api_url = (
                f"https://archive.org/wayback/available?"
                f"url={quote(url)}&timestamp={self.target_date}"
            )
            logger.info(
                f"Searching for archive snapshot of {url} near {self.target_date}"
            )

            response = requests.get(api_url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get("archived_snapshots", {}).get("closest", {}).get("available"):
                closest_snapshot = data["archived_snapshots"]["closest"]
                snapshot_url = closest_snapshot["url"]
                snapshot_date = closest_snapshot.get("timestamp", "unknown")
                logger.info(
                    f"Found closest snapshot for {url}: {snapshot_date} -> {snapshot_url}"
                )
                return snapshot_url
            else:
                logger.warning(f"No archive snapshots available for {url}")
                return None
        except Exception as e:
            logger.error(f"Failed to find archive snapshot for {url}: {e}")
            return None

    async def _async_find_closest_snapshot(self, url: str) -> str | None:
        """Find the closest archived snapshot to the target date (async version)."""
        try:
            api_url = (
                f"https://archive.org/wayback/available?"
                f"url={quote(url)}&timestamp={self.target_date}"
            )
            logger.debug(
                f"Searching for archive snapshot of {url} near {self.target_date}"
            )

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(api_url) as response:
                    if response.status == 429 and self.retry_on_429:
                        logger.warning(
                            f"Rate limited by archive.org for {url}, waiting {self.wait_time_429}s"
                        )
                        await asyncio.sleep(self.wait_time_429)
                        # Retry once
                        async with session.get(api_url) as retry_response:
                            response = retry_response

                    response.raise_for_status()
                    data = await response.json()

            if data.get("archived_snapshots", {}).get("closest", {}).get("available"):
                closest_snapshot = data["archived_snapshots"]["closest"]
                snapshot_url = closest_snapshot["url"]
                snapshot_date = closest_snapshot.get("timestamp", "unknown")
                logger.debug(
                    f"Found closest snapshot for {url}: {snapshot_date} -> {snapshot_url}"
                )
                return snapshot_url
            else:
                logger.debug(f"No archive snapshots available for {url}")
                return None
        except Exception as e:
            logger.error(f"Failed to find archive snapshot for {url}: {e}")
            return None

    async def _rate_limited_snapshot_lookup(
        self, url: str, semaphore: asyncio.Semaphore
    ) -> tuple[str, str | None]:
        """Find snapshot with rate limiting."""
        async with semaphore:
            await asyncio.sleep(self.cdx_rate_limit)
            snapshot_url = await self._async_find_closest_snapshot(url)
            return url, snapshot_url

    async def _extract_from_archived_page(
        self, page: Page, url: str, snapshot_url: str, screenshot_path: str | None
    ) -> FetchResult:
        """Extract content from an archived page."""
        result = FetchResult(url=url, success=False)

        try:
            # Navigate to archived page
            timeout = self.config.get("playwright_timeout", 30000)
            await page.goto(snapshot_url, wait_until="networkidle", timeout=timeout)

            # Extract HTML and clean archive.org wrapper
            full_html = await page.content()
            soup = BeautifulSoup(full_html, "html.parser")

            # Remove archive.org specific elements
            for element in soup.find_all(
                ["script", "link", "div"],
                attrs={
                    "src": lambda x: x and "archive.org" in x,
                    "href": lambda x: x and "archive.org" in x,
                },
            ):
                element.decompose()

            # Remove wayback machine toolbar
            for element in soup.find_all(id=lambda x: x and "wm-" in str(x)):
                element.decompose()

            for element in soup.find_all(
                attrs={"class": lambda x: x and "wayback" in str(x).lower()}
            ):
                element.decompose()

            result.html = str(soup)

            # Extract text content
            result.text = await page.evaluate(
                """() => {
                // Remove archive.org elements first
                const archiveElements = document.querySelectorAll(
                    '[id*="wm-"], [class*="wayback"], [src*="archive.org"]'
                );
                archiveElements.forEach(el => el.remove());

                // Then remove standard non-content elements
                const elementsToRemove = document.querySelectorAll(
                    'script, style, noscript, iframe, object, embed, applet'
                );
                elementsToRemove.forEach(el => el.remove());

                const body = document.body;
                if (!body) return '';

                let textContent = body.innerText || body.textContent || '';
                textContent = textContent.replace(/\n\n\n+/g, '\n\n');
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

            # Take screenshot if needed
            if screenshot_path:
                Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=screenshot_path, full_page=False, type="png")
                result.screenshot_path = screenshot_path

            result.success = True
            logger.info(f"Successfully extracted archived content from {url}")

        except Exception as e:
            result.error = str(e)
            logger.error(f"Failed to extract archived content from {url}: {e}")

        return result

    async def fetch_single(
        self, url: str, screenshot_path: str | None = None
    ) -> FetchResult:
        """Fetch content from archive.org snapshot."""
        # Security validation
        is_safe, msg = self._validate_url_security(url)
        if not is_safe:
            return FetchResult(url=url, success=False, error=msg)

        # Find snapshot
        snapshot_url = self._find_closest_snapshot(url)
        if not snapshot_url:
            return FetchResult(
                url=url,
                success=False,
                error=f"No archive snapshot found near {self.target_date}",
            )

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.config.get("playwright_headless", True)
            )

            viewport = self.config.get(
                "playwright_viewport", {"width": 1280, "height": 1024}
            )
            context = await browser.new_context(
                user_agent=self.config.user_agent, viewport=viewport
            )

            page = await context.new_page()

            # No need to configure blocking for archive.org
            result = await self._extract_from_archived_page(
                page, url, snapshot_url, screenshot_path
            )

            await context.close()
            await browser.close()

        return result

    async def fetch_batch(
        self, urls: list[str], cache_dir: str = "cache"
    ) -> list[FetchResult]:
        """Fetch multiple URLs from archive.org in parallel with rate limiting."""
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
            f"Starting batch archive fetch for {len(validated_urls)} URLs "
            f"with {self.max_parallel} parallel workers, rate limit: {self.cdx_rate_limit}s"
        )

        # Phase 1: Find snapshots in parallel with rate limiting
        cdx_semaphore = asyncio.Semaphore(1)  # Sequential CDX lookups for rate limiting
        snapshot_tasks = []

        for url in validated_urls:
            task = self._rate_limited_snapshot_lookup(url, cdx_semaphore)
            snapshot_tasks.append(task)

        logger.info(f"Looking up {len(snapshot_tasks)} snapshots from archive.org")
        snapshot_results = await asyncio.gather(*snapshot_tasks, return_exceptions=True)

        # Collect valid snapshots
        valid_snapshots = []
        for i, result in enumerate(snapshot_results):
            if isinstance(result, Exception):
                url = validated_urls[i]
                logger.error(f"Snapshot lookup failed for {url}: {result}")
                results.append(
                    FetchResult(
                        url=url,
                        success=False,
                        error=f"Snapshot lookup failed: {result}",
                    )
                )
            else:
                url, snapshot_url = result
                if snapshot_url:
                    valid_snapshots.append((url, snapshot_url))
                else:
                    results.append(
                        FetchResult(
                            url=url,
                            success=False,
                            error=f"No archive snapshot found near {self.target_date}",
                        )
                    )

        if not valid_snapshots:
            logger.warning("No valid snapshots found for any URLs")
            return results

        # Phase 2: Fetch pages in parallel with controlled concurrency
        logger.info(f"Fetching {len(valid_snapshots)} pages from archive.org")

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.config.get("playwright_headless", True)
            )

            viewport = self.config.get(
                "playwright_viewport", {"width": 1280, "height": 1024}
            )

            # Create parallel contexts (limited by max_parallel)
            contexts = []
            for i in range(min(self.max_parallel, len(valid_snapshots))):
                context = await browser.new_context(
                    user_agent=self.config.user_agent, viewport=viewport
                )
                contexts.append(context)
                logger.debug(f"Created archive context {i + 1}/{self.max_parallel}")

            # Create fetch tasks with round-robin context assignment
            fetch_tasks = []
            for i, (url, snapshot_url) in enumerate(valid_snapshots):
                context = contexts[i % len(contexts)]
                page = await context.new_page()

                # Generate screenshot path
                domain_name = (
                    url.replace("https://", "").replace("http://", "").split("/")[0]
                )
                screenshot_path = f"{cache_dir}/images/{domain_name}.png"
                Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)

                # Add delay between page creations to respect rate limits
                if i > 0:
                    await asyncio.sleep(self.page_delay)

                task = self._extract_from_archived_page(
                    page, url, snapshot_url, screenshot_path
                )
                fetch_tasks.append(task)

            logger.info(f"Executing {len(fetch_tasks)} parallel archive fetch tasks")
            page_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Process fetch results
            for i, result in enumerate(page_results):
                if isinstance(result, Exception):
                    url, _ = valid_snapshots[i]
                    logger.error(f"Archive page fetch failed for {url}: {result}")
                    results.append(
                        FetchResult(
                            url=url, success=False, error=f"Page fetch failed: {result}"
                        )
                    )
                else:
                    results.append(result)

            # Clean up contexts
            for context in contexts:
                await context.close()
            await browser.close()

        logger.info(
            f"Archive batch fetch complete. Success: {sum(r.success for r in results)}/{len(results)}"
        )
        return results

    # Sync wrappers
    def fetch_html(self, url: str, **kwargs) -> tuple[bool, str, str]:
        """Sync wrapper for HTML fetching from archive."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.fetch_single(url))
            return result.success, result.html, result.error
        finally:
            loop.close()

    def fetch_screenshot(
        self, url: str, output_path: str, **kwargs
    ) -> tuple[bool, str]:
        """Sync wrapper for screenshot from archive."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.fetch_single(url, output_path))
            return result.success, result.error
        finally:
            loop.close()


def get_fetcher(
    archive_date: str | datetime | None = None, max_parallel: int = 4
) -> BaseFetcher:
    """
    Factory function to get appropriate fetcher.

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
