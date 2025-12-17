#!/usr/bin/env python3
"""
Content processor for coordinating HTML and image extraction using Playwright.
Handles file I/O, caching, and coordination between different content types.
"""

import asyncio
import os

import numpy as np
from PIL import Image

from ..config import get_config
from ..fetchers import FetchResult, get_fetcher
from ..piedomains_logging import get_logger
from .text_processor import TextProcessor

logger = get_logger()


class ContentProcessor:
    """Coordinates content extraction for domains using Playwright fetcher."""

    def __init__(self, cache_dir: str | None = None, archive_date: str | None = None):
        """
        Initialize content processor.

        Args:
            cache_dir (str, optional): Directory for caching content
            archive_date (str, optional): Date for archive.org snapshots (YYYYMMDD format)
        """
        self.config = get_config()
        self.cache_dir = cache_dir or "cache"
        self.archive_date = archive_date
        self.fetcher = get_fetcher(
            archive_date, max_parallel=self.config.get("max_parallel", 4)
        )

        # Ensure cache directories exist
        self.html_dir = os.path.join(self.cache_dir, "html")
        self.image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def extract_all_content(
        self, domains: list[str], use_cache: bool = True, parallel: bool = True
    ) -> dict[str, dict]:
        """
        Extract all content (HTML, text, screenshots) from domains.
        Uses unified Playwright fetcher for everything.

        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached content
            parallel (bool): Whether to fetch in parallel

        Returns:
            Dict[str, dict]: Results keyed by domain name
        """
        results = {}

        # Check cache and determine what needs fetching
        to_fetch = []
        for domain in domains:
            domain_name = self._parse_domain_name(domain)

            if use_cache:
                # Check if we have both HTML and screenshot cached
                html_file = os.path.join(self.html_dir, f"{domain_name}.html")
                image_file = os.path.join(self.image_dir, f"{domain_name}.png")

                if os.path.exists(html_file) and os.path.exists(image_file):
                    try:
                        with open(html_file, encoding="utf-8") as f:
                            html_content = f.read()

                        # Process the cached HTML to get text
                        text_content = TextProcessor.process_html_to_text(html_content)

                        results[domain_name] = {
                            "success": True,
                            "html": html_content,
                            "text": text_content,
                            "screenshot_path": image_file,
                            "cached": True,
                        }
                        logger.info(f"Using cached content for {domain_name}")
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to read cache for {domain_name}: {e}")

            to_fetch.append(domain)

        # Fetch missing content
        if to_fetch:
            if parallel and len(to_fetch) > 1:
                # Use async batch fetching
                fetch_results = self._fetch_batch_async(to_fetch)
            else:
                # Sequential fetching
                fetch_results = []
                for url in to_fetch:
                    domain_name = self._parse_domain_name(url)
                    screenshot_path = os.path.join(self.image_dir, f"{domain_name}.png")
                    try:
                        result = self.fetcher.fetch_both(url, screenshot_path)
                        fetch_results.append(result)
                    except Exception as e:
                        # Create a failed FetchResult for network errors
                        from ..fetchers import FetchResult

                        failed_result = FetchResult(
                            url=url, success=False, error=f"Network error: {str(e)}"
                        )
                        fetch_results.append(failed_result)

            # Process fetch results
            for result in fetch_results:
                if isinstance(result, FetchResult):
                    domain_name = self._parse_domain_name(result.url)

                    if result.success:
                        # Cache the HTML
                        if use_cache and result.html:
                            html_file = os.path.join(
                                self.html_dir, f"{domain_name}.html"
                            )
                            try:
                                with open(html_file, "w", encoding="utf-8") as f:
                                    f.write(result.html)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to cache HTML for {domain_name}: {e}"
                                )

                        # Process HTML to get clean text if not already provided
                        if result.text:
                            # Use the text extracted by Playwright
                            clean_text = TextProcessor.clean_and_normalize_text(
                                result.text
                            )
                        else:
                            # Fallback to HTML processing
                            clean_text = TextProcessor.process_html_to_text(result.html)

                        results[domain_name] = {
                            "success": True,
                            "html": result.html,
                            "text": clean_text,
                            "screenshot_path": result.screenshot_path,
                            "title": result.title,
                            "meta_description": result.meta_description,
                            "cached": False,
                        }
                    else:
                        results[domain_name] = {
                            "success": False,
                            "error": result.error,
                            "cached": False,
                        }

        return results

    def extract_html_content(
        self,
        domains: list[str],
        use_cache: bool = True,
        *,
        force_fetch: bool = False,
        allow_content_types: list[str] = None,
        ignore_extensions: bool = False,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Extract HTML content for domains.
        Maintains backwards compatibility with existing API.

        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached HTML files
            force_fetch (bool): Skip security validation (dangerous)
            allow_content_types (list): Override allowed content types
            ignore_extensions (bool): Skip file extension validation

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: (html_content_dict, errors_dict)
        """
        all_content = self.extract_all_content(domains, use_cache=use_cache)

        html_content = {}
        errors = {}

        for domain_name, result in all_content.items():
            if result.get("success"):
                html_content[domain_name] = result.get("html", "")
            else:
                errors[domain_name] = result.get("error", "Unknown error")

        return html_content, errors

    def extract_text_content(
        self,
        domains: list[str],
        use_cache: bool = True,
        *,
        force_fetch: bool = False,
        allow_content_types: list[str] = None,
        ignore_extensions: bool = False,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Extract and process text content from domains.
        Maintains backwards compatibility with existing API.

        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached content
            force_fetch (bool): Skip security validation (dangerous)
            allow_content_types (list): Override allowed content types
            ignore_extensions (bool): Skip file extension validation

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: (processed_text_dict, errors_dict)
        """
        all_content = self.extract_all_content(domains, use_cache=use_cache)

        text_content = {}
        text_errors = {}

        for domain_name, result in all_content.items():
            if result.get("success"):
                text_content[domain_name] = result.get("text", "")
            else:
                text_errors[domain_name] = result.get("error", "Unknown error")

        return text_content, text_errors

    def extract_image_content(
        self,
        domains: list[str],
        use_cache: bool = True,
        *,
        force_fetch: bool = False,
        ignore_extensions: bool = False,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Extract screenshot images for domains.
        Maintains backwards compatibility with existing API.

        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached images
            force_fetch (bool): Skip security validation (dangerous)
            ignore_extensions (bool): Skip file extension validation

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: (image_paths_dict, errors_dict)
        """
        all_content = self.extract_all_content(domains, use_cache=use_cache)

        image_paths = {}
        errors = {}

        for domain_name, result in all_content.items():
            if result.get("success") and result.get("screenshot_path"):
                image_paths[domain_name] = result["screenshot_path"]
            elif not result.get("success"):
                errors[domain_name] = result.get("error", "Unknown error")

        return image_paths, errors

    def prepare_image_tensors(
        self, image_paths: dict[str, str]
    ) -> dict[str, np.ndarray]:
        """
        Convert images to numpy arrays for model input.

        Args:
            image_paths (Dict[str, str]): Domain name to image path mapping

        Returns:
            Dict[str, np.ndarray]: Domain name to image tensor mapping
        """
        tensors = {}

        for domain_name, image_path in image_paths.items():
            try:
                logger.info(f"Processing image tensor for {domain_name}: {image_path}")
                # Load and resize image
                img = Image.open(image_path)
                logger.info(f"Loaded image {img.size} for {domain_name}")
                img = img.convert("RGB")
                img = img.resize((254, 254))

                # Convert to numpy array and normalize
                img_array = np.array(img)
                img_array = img_array.astype("float32") / 255.0
                logger.info(f"Created tensor shape {img_array.shape} for {domain_name}")

                tensors[domain_name] = img_array

            except Exception as e:
                logger.error(f"Failed to process image for {domain_name}: {e}")

        return tensors

    def _fetch_batch_async(self, urls: list[str]) -> list[FetchResult]:
        """
        Helper to run async batch fetch in sync context.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of FetchResult objects
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Use the fetcher's batch method
            from ..fetchers import PlaywrightFetcher

            if isinstance(self.fetcher, PlaywrightFetcher):
                results = loop.run_until_complete(
                    self.fetcher.fetch_batch(urls, self.cache_dir)
                )
            else:
                # Fallback for ArchiveFetcher (no batch support yet)
                results = []
                for url in urls:
                    domain_name = self._parse_domain_name(url)
                    screenshot_path = os.path.join(self.image_dir, f"{domain_name}.png")
                    result = loop.run_until_complete(
                        self.fetcher.fetch_single(url, screenshot_path)
                    )
                    results.append(result)
            return results
        finally:
            loop.close()

    def _parse_domain_name(self, url_or_domain: str) -> str:
        """
        Extract clean domain name from URL or domain string.

        Args:
            url_or_domain (str): URL or domain name

        Returns:
            str: Clean domain name
        """
        # Import here to avoid circular imports
        from ..piedomain import Piedomain

        return Piedomain.parse_url_to_domain(url_or_domain)
