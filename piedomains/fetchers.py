#!/usr/bin/env python3
"""
Modular page fetcher interfaces for different content sources.
Supports live content fetching and archive.org historical snapshots.
"""

import time
from datetime import datetime
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

from .config import get_config
from .logging import get_logger

logger = get_logger()


class BaseFetcher:
    """Base class for content fetchers."""

    def fetch_html(self, url: str) -> tuple[bool, str, str]:
        """
        Fetch HTML content from a URL.

        Args:
            url (str): URL to fetch

        Returns:
            tuple: (success, html_content, error_message)
        """
        raise NotImplementedError

    def fetch_screenshot(self, url: str, output_path: str) -> tuple[bool, str]:
        """
        Take screenshot of a webpage.

        Args:
            url (str): URL to screenshot
            output_path (str): Path to save screenshot

        Returns:
            tuple: (success, error_message)
        """
        raise NotImplementedError


class LiveFetcher(BaseFetcher):
    """Fetcher for live web content (current implementation)."""

    def __init__(self):
        self.config = get_config()

    def fetch_html(self, url: str) -> tuple[bool, str, str]:
        """Fetch HTML from live web."""
        try:
            headers = {"User-Agent": self.config.user_agent}
            response = requests.get(
                url,
                timeout=self.config.http_timeout,
                headers=headers,
                allow_redirects=True,
            )
            response.raise_for_status()
            return True, response.text, ""
        except Exception as e:
            return False, "", str(e)

    def fetch_screenshot(self, url: str, output_path: str) -> tuple[bool, str]:
        """
        Take screenshot using Selenium with robust error handling.

        Args:
            url (str): URL to capture screenshot from
            output_path (str): Path where screenshot should be saved

        Returns:
            tuple[bool, str]: Success status and error message if failed
        """
        driver = None
        try:
            from selenium.common.exceptions import (
                SessionNotCreatedException,
                TimeoutException,
                WebDriverException,
            )
            from webdriver_manager.chrome import ChromeDriverManager

            logger.info(f"Taking screenshot of {url}")
            logger.debug(f"Screenshot output path: {output_path}")

            # Configure Chrome options for headless operation
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-extensions")
            options.add_argument("--no-sandbox")
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-dev-shm-usage")  # Prevent memory issues
            options.add_argument("--disable-background-timer-throttling")
            options.add_argument("--disable-backgrounding-occluded-windows")
            options.add_argument("--disable-renderer-backgrounding")
            options.add_argument(f"--window-size={self.config.webdriver_window_size}")
            options.add_argument(f"--user-agent={self.config.user_agent}")

            # Create WebDriver with proper resource management
            driver = webdriver.Chrome(
                service=webdriver.ChromeService(ChromeDriverManager().install()),
                options=options,
            )

            try:
                # Configure timeouts
                driver.set_page_load_timeout(self.config.page_load_timeout)
                driver.implicitly_wait(5)  # Implicit wait for elements

                # Navigate to page and take screenshot
                logger.debug(f"Navigating to {url}")
                driver.get(url)

                # Wait for page to settle
                time.sleep(self.config.screenshot_wait_time)

                # Take screenshot
                logger.debug(f"Saving screenshot to {output_path}")
                driver.save_screenshot(output_path)

                logger.info(f"Successfully captured screenshot for {url}")
                return True, ""

            except TimeoutException as e:
                logger.error(f"Timeout while loading {url}: {e}")
                return False, f"Page load timeout: {str(e)}"

            except WebDriverException as e:
                logger.error(f"WebDriver error for {url}: {e}")
                return False, f"WebDriver error: {str(e)}"

            finally:
                # Ensure driver is always cleaned up
                if driver:
                    try:
                        driver.quit()
                        logger.debug("WebDriver cleaned up successfully")
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Error during WebDriver cleanup: {cleanup_error}"
                        )

        except SessionNotCreatedException as e:
            logger.error(f"Failed to create WebDriver session: {e}")
            return False, f"WebDriver session creation failed: {str(e)}"

        except Exception as e:
            logger.error(f"Unexpected error taking screenshot of {url}: {e}")
            return False, f"Unexpected error: {str(e)}"


class ArchiveFetcher(BaseFetcher):
    """Fetcher for archive.org historical snapshots."""

    def __init__(self, target_date: str | datetime):
        """
        Initialize archive fetcher.

        Args:
            target_date: Target date as 'YYYYMMDD' string or datetime object
        """
        self.config = get_config()
        if isinstance(target_date, datetime):
            self.target_date = target_date.strftime("%Y%m%d")
        else:
            self.target_date = target_date

    def _find_closest_snapshot(self, url: str) -> str | None:
        """
        Find the closest archived snapshot to the target date using Wayback Machine API.

        This method queries the Internet Archive's availability API to find the snapshot
        that was captured closest in time to the target date specified during
        ArchiveFetcher initialization.

        Args:
            url (str): The original URL to find archived snapshots for.
                      Should be a valid HTTP/HTTPS URL.

        Returns:
            str | None: Complete Wayback Machine URL for the closest snapshot
                       if found, None if no snapshots are available.

        Example:
            >>> fetcher = ArchiveFetcher("20200101")
            >>> snapshot = fetcher._find_closest_snapshot("https://example.com")
            >>> if snapshot:
            ...     print(f"Found snapshot: {snapshot}")

        Note:
            - Uses the Wayback Machine availability API for efficient lookup
            - Preference is given to snapshots after the target date if available
            - May return None if the domain was never archived or not available
        """
        try:
            # Use Wayback Machine availability API
            api_url = f"https://archive.org/wayback/available?url={quote(url)}&timestamp={self.target_date}"
            logger.info(
                f"Searching for archive snapshot of {url} near {self.target_date}"
            )
            logger.info(f"Archive API URL: {api_url}")
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Archive API response: {data}")
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

    def fetch_html(self, url: str) -> tuple[bool, str, str]:
        """Fetch HTML from archive.org snapshot."""
        try:
            snapshot_url = self._find_closest_snapshot(url)
            if not snapshot_url:
                return (
                    False,
                    "",
                    f"No archive snapshot found for {url} near {self.target_date}",
                )

            headers = {"User-Agent": self.config.user_agent}
            response = requests.get(
                snapshot_url,
                timeout=self.config.http_timeout,
                headers=headers,
                allow_redirects=True,
            )
            response.raise_for_status()

            # Clean up archive.org wrapper content
            html = response.text
            # Remove archive.org toolbar/navigation if present
            soup = BeautifulSoup(html, "html.parser")

            # Remove archive.org specific elements
            for element in soup.find_all(
                ["script", "link", "div"],
                attrs={"src": lambda x: x and (urlparse(x).hostname == "archive.org")},
            ):
                element.decompose()
            for element in soup.find_all(
                attrs={"class": lambda x: x and "wayback" in str(x).lower()}
            ):
                element.decompose()

            return True, str(soup), ""
        except Exception as e:
            return False, "", str(e)

    def fetch_screenshot(self, url: str, output_path: str) -> tuple[bool, str]:
        """
        Take screenshot of archived page with robust error handling.

        Args:
            url (str): Original URL to find archived snapshot for
            output_path (str): Path where screenshot should be saved

        Returns:
            tuple[bool, str]: Success status and error message if failed
        """
        driver = None
        try:
            snapshot_url = self._find_closest_snapshot(url)
            if not snapshot_url:
                error_msg = (
                    f"No archive snapshot found for {url} near {self.target_date}"
                )
                logger.warning(error_msg)
                return False, error_msg

            from selenium.common.exceptions import (
                SessionNotCreatedException,
                TimeoutException,
                WebDriverException,
            )
            from webdriver_manager.chrome import ChromeDriverManager

            logger.info(f"Taking screenshot of archived page: {snapshot_url}")
            logger.debug(f"Screenshot output path: {output_path}")

            # Configure Chrome options for headless operation
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-extensions")
            options.add_argument("--no-sandbox")
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-dev-shm-usage")  # Prevent memory issues
            options.add_argument("--disable-background-timer-throttling")
            options.add_argument("--disable-backgrounding-occluded-windows")
            options.add_argument("--disable-renderer-backgrounding")
            options.add_argument(f"--window-size={self.config.webdriver_window_size}")
            options.add_argument(f"--user-agent={self.config.user_agent}")

            # Create WebDriver with proper resource management
            driver = webdriver.Chrome(
                service=webdriver.ChromeService(ChromeDriverManager().install()),
                options=options,
            )

            try:
                # Configure timeouts
                driver.set_page_load_timeout(self.config.page_load_timeout)
                driver.implicitly_wait(5)  # Implicit wait for elements

                # Navigate to archived page and take screenshot
                logger.debug(f"Navigating to archived page: {snapshot_url}")
                driver.get(snapshot_url)

                # Wait for page to settle (archived pages may be slower)
                time.sleep(self.config.screenshot_wait_time)

                # Take screenshot
                logger.debug(f"Saving screenshot to {output_path}")
                driver.save_screenshot(output_path)

                logger.info(f"Successfully captured screenshot for archived {url}")
                return True, ""

            except TimeoutException as e:
                logger.error(f"Timeout while loading archived page {snapshot_url}: {e}")
                return False, f"Archive page load timeout: {str(e)}"

            except WebDriverException as e:
                logger.error(f"WebDriver error for archived page {snapshot_url}: {e}")
                return False, f"WebDriver error: {str(e)}"

            finally:
                # Ensure driver is always cleaned up
                if driver:
                    try:
                        driver.quit()
                        logger.debug("WebDriver cleaned up successfully")
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Error during WebDriver cleanup: {cleanup_error}"
                        )

        except SessionNotCreatedException as e:
            logger.error(f"Failed to create WebDriver session for archive: {e}")
            return False, f"WebDriver session creation failed: {str(e)}"

        except Exception as e:
            logger.error(f"Unexpected error taking screenshot of archived {url}: {e}")
            return False, f"Unexpected error: {str(e)}"


def get_fetcher(archive_date: str | datetime | None = None) -> BaseFetcher:
    """
    Factory function to get appropriate fetcher.

    Args:
        archive_date: If provided, returns ArchiveFetcher for this date.
                     If None, returns LiveFetcher for current content.

    Returns:
        BaseFetcher: Appropriate fetcher instance
    """
    if archive_date:
        return ArchiveFetcher(archive_date)
    else:
        return LiveFetcher()
