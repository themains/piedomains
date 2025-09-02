#!/usr/bin/env python3
"""
Modular page fetcher interfaces for different content sources.
Supports live content fetching and archive.org historical snapshots.
"""

import requests
import time
from datetime import datetime
from typing import Optional, Tuple, Union
from urllib.parse import urlparse, quote
from bs4 import BeautifulSoup
from selenium import webdriver

from .logging import get_logger
from .config import get_config

logger = get_logger()


class BaseFetcher:
    """Base class for content fetchers."""
    
    def fetch_html(self, url: str) -> Tuple[bool, str, str]:
        """
        Fetch HTML content from a URL.
        
        Args:
            url (str): URL to fetch
            
        Returns:
            tuple: (success, html_content, error_message)
        """
        raise NotImplementedError
        
    def fetch_screenshot(self, url: str, output_path: str) -> Tuple[bool, str]:
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
        
    def fetch_html(self, url: str) -> Tuple[bool, str, str]:
        """Fetch HTML from live web."""
        try:
            headers = {"User-Agent": self.config.user_agent}
            response = requests.get(
                url, 
                timeout=self.config.http_timeout,
                headers=headers,
                allow_redirects=True
            )
            response.raise_for_status()
            return True, response.text, ""
        except Exception as e:
            return False, "", str(e)
            
    def fetch_screenshot(self, url: str, output_path: str) -> Tuple[bool, str]:
        """Take screenshot using Selenium."""
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-extensions")
            options.add_argument("--no-sandbox")
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument(f"--window-size={self.config.webdriver_window_size}")
            options.add_argument(f"--user-agent={self.config.user_agent}")
            
            with webdriver.Chrome(
                service=webdriver.ChromeService(ChromeDriverManager().install()), 
                options=options
            ) as driver:
                driver.set_page_load_timeout(self.config.page_load_timeout)
                driver.get(url)
                time.sleep(self.config.screenshot_wait_time)
                driver.save_screenshot(output_path)
                return True, ""
        except Exception as e:
            return False, str(e)


class ArchiveFetcher(BaseFetcher):
    """Fetcher for archive.org historical snapshots."""
    
    def __init__(self, target_date: Union[str, datetime]):
        """
        Initialize archive fetcher.
        
        Args:
            target_date: Target date as 'YYYYMMDD' string or datetime object
        """
        self.config = get_config()
        if isinstance(target_date, datetime):
            self.target_date = target_date.strftime('%Y%m%d')
        else:
            self.target_date = target_date
            
    def _find_closest_snapshot(self, url: str) -> Optional[str]:
        """Find closest archived snapshot to target date."""
        try:
            # Use Wayback Machine availability API
            api_url = f"https://archive.org/wayback/available?url={quote(url)}&timestamp={self.target_date}"
            logger.info(f"Searching for archive snapshot of {url} near {self.target_date}")
            logger.info(f"Archive API URL: {api_url}")
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Archive API response: {data}")
            if data.get('archived_snapshots', {}).get('closest', {}).get('available'):
                closest_snapshot = data['archived_snapshots']['closest']
                snapshot_url = closest_snapshot['url']
                snapshot_date = closest_snapshot.get('timestamp', 'unknown')
                logger.info(f"Found closest snapshot for {url}: {snapshot_date} -> {snapshot_url}")
                return snapshot_url
            else:
                logger.warning(f"No archive snapshots available for {url}")
                return None
        except Exception as e:
            logger.error(f"Failed to find archive snapshot for {url}: {e}")
            return None
            
    def fetch_html(self, url: str) -> Tuple[bool, str, str]:
        """Fetch HTML from archive.org snapshot."""
        try:
            snapshot_url = self._find_closest_snapshot(url)
            if not snapshot_url:
                return False, "", f"No archive snapshot found for {url} near {self.target_date}"
                
            headers = {"User-Agent": self.config.user_agent}
            response = requests.get(
                snapshot_url, 
                timeout=self.config.http_timeout,
                headers=headers,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Clean up archive.org wrapper content
            html = response.text
            # Remove archive.org toolbar/navigation if present
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove archive.org specific elements
            for element in soup.find_all(['script', 'link', 'div'], 
                                       attrs={'src': lambda x: x and 'archive.org' in x}):
                element.decompose()
            for element in soup.find_all(attrs={'class': lambda x: x and 'wayback' in str(x).lower()}):
                element.decompose()
                
            return True, str(soup), ""
        except Exception as e:
            return False, "", str(e)
            
    def fetch_screenshot(self, url: str, output_path: str) -> Tuple[bool, str]:
        """Take screenshot of archived page."""
        try:
            snapshot_url = self._find_closest_snapshot(url)
            if not snapshot_url:
                return False, f"No archive snapshot found for {url} near {self.target_date}"
                
            from webdriver_manager.chrome import ChromeDriverManager
            
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-extensions")
            options.add_argument("--no-sandbox")
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument(f"--window-size={self.config.webdriver_window_size}")
            options.add_argument(f"--user-agent={self.config.user_agent}")
            
            with webdriver.Chrome(
                service=webdriver.ChromeService(ChromeDriverManager().install()), 
                options=options
            ) as driver:
                driver.set_page_load_timeout(self.config.page_load_timeout)
                driver.get(snapshot_url)
                time.sleep(self.config.screenshot_wait_time)
                driver.save_screenshot(output_path)
                return True, ""
        except Exception as e:
            return False, str(e)


def get_fetcher(archive_date: Optional[Union[str, datetime]] = None) -> BaseFetcher:
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