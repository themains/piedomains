#!/usr/bin/env python3
"""
Archive.org content retrieval and historical data access utilities.

This module provides functionality for accessing historical snapshots of web content
through the Internet Archive's Wayback Machine. It includes utilities for querying
available snapshots within date ranges and downloading content from archived pages.

The module supports the piedomains package's historical domain analysis capabilities
by providing structured access to archived web content for training and classification
on historical data.

Example:
    Basic usage for getting archived content:
        >>> from piedomains.archive_org_downloader import get_urls_year, download_from_archive_org
        >>> urls = get_urls_year("example.com", year=2020)
        >>> if urls:
        ...     content = download_from_archive_org(urls[0])
        ...     print(f"Retrieved {len(content)} characters of historical content")
"""

import json
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

from .piedomains_logging import get_logger

logger = get_logger()


def _retry_with_backoff(
    func: Callable[..., Any],
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs,
) -> Any:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        **kwargs: Keyword arguments for the function

    Returns:
        Result of successful function execution

    Raises:
        Exception: Last exception encountered if all retries fail
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries:
                logger.debug(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
                raise e
        except Exception as e:
            # Don't retry non-network exceptions
            logger.error(f"Non-retriable error: {e}")
            raise e

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception


def get_urls_year(
    domain: str, year: int = 2014, status_code: int = 200, limit: int | None = None
) -> list[str]:
    """
    Retrieve all archived URLs for a domain within a specific year.

    This function queries the Internet Archive's CDX API to find all available
    snapshots of a domain within the specified year that returned the given
    HTTP status code.

    Args:
        domain (str): Domain name to search for (e.g., "example.com" or "https://example.com").
                     The function will handle both domain names and full URLs.
        year (int): Year to search within (e.g., 2020). Defaults to 2014.
                   Must be between 1996 (first archive) and current year.
        status_code (int): HTTP status code to filter by. Only snapshots that
                          returned this status code will be included. Defaults to 200.
        limit (Optional[int]): Maximum number of URLs to return. If None, returns all
                              available URLs. Useful for large domains with many snapshots.

    Returns:
        List[str]: List of complete Wayback Machine URLs for accessing archived content.
                  Each URL is formatted as 'https://web.archive.org/web/{timestamp}/{original_url}'.
                  Returns empty list if no snapshots are found or if an error occurs.

    Raises:
        ValueError: If year is outside valid range or domain is invalid.
        requests.RequestException: If API request fails (logged but not raised).

    Example:
        >>> # Get all snapshots for 2020
        >>> urls = get_urls_year("cnn.com", year=2020)
        >>> print(f"Found {len(urls)} snapshots")

        >>> # Get limited snapshots with error handling
        >>> urls = get_urls_year("example.com", year=2019, limit=10)
        >>> if urls:
        ...     print(f"First snapshot: {urls[0]}")
        ... else:
        ...     print("No snapshots found")

    Note:
        - The function searches for snapshots with successful HTTP responses (200 by default)
        - Results are ordered chronologically by snapshot timestamp
        - Large popular domains may have thousands of snapshots per year
        - Use the limit parameter to avoid excessive API calls
    """
    # Validate inputs
    current_year = datetime.now().year
    if year < 1996 or year > current_year:
        raise ValueError(f"Year must be between 1996 and {current_year}, got {year}")

    if not domain or not isinstance(domain, str):
        raise ValueError("Domain must be a non-empty string")

    # Clean up domain - remove protocol if present
    parsed = urlparse(domain if "://" in domain else f"http://{domain}")
    clean_domain = parsed.netloc or parsed.path

    if not clean_domain:
        raise ValueError(f"Invalid domain format: {domain}")

    # Construct date range for the year
    from_date = f"{year}0101"
    to_date = f"{year}1231"

    # Build API URL
    api_url = (
        f"https://web.archive.org/cdx/search"
        f"?url={quote(clean_domain)}"
        f"&matchType=prefix"
        f"&filter=statuscode:{status_code}"
        f"&from={from_date}"
        f"&to={to_date}"
        f"&output=json"
    )

    if limit:
        api_url += f"&limit={limit}"

    logger.info(f"Querying archive.org for {clean_domain} snapshots in {year}")
    logger.debug(f"API URL: {api_url}")

    try:
        # Send request to Wayback Machine CDX API with retry logic
        logger.debug(f"Attempting to query CDX API: {api_url}")

        def _make_request():
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            return response

        response = _retry_with_backoff(_make_request, max_retries=3, initial_delay=1.0)

        # Parse JSON response
        data = json.loads(response.content.decode("utf-8"))

        if not data:
            logger.warning(f"No snapshots found for {clean_domain} in {year}")
            return []

        # Skip header row (first element) if present
        rows = (
            data[1:]
            if data and len(data[0]) > 0 and isinstance(data[0][0], str)
            else data
        )

        # Convert to Wayback Machine URLs
        urls = []
        for row in rows:
            if len(row) >= 3:  # Ensure row has required fields
                timestamp = row[1]
                original_url = row[2]
                wayback_url = f"https://web.archive.org/web/{timestamp}/{original_url}"
                urls.append(wayback_url)

        logger.info(
            f"Found {len(urls)} archived snapshots for {clean_domain} in {year}"
        )
        return urls

    except requests.RequestException as e:
        logger.error(f"Failed to query archive.org API for {clean_domain}: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse archive.org API response: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error querying archive.org for {clean_domain}: {e}")
        return []


def download_from_archive_org(
    url: str, timeout: int = 30, clean_content: bool = True
) -> str:
    """
    Download and extract text content from an archived webpage.

    This function retrieves content from a Wayback Machine URL and extracts
    the visible text content, optionally cleaning archive-specific elements.

    Args:
        url (str): Complete Wayback Machine URL (from get_urls_year or similar).
                  Should be in format 'https://web.archive.org/web/{timestamp}/{original_url}'.
        timeout (int): HTTP request timeout in seconds. Defaults to 30 seconds
                      as archived pages can be slow to load.
        clean_content (bool): If True, removes archive.org specific navigation
                             and metadata elements. Defaults to True for cleaner content.

    Returns:
        str: Extracted text content from the archived page. Returns empty string
             if download fails or no content is found.

    Raises:
        ValueError: If URL is not a valid Wayback Machine URL.
        requests.RequestException: If HTTP request fails (logged but not raised).

    Example:
        >>> # Download content from archived page
        >>> wayback_url = "https://web.archive.org/web/20200101120000/https://example.com"
        >>> content = download_from_archive_org(wayback_url)
        >>> print(f"Retrieved {len(content)} characters")

        >>> # Download with custom timeout and no cleaning
        >>> raw_content = download_from_archive_org(
        ...     wayback_url,
        ...     timeout=60,
        ...     clean_content=False
        ... )

    Note:
        - Only works with archive.org URLs, not live web pages
        - Extracted text includes all visible page content
        - Archive pages may load slowly due to Internet Archive infrastructure
        - Some archived pages may be incomplete or corrupted
    """
    # Validate URL format
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    if "web.archive.org/web/" not in url:
        raise ValueError(f"URL must be a Wayback Machine URL: {url}")

    logger.info(f"Downloading content from archive.org: {url}")

    try:
        # Send GET request to archived page with retry logic
        headers = {
            "User-Agent": "Mozilla/5.0 (piedomains research tool) AppleWebKit/537.36"
        }

        def _download_page():
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response

        logger.debug(f"Attempting to download archived page: {url}")
        response = _retry_with_backoff(_download_page, max_retries=3, initial_delay=2.0)

        # Parse HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        if clean_content:
            # Remove archive.org specific elements
            # Remove wayback machine toolbar and navigation
            for element in soup.find_all(
                ["div", "table"], class_=lambda x: x and "wayback" in str(x).lower()
            ):
                element.decompose()

            # Remove archive-specific scripts and styles
            for element in soup.find_all(["script", "style"]):
                if element.get("src") and "archive.org" in element.get("src"):
                    element.decompose()
                elif "archive.org" in element.get_text():
                    element.decompose()

        # Extract all visible text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up excessive whitespace
        import re

        text = re.sub(r"\s+", " ", text).strip()

        logger.info(f"Successfully extracted {len(text)} characters from archived page")
        return text

    except requests.RequestException as e:
        logger.error(f"Failed to download from {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return ""


def get_closest_snapshot(
    domain: str, target_date: str | datetime, status_code: int = 200
) -> str | None:
    """
    Find the archived snapshot closest to a specific target date.

    This function uses the Wayback Machine availability API to find the snapshot
    that was captured closest in time to the specified target date.

    Args:
        domain (str): Domain name to search for (e.g., "example.com").
        target_date (Union[str, datetime]): Target date as 'YYYYMMDD' string
                                          or datetime object.
        status_code (int): HTTP status code to filter by. Defaults to 200.

    Returns:
        Optional[str]: Wayback Machine URL of the closest snapshot if found,
                      None if no snapshots are available near the target date.

    Raises:
        ValueError: If target_date format is invalid or domain is invalid.

    Example:
        >>> from datetime import datetime
        >>>
        >>> # Using string date
        >>> url = get_closest_snapshot("cnn.com", "20200315")
        >>> if url:
        ...     content = download_from_archive_org(url)

        >>> # Using datetime object
        >>> target = datetime(2019, 6, 15)
        >>> url = get_closest_snapshot("example.com", target)

    Note:
        - Returns the snapshot with timestamp closest to target_date
        - Preference is given to snapshots after the target date if available
        - Uses the Wayback Machine availability API for efficient lookup
    """
    # Validate and convert target_date
    if isinstance(target_date, datetime):
        date_str = target_date.strftime("%Y%m%d")
    elif isinstance(target_date, str):
        # Validate string format
        if not (len(target_date) == 8 and target_date.isdigit()):
            raise ValueError(
                f"Date string must be in YYYYMMDD format, got: {target_date}"
            )
        date_str = target_date
    else:
        raise ValueError(
            f"target_date must be string or datetime, got {type(target_date)}"
        )

    # Clean domain
    parsed = urlparse(domain if "://" in domain else f"http://{domain}")
    clean_domain = parsed.netloc or parsed.path

    if not clean_domain:
        raise ValueError(f"Invalid domain format: {domain}")

    # Use Wayback Machine availability API
    api_url = f"https://archive.org/wayback/available?url={quote(clean_domain)}&timestamp={date_str}"

    logger.info(f"Finding closest snapshot for {clean_domain} near {date_str}")
    logger.debug(f"Availability API URL: {api_url}")

    try:

        def _query_availability():
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            return response

        logger.debug(f"Attempting to query availability API: {api_url}")
        response = _retry_with_backoff(
            _query_availability, max_retries=2, initial_delay=1.0
        )

        data = response.json()

        if data.get("archived_snapshots", {}).get("closest", {}).get("available"):
            closest = data["archived_snapshots"]["closest"]
            snapshot_url = closest["url"]
            snapshot_timestamp = closest.get("timestamp", "unknown")

            logger.info(
                f"Found closest snapshot for {clean_domain}: {snapshot_timestamp}"
            )
            return snapshot_url
        else:
            logger.warning(f"No snapshots available for {clean_domain} near {date_str}")
            return None

    except requests.RequestException as e:
        logger.error(f"Failed to query availability API for {clean_domain}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse availability API response: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error finding closest snapshot: {e}")
        return None
