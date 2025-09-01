#!/usr/bin/env python3
"""
Archive.org support for piedomains.
Extended API functions that support historical snapshots.
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Union, Optional
from urllib.parse import quote

from .piedomain import Piedomain
from .logging import get_logger

logger = get_logger()


def _fetch_archive_snapshot_url(url: str, target_date: str) -> Optional[str]:
    """Find closest archive.org snapshot to target date."""
    try:
        api_url = f"https://archive.org/wayback/available?url={quote(url)}&timestamp={target_date}"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get('archived_snapshots', {}).get('closest', {}).get('available'):
            return data['archived_snapshots']['closest']['url']
        return None
    except Exception as e:
        logger.error(f"Failed to find archive snapshot for {url}: {e}")
        return None


def pred_shalla_cat_archive(
    input: list, 
    archive_date: Union[str, datetime],
    html_path: str = None, 
    image_path: str = None,
    use_cache: bool = True, 
    latest: bool = False
) -> pd.DataFrame:
    """
    Predict domain categories using archived content from a specific date.
    
    Args:
        input (list): List of URLs or domain names to classify
        archive_date: Target date as 'YYYYMMDD' string or datetime object
        html_path (str): Path to directory with HTML files (optional)
        image_path (str): Path to directory with screenshot images (optional)
        use_cache (bool): Whether to reuse existing files
        latest (bool): Whether to download latest model version
        
    Returns:
        pd.DataFrame: Same format as pred_shalla_cat but using archived content
    """
    if isinstance(archive_date, datetime):
        archive_date = archive_date.strftime('%Y%m%d')
        
    logger.info(f"Starting archive-based prediction for {len(input)} URLs/domains from date {archive_date}")
    
    # Temporarily modify HTML extraction to use archive URLs
    original_method = Piedomain.extract_htmls
    
    def archive_extract_htmls(cls, urls_or_domains, use_cache, html_path):
        """Modified HTML extraction that uses archive.org snapshots."""
        errors = {}
        import os
        os.makedirs(html_path, exist_ok=True)
        
        for url_or_domain in urls_or_domains:
            domain = cls.parse_url_to_domain(url_or_domain)
            html_file = f"{html_path}/{domain}.html"
            
            if use_cache and os.path.exists(html_file):
                continue
                
            # Construct full URL if needed
            if not url_or_domain.startswith(('http://', 'https://')):
                full_url = f"https://{url_or_domain}"
            else:
                full_url = url_or_domain
                
            # Get archive snapshot URL
            snapshot_url = _fetch_archive_snapshot_url(full_url, archive_date)
            if not snapshot_url:
                errors[domain] = f"No archive snapshot found for {full_url} near {archive_date}"
                continue
                
            try:
                response = requests.get(snapshot_url, timeout=30)
                response.raise_for_status()
                
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                    
            except Exception as e:
                errors[domain] = str(e)
                
        return errors
    
    # Temporarily replace the method
    Piedomain.extract_htmls = classmethod(archive_extract_htmls)
    
    try:
        # Use existing text prediction with modified HTML extraction
        result = Piedomain.pred_shalla_cat_with_text(
            input=input,
            html_path=html_path or f"archive_html_{archive_date}",
            use_cache=use_cache,
            latest=latest
        )
        result['archive_date'] = archive_date
        return result
        
    finally:
        # Restore original method
        Piedomain.extract_htmls = original_method


def pred_shalla_cat_with_text_archive(
    input: list,
    archive_date: Union[str, datetime], 
    html_path: str = None,
    use_cache: bool = True,
    latest: bool = True
) -> pd.DataFrame:
    """
    Predict domain categories using archived text content.
    
    Args:
        input (list): List of URLs or domain names to classify
        archive_date: Target date as 'YYYYMMDD' string or datetime object
        html_path (str): Path to directory with HTML files (optional)
        use_cache (bool): Whether to reuse existing HTML files
        latest (bool): Whether to download latest model version
        
    Returns:
        pd.DataFrame: Text predictions using archived content
    """
    return pred_shalla_cat_archive(
        input=input,
        archive_date=archive_date,
        html_path=html_path,
        use_cache=use_cache,
        latest=latest
    )


def pred_shalla_cat_with_images_archive(
    input: list,
    archive_date: Union[str, datetime],
    image_path: str = None,
    use_cache: bool = True,
    latest: bool = True
) -> pd.DataFrame:
    """
    Predict domain categories using archived screenshots.
    
    Args:
        input (list): List of URLs or domain names to classify
        archive_date: Target date as 'YYYYMMDD' string or datetime object
        image_path (str): Path to directory with screenshot images (optional)
        use_cache (bool): Whether to reuse existing screenshot files
        latest (bool): Whether to download latest model version
        
    Returns:
        pd.DataFrame: Image predictions using archived screenshots
    """
    if isinstance(archive_date, datetime):
        archive_date = archive_date.strftime('%Y%m%d')
        
    logger.info(f"Starting archive image prediction for {len(input)} URLs/domains from date {archive_date}")
    
    # Temporarily modify image extraction to use archive URLs
    original_method = Piedomain.extract_images
    
    def archive_extract_images(cls, urls_or_domains, use_cache, image_path):
        """Modified image extraction that uses archive.org snapshots."""
        errors = {}
        import os
        os.makedirs(image_path, exist_ok=True)
        
        for url_or_domain in urls_or_domains:
            domain = cls.parse_url_to_domain(url_or_domain)
            image_file = f"{image_path}/{domain}.png"
            
            if use_cache and os.path.exists(image_file):
                continue
                
            # Construct full URL if needed
            if not url_or_domain.startswith(('http://', 'https://')):
                full_url = f"https://{url_or_domain}"
            else:
                full_url = url_or_domain
                
            # Get archive snapshot URL
            snapshot_url = _fetch_archive_snapshot_url(full_url, archive_date)
            if not snapshot_url:
                errors[domain] = f"No archive snapshot found for {full_url} near {archive_date}"
                continue
                
            # Take screenshot of archived page
            try:
                success, error = cls._take_archive_screenshot(snapshot_url, image_file)
                if not success:
                    errors[domain] = error
            except Exception as e:
                errors[domain] = str(e)
                
        return True, errors
    
    # Add helper method for archive screenshots
    def _take_archive_screenshot(cls, snapshot_url: str, output_path: str):
        """Take screenshot of archive.org snapshot."""
        try:
            from selenium import webdriver
            from webdriver_manager.chrome import ChromeDriverManager
            from .config import get_config
            import time
            
            config = get_config()
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-extensions")
            options.add_argument("--no-sandbox")
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument(f"--window-size={config.webdriver_window_size}")
            options.add_argument(f"--user-agent={config.user_agent}")
            
            with webdriver.Chrome(
                service=webdriver.ChromeService(ChromeDriverManager().install()), 
                options=options
            ) as driver:
                driver.set_page_load_timeout(config.page_load_timeout)
                driver.get(snapshot_url)
                time.sleep(config.screenshot_wait_time)
                driver.save_screenshot(output_path)
                return True, ""
        except Exception as e:
            return False, str(e)
    
    # Temporarily replace the methods
    Piedomain.extract_images = classmethod(archive_extract_images)
    Piedomain._take_archive_screenshot = classmethod(_take_archive_screenshot)
    
    try:
        # Use existing image prediction with modified image extraction
        result = Piedomain.pred_shalla_cat_with_images(
            input=input,
            image_path=image_path or f"archive_images_{archive_date}",
            use_cache=use_cache,
            latest=latest
        )
        result['archive_date'] = archive_date
        return result
        
    finally:
        # Restore original method
        Piedomain.extract_images = original_method
        if hasattr(Piedomain, '_take_archive_screenshot'):
            delattr(Piedomain, '_take_archive_screenshot')