#!/usr/bin/env python3
"""
Content processor for coordinating HTML and image extraction.
Handles file I/O, caching, and coordination between different content types.
"""

import os
import time
from typing import List, Dict, Tuple, Optional
from PIL import Image
import numpy as np

from ..config import get_config
from ..logging import get_logger
from ..fetchers import get_fetcher
from .text_processor import TextProcessor

logger = get_logger()


class ContentProcessor:
    """Coordinates content extraction for domains using various fetchers."""
    
    def __init__(self, cache_dir: Optional[str] = None, archive_date: Optional[str] = None):
        """
        Initialize content processor.
        
        Args:
            cache_dir (str, optional): Directory for caching content
            archive_date (str, optional): Date for archive.org snapshots (YYYYMMDD format)
        """
        self.config = get_config()
        self.cache_dir = cache_dir or "cache"
        self.archive_date = archive_date
        self.fetcher = get_fetcher(archive_date)
        
        # Ensure cache directories exist
        self.html_dir = os.path.join(self.cache_dir, "html")
        self.image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
    
    def extract_html_content(self, domains: List[str], use_cache: bool = True) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Extract HTML content for domains.
        
        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached HTML files
            
        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: (html_content_dict, errors_dict)
        """
        html_content = {}
        errors = {}
        
        for domain in domains:
            domain_name = self._parse_domain_name(domain)
            html_file = os.path.join(self.html_dir, f"{domain_name}.html")
            
            # Check cache first
            if use_cache and os.path.exists(html_file):
                try:
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content[domain_name] = f.read()
                    continue
                except Exception as e:
                    logger.warning(f"Failed to read cached HTML for {domain_name}: {e}")
            
            # Fetch fresh content
            success, content, error = self.fetcher.fetch_html(domain)
            if success:
                # Save to cache
                try:
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    html_content[domain_name] = content
                except Exception as e:
                    logger.warning(f"Failed to cache HTML for {domain_name}: {e}")
                    html_content[domain_name] = content  # Still use the content
            else:
                errors[domain_name] = error
                
        return html_content, errors
    
    def extract_text_content(self, domains: List[str], use_cache: bool = True) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Extract and process text content from domains.
        
        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached content
            
        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: (processed_text_dict, errors_dict)
        """
        # Get HTML content first
        html_content, html_errors = self.extract_html_content(domains, use_cache)
        
        text_content = {}
        text_errors = html_errors.copy()
        
        # Process HTML to text
        for domain_name, html in html_content.items():
            try:
                processed_text = TextProcessor.process_html_to_text(html)
                text_content[domain_name] = processed_text
            except Exception as e:
                text_errors[domain_name] = f"Text processing error: {e}"
                
        return text_content, text_errors
    
    def extract_image_content(self, domains: List[str], use_cache: bool = True) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Extract screenshot images for domains.
        
        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached images
            
        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: (image_paths_dict, errors_dict)
        """
        image_paths = {}
        errors = {}
        
        for domain in domains:
            domain_name = self._parse_domain_name(domain)
            image_file = os.path.join(self.image_dir, f"{domain_name}.png")
            
            # Check cache first
            if use_cache and os.path.exists(image_file):
                image_paths[domain_name] = image_file
                continue
            
            # Take fresh screenshot
            success, error = self.fetcher.fetch_screenshot(domain, image_file)
            if success:
                image_paths[domain_name] = image_file
            else:
                errors[domain_name] = error
                
        return image_paths, errors
    
    def prepare_image_tensors(self, image_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
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
                # Load and resize image
                img = Image.open(image_path)
                img = img.convert('RGB')
                img = img.resize((254, 254))
                
                # Convert to numpy array and normalize
                img_array = np.array(img)
                img_array = img_array.astype('float32') / 255.0
                
                tensors[domain_name] = img_array
                
            except Exception as e:
                logger.error(f"Failed to process image for {domain_name}: {e}")
                
        return tensors
    
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