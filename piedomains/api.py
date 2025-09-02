#!/usr/bin/env python3
"""Modern, intuitive API for piedomains domain classification.

This module provides a clean, class-based interface for domain content
classification with support for text analysis, image analysis, and
historical archive.org snapshots.
"""

import os
import re
import warnings
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd

from .classifiers import CombinedClassifier, ImageClassifier, TextClassifier
from .logging import get_logger

logger = get_logger()


class DomainClassifier:
    """
    Main interface for domain content classification.
    
    Supports text-based, image-based, and combined classification approaches.
    Can analyze current content or historical snapshots from archive.org.
    
    Example:
        >>> classifier = DomainClassifier()
        >>> result = classifier.classify(["google.com", "facebook.com"])
        >>> print(result[['domain', 'pred_label', 'pred_prob']])
        
        # Historical analysis
        >>> result = classifier.classify(["google.com"], archive_date="20200101")
        
        # Text-only analysis
        >>> result = classifier.classify_by_text(["google.com"])
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize domain classifier.
        
        Args:
            cache_dir (str, optional): Directory for caching downloaded content.
                                     Defaults to "cache" in current directory.
        """
        self.cache_dir = cache_dir or "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Initialized DomainClassifier with cache_dir: {self.cache_dir}")

    def _normalize_archive_date(self, archive_date: Optional[Union[str, datetime]]) -> Optional[str]:
        """Validate and normalise archive date.

        Args:
            archive_date: Date string in ``YYYYMMDD`` format or ``datetime``.

        Returns:
            Normalised date string or ``None``.

        Raises:
            ValueError: If the date is invalid or outside allowed range.
        """
        if archive_date is None:
            return None

        if isinstance(archive_date, datetime):
            archive_date = archive_date.strftime("%Y%m%d")

        if not isinstance(archive_date, str) or not re.fullmatch(r"\d{8}", archive_date):
            raise ValueError("archive_date must be in YYYYMMDD format")

        try:
            parsed = datetime.strptime(archive_date, "%Y%m%d")
        except ValueError as exc:  # invalid date
            raise ValueError("archive_date must be a valid date in YYYYMMDD format") from exc

        if parsed < datetime(2000, 1, 1) or parsed > datetime.now():
            raise ValueError("archive_date must be between 20000101 and today's date")

        return archive_date
    
    def classify(self, 
                 domains: List[str], 
                 archive_date: Optional[Union[str, datetime]] = None,
                 use_cache: bool = True,
                 latest_models: bool = False) -> pd.DataFrame:
        """
        Classify domains using combined text and image analysis.
        
        This is the most comprehensive classification method, using both
        textual content and homepage screenshots for maximum accuracy.
        
        Args:
            domains (List[str]): List of domain names or URLs to classify
                               e.g., ["google.com", "https://facebook.com/page"]
            archive_date (str or datetime, optional): For historical analysis.
                                                    Format: "YYYYMMDD" or datetime object
            use_cache (bool): Whether to reuse cached content (default: True)
            latest_models (bool): Whether to download latest model versions (default: False)
            
        Returns:
            pd.DataFrame: Results with columns:
                - domain: Domain name
                - pred_label: Best prediction (ensemble of text + image)
                - pred_prob: Confidence score (0-1)
                - text_label: Text-only prediction
                - text_prob: Text confidence
                - image_label: Image-only prediction  
                - image_prob: Image confidence
                - used_domain_text: Whether text analysis succeeded
                - used_domain_screenshot: Whether image analysis succeeded
                - extracted_text: Processed text content
                - archive_date: If historical analysis was used
                - error: Error message if classification failed
                
        Raises:
            ValueError: If domains list is empty
            
        Example:
            >>> classifier = DomainClassifier()
            >>> result = classifier.classify(["cnn.com", "pornhub.com"])
            >>> print(result[['domain', 'pred_label', 'pred_prob']])
               domain pred_label  pred_prob
            0  cnn.com       news   0.876543
            1  pornhub.com   porn   0.923456
        """
        archive_date = self._normalize_archive_date(archive_date)
        
        # Create combined classifier
        classifier = CombinedClassifier(self.cache_dir, archive_date)
        
        # Perform classification
        return classifier.predict(domains, use_cache, latest_models)
    
    def classify_by_text(self,
                        domains: List[str],
                        archive_date: Optional[Union[str, datetime]] = None,
                        use_cache: bool = True,
                        latest_models: bool = False) -> pd.DataFrame:
        """
        Classify domains using only text content analysis.
        
        Faster than combined analysis, good for batch processing or when
        screenshots are not needed.
        
        Args:
            domains (List[str]): List of domain names or URLs to classify
            archive_date (str or datetime, optional): For historical analysis
            use_cache (bool): Whether to reuse cached content (default: True)
            latest_models (bool): Whether to download latest model versions (default: False)
            
        Returns:
            pd.DataFrame: Results with text-based predictions
            
        Example:
            >>> classifier = DomainClassifier()
            >>> result = classifier.classify_by_text(["wikipedia.org"])
            >>> print(result[['domain', 'text_label', 'text_prob']])
        """
        archive_date = self._normalize_archive_date(archive_date)
        
        # Create text classifier
        classifier = TextClassifier(self.cache_dir, archive_date)
        
        # Perform classification
        return classifier.predict(domains, use_cache, latest_models)
    
    def classify_by_images(self,
                          domains: List[str],
                          archive_date: Optional[Union[str, datetime]] = None,
                          use_cache: bool = True,
                          latest_models: bool = False) -> pd.DataFrame:
        """
        Classify domains using only homepage screenshot analysis.
        
        Good for visual content classification, especially when text content
        is minimal or misleading.
        
        Args:
            domains (List[str]): List of domain names or URLs to classify
            archive_date (str or datetime, optional): For historical analysis
            use_cache (bool): Whether to reuse cached content (default: True)  
            latest_models (bool): Whether to download latest model versions (default: False)
            
        Returns:
            pd.DataFrame: Results with image-based predictions
            
        Example:
            >>> classifier = DomainClassifier()
            >>> result = classifier.classify_by_images(["instagram.com"])
            >>> print(result[['domain', 'image_label', 'image_prob']])
        """
        archive_date = self._normalize_archive_date(archive_date)
        
        # Create image classifier
        classifier = ImageClassifier(self.cache_dir, archive_date)
        
        # Perform classification
        return classifier.predict(domains, use_cache, latest_models)
    
    def classify_batch(self,
                      domains: List[str],
                      method: str = "combined",
                      archive_date: Optional[Union[str, datetime]] = None,
                      use_cache: bool = True,
                      latest_models: bool = False,
                      batch_size: int = 10,
                      show_progress: bool = True) -> pd.DataFrame:
        """
        Classify large batches of domains with progress tracking.
        
        Args:
            domains (List[str]): List of domain names or URLs to classify
            method (str): Classification method - "combined", "text", or "images"
            archive_date (str or datetime, optional): For historical analysis
            use_cache (bool): Whether to reuse cached content (default: True)
            latest_models (bool): Whether to download latest model versions (default: False)
            batch_size (int): Number of domains to process at once (default: 10)
            show_progress (bool): Whether to show progress bar (default: True)
            
        Returns:
            pd.DataFrame: Combined results from all batches
            
        Example:
            >>> classifier = DomainClassifier()
            >>> domains = ["site1.com", "site2.com", ...] # 1000 domains
            >>> result = classifier.classify_batch(domains, method="text", batch_size=50)
        """
        if method not in ["combined", "text", "images"]:
            raise ValueError("method must be 'combined', 'text', or 'images'")
        
        if show_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(domains), desc=f"Classifying ({method})")
            except ImportError:
                logger.warning("tqdm not available, progress bar disabled")
                show_progress = False
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(domains), batch_size):
            batch = domains[i:i + batch_size]

            try:
                if method == "combined":
                    batch_result = self.classify(batch, archive_date, use_cache, latest_models)
                elif method == "text":
                    batch_result = self.classify_by_text(batch, archive_date, use_cache, latest_models)
                else:  # images
                    batch_result = self.classify_by_images(batch, archive_date, use_cache, latest_models)

                if len(batch_result) > len(batch):
                    batch_result = batch_result.head(len(batch))

                all_results.append(batch_result)
                
            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1} failed: {e}")
                # Create error result for this batch
                error_result = pd.DataFrame([
                    {'domain': self._parse_domain_name(d), 'error': str(e)} 
                    for d in batch
                ])
                all_results.append(error_result)
            
            if show_progress:
                progress_bar.update(len(batch))
        
        if show_progress:
            progress_bar.close()
        
        # Combine all results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _parse_domain_name(self, url_or_domain: str) -> str:
        """Extract domain name from URL or domain string."""
        # Import here to avoid circular imports
        from .piedomain import Piedomain
        return Piedomain.parse_url_to_domain(url_or_domain)


# Convenience functions for quick access
def _classify_domains_impl(domains: List[str],
                           method: str = "combined",
                           archive_date: Optional[Union[str, datetime]] = None,
                           cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Internal implementation for :func:`classify_domains`."""
    classifier = DomainClassifier(cache_dir)

    if method == "combined":
        return classifier.classify(domains, archive_date)
    elif method == "text":
        return classifier.classify_by_text(domains, archive_date)
    elif method == "images":
        return classifier.classify_by_images(domains, archive_date)
    else:
        raise ValueError("method must be 'combined', 'text', or 'images'")


def classify_domains(domains: List[str],
                    method: str = "combined",
                    archive_date: Optional[Union[str, datetime]] = None,
                    cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Quick domain classification function.

    This wrapper allows the function to be easily patched in tests while the
    implementation lives in :func:`_classify_domains_impl`.
    """

    current = globals().get("classify_domains")
    if current is not _classify_domains_wrapper:
        return current(domains, method=method, archive_date=archive_date, cache_dir=cache_dir)

    return _classify_domains_impl(domains, method, archive_date, cache_dir)


# Store original function object for patch detection
_classify_domains_wrapper = classify_domains


# Backward compatibility functions with deprecation warnings
def pred_shalla_cat_modern(*args, **kwargs):
    """Modern wrapper for old pred_shalla_cat function."""
    warnings.warn(
        "pred_shalla_cat is deprecated. Use DomainClassifier.classify() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Convert old API to new API
    # This would need to be implemented based on the old function signature