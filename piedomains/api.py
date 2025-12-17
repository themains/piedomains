#!/usr/bin/env python3
"""Modern, intuitive API for piedomains domain classification.

This module provides a clean, class-based interface for domain content
classification with support for text analysis, image analysis, and
historical archive.org snapshots.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

import pandas as pd

from .classifiers import CombinedClassifier, ImageClassifier, TextClassifier

# LLM imports happen lazily when needed
from .piedomains_logging import get_logger

logger = get_logger()


class DomainClassifier:
    """
    Main interface for domain content classification.

    Supports multiple classification approaches:
    - Traditional ML: Text-based, image-based, and combined classification
    - Modern AI: LLM-based classification with multimodal support
    - Historical analysis via archive.org snapshots

    Example (Traditional ML):
        >>> classifier = DomainClassifier()
        >>> result = classifier.classify(["google.com", "facebook.com"])
        >>> print(result[['domain', 'pred_label', 'pred_prob']])

        # Historical analysis
        >>> result = classifier.classify(["google.com"], archive_date="20200101")

    Example (LLM-based):
        >>> classifier = DomainClassifier()
        >>> classifier.configure_llm(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     api_key="sk-...",
        ...     categories=["news", "shopping", "social", "tech"]
        ... )
        >>> result = classifier.classify_by_llm(["cnn.com"])
        >>> result = classifier.classify_by_llm_multimodal(["amazon.com"])
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize domain classifier.

        Args:
            cache_dir (str, optional): Directory for caching downloaded content.
                                     Defaults to "cache" in current directory.
        """
        self.cache_dir = cache_dir or "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self._llm_config = None
        self._llm_classifier = None
        logger.info(f"Initialized DomainClassifier with cache_dir: {self.cache_dir}")

    def _normalize_archive_date(
        self, archive_date: str | datetime | None
    ) -> str | None:
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

        if not isinstance(archive_date, str) or not re.fullmatch(
            r"\d{8}", archive_date
        ):
            raise ValueError("archive_date must be in YYYYMMDD format")

        try:
            parsed = datetime.strptime(archive_date, "%Y%m%d")
        except ValueError as exc:  # invalid date
            raise ValueError(
                "archive_date must be a valid date in YYYYMMDD format"
            ) from exc

        if parsed < datetime(2000, 1, 1) or parsed > datetime.now():
            raise ValueError("archive_date must be between 20000101 and today's date")

        return archive_date

    def classify(
        self,
        domains: list[str],
        archive_date: str | datetime | None = None,
        use_cache: bool = True,
        latest_models: bool = False,
    ) -> pd.DataFrame:
        """
        Classify domains using combined text and image analysis.

        This is the most comprehensive classification method, using both
        textual content and homepage screenshots for maximum accuracy.

        Args:
            domains (list[str]): List of domain names or URLs to classify
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

    def classify_by_text(
        self,
        domains: list[str],
        archive_date: str | datetime | None = None,
        use_cache: bool = True,
        latest_models: bool = False,
    ) -> pd.DataFrame:
        """
        Classify domains using only text content analysis.

        Faster than combined analysis, good for batch processing or when
        screenshots are not needed.

        Args:
            domains (list[str]): List of domain names or URLs to classify
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

    def classify_by_images(
        self,
        domains: list[str],
        archive_date: str | datetime | None = None,
        use_cache: bool = True,
        latest_models: bool = False,
    ) -> pd.DataFrame:
        """
        Classify domains using only homepage screenshot analysis.

        Good for visual content classification, especially when text content
        is minimal or misleading.

        Args:
            domains (list[str]): List of domain names or URLs to classify
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

    def classify_batch(
        self,
        domains: list[str],
        method: str = "combined",
        archive_date: str | datetime | None = None,
        use_cache: bool = True,
        latest_models: bool = False,
        batch_size: int = 10,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Classify large batches of domains with progress tracking.

        Args:
            domains (list[str]): List of domain names or URLs to classify
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
            batch = domains[i : i + batch_size]

            try:
                if method == "combined":
                    batch_result = self.classify(
                        batch, archive_date, use_cache, latest_models
                    )
                elif method == "text":
                    batch_result = self.classify_by_text(
                        batch, archive_date, use_cache, latest_models
                    )
                else:  # images
                    batch_result = self.classify_by_images(
                        batch, archive_date, use_cache, latest_models
                    )

                if len(batch_result) > len(batch):
                    batch_result = batch_result.head(len(batch))

                all_results.append(batch_result)

            except Exception as e:
                logger.error(f"Batch {i // batch_size + 1} failed: {e}")
                # Create error result for this batch
                error_result = pd.DataFrame(
                    [
                        {"domain": self._parse_domain_name(d), "error": str(e)}
                        for d in batch
                    ]
                )
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

    def configure_llm(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        categories: list[str] | None = None,
        **kwargs,
    ) -> None:
        """
        Configure LLM for AI-powered domain classification.

        Args:
            provider: LLM provider ('openai', 'anthropic', 'google', etc.)
            model: Model name ('gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro')
            api_key: API key for the provider (or set via environment variable)
            categories: Custom classification categories
            **kwargs: Additional LLMConfig parameters (temperature, max_tokens, etc.)

        Example:
            >>> classifier = DomainClassifier()
            >>> classifier.configure_llm(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     api_key="sk-...",
            ...     categories=["news", "shopping", "social", "tech"]
            ... )
        """
        # Import LLM classes - these are required dependencies
        from .classifiers.llm_classifier import LLMClassifier
        from .llm.config import LLMConfig

        self._llm_config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            categories=categories,
            **kwargs,
        )

        self._llm_classifier = LLMClassifier(self._llm_config)
        logger.info(f"Configured LLM: {provider}/{model}")

    def classify_by_llm(
        self,
        domains: list[str],
        custom_instructions: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Classify domains using LLM text analysis.

        Args:
            domains: List of domain names to classify
            custom_instructions: Optional custom classification instructions
            use_cache: Whether to use cached content (default: True)

        Returns:
            pd.DataFrame: Results with LLM classifications

        Raises:
            RuntimeError: If LLM not configured

        Example:
            >>> classifier = DomainClassifier()
            >>> classifier.configure_llm("openai", "gpt-4o", api_key="sk-...")
            >>> result = classifier.classify_by_llm(["cnn.com", "amazon.com"])
        """
        if self._llm_classifier is None:
            raise RuntimeError("LLM not configured. Call configure_llm() first.")

        # Get text content using existing infrastructure
        text_classifier = TextClassifier(self.cache_dir)
        text_results = text_classifier.predict(domains, use_cache, latest=False)

        # Extract content for LLM
        content_dict = {}
        for _, row in text_results.iterrows():
            domain = row.get("domain", "")
            content = row.get("extracted_text", "")
            if domain and content:
                content_dict[domain] = content

        # Classify with LLM
        return self._llm_classifier.classify_text(
            domains, content_dict, custom_instructions
        )

    def classify_by_llm_multimodal(
        self,
        domains: list[str],
        custom_instructions: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Classify domains using LLM multimodal analysis (text + screenshots).

        Args:
            domains: List of domain names to classify
            custom_instructions: Optional custom classification instructions
            use_cache: Whether to use cached content (default: True)

        Returns:
            pd.DataFrame: Results with multimodal LLM classifications

        Raises:
            RuntimeError: If LLM not configured

        Example:
            >>> classifier = DomainClassifier()
            >>> classifier.configure_llm("openai", "gpt-4o", api_key="sk-...")
            >>> result = classifier.classify_by_llm_multimodal(["cnn.com"])
        """
        if self._llm_classifier is None:
            raise RuntimeError("LLM not configured. Call configure_llm() first.")

        # Get both text and image content using existing infrastructure
        combined_classifier = CombinedClassifier(self.cache_dir)
        combined_results = combined_classifier.predict(domains, use_cache, latest=False)

        # Extract content and screenshots for LLM
        content_dict = {}
        screenshot_dict = {}

        for _, row in combined_results.iterrows():
            domain = row.get("domain", "")
            content = row.get("extracted_text", "")

            if domain:
                if content:
                    content_dict[domain] = content

                # Check if screenshot was captured
                if row.get("used_domain_screenshot", False):
                    # Screenshot should be in cache
                    screenshot_path = os.path.join(
                        self.cache_dir, "images", f"{domain}.png"
                    )
                    if os.path.exists(screenshot_path):
                        screenshot_dict[domain] = screenshot_path

        # Classify with LLM multimodal
        return self._llm_classifier.classify_multimodal(
            domains, content_dict, screenshot_dict, custom_instructions
        )

    def get_llm_usage_stats(self) -> dict | None:
        """
        Get LLM usage statistics and cost tracking.

        Returns:
            Dictionary with usage stats or None if LLM not configured

        Example:
            >>> classifier = DomainClassifier()
            >>> classifier.configure_llm("openai", "gpt-4o")
            >>> classifier.classify_by_llm(["example.com"])
            >>> stats = classifier.get_llm_usage_stats()
            >>> print(f"Cost: ${stats['estimated_cost_usd']:.4f}")
        """
        if self._llm_classifier is None:
            return None
        return self._llm_classifier.get_usage_stats()

    def _parse_domain_name(self, url_or_domain: str) -> str:
        """Extract domain name from URL or domain string."""
        # Import here to avoid circular imports
        from .piedomain import Piedomain

        return Piedomain.parse_url_to_domain(url_or_domain)


# Convenience functions for quick access
def _classify_domains_impl(
    domains: list[str],
    method: str = "combined",
    archive_date: str | datetime | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
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


def classify_domains(
    domains: list[str],
    method: str = "combined",
    archive_date: str | datetime | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Quick domain classification function.

    This wrapper allows the function to be easily patched in tests while the
    implementation lives in :func:`_classify_domains_impl`.
    """

    current = globals().get("classify_domains")
    if current is not _classify_domains_wrapper:
        return current(
            domains, method=method, archive_date=archive_date, cache_dir=cache_dir
        )

    return _classify_domains_impl(domains, method, archive_date, cache_dir)


# Store original function object for patch detection
_classify_domains_wrapper = classify_domains
