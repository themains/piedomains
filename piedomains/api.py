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
        >>> results = classifier.classify(["google.com", "facebook.com"])
        >>> for result in results:
        ...     print(f"{result['domain']}: {result['category']} ({result['confidence']:.3f})")
        google.com: search (0.892)
        facebook.com: socialnet (0.967)

        # Historical analysis
        >>> results = classifier.classify(["google.com"], archive_date="20200101")
        >>> print(f"Archive: {results[0]['category']} from {results[0]['date_time_collected']}")

    Example (LLM-based):
        >>> classifier = DomainClassifier()
        >>> classifier.configure_llm(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     api_key="sk-...",
        ...     categories=["news", "shopping", "social", "tech"]
        ... )
        >>> results = classifier.classify_by_llm(["cnn.com"])
        >>> print(f"LLM: {results[0]['category']} - {results[0]['reason']}")

    Example (Separated workflow):
        >>> collector = DataCollector()
        >>> collection = collector.collect(["example.com"])
        >>> text_results = classifier.classify_from_collection(collection, method="text")
        >>> image_results = classifier.classify_from_collection(collection, method="images")
        >>> # Same collected content, different classification approaches

    JSON Output Schema:
        All classification methods return List[Dict] with consistent structure:

        Collection Data Schema (from collect_content):
        {
            "collection_id": str,           # Unique identifier for collection
            "timestamp": str,               # ISO 8601 collection timestamp
            "config": {
                "cache_dir": str,           # Cache directory path
                "archive_date": str,        # Archive.org date (YYYYMMDD) or null
                "fetcher_type": str,        # "live" or "archive"
                "max_parallel": int         # Parallel fetch limit
            },
            "domains": [                    # List of domain results
                {
                    "url": str,             # Original input URL/domain
                    "domain": str,          # Parsed domain name
                    "text_path": str,       # Path to HTML file (relative to cache_dir)
                    "image_path": str,      # Path to screenshot (relative to cache_dir)
                    "date_time_collected": str,  # ISO 8601 timestamp
                    "fetch_success": bool,  # Whether data collection succeeded
                    "cached": bool,         # Whether data was retrieved from cache
                    "error": str,           # Error message if fetch_success is false
                    "title": str,           # Page title (optional)
                    "meta_description": str # Meta description (optional)
                }
            ],
            "summary": {
                "total_domains": int,       # Total domains requested
                "successful": int,          # Successfully collected
                "failed": int              # Failed collections
            }
        }

        Classification Result Schema (from classify methods):
        [
            {
                "url": str,                 # Original input URL/domain
                "domain": str,              # Parsed domain name
                "text_path": str,           # Path to HTML file
                "image_path": str,          # Path to screenshot
                "date_time_collected": str, # ISO 8601 timestamp
                "model_used": str,          # Model identifier (e.g. "text/shallalist_ml")
                "category": str,            # Predicted category
                "confidence": float,        # Confidence score (0.0-1.0)
                "reason": str,              # LLM reasoning (null for ML models)
                "error": str,               # Error message if classification failed
                "raw_predictions": dict,    # Full probability distribution

                # Combined classification specific fields:
                "text_category": str,       # Text-only prediction
                "text_confidence": float,   # Text confidence
                "image_category": str,      # Image-only prediction
                "image_confidence": float   # Image confidence
            }
        ]

        Supported Categories:
        adv, aggressive, alcohol, anonvpn, automobile, chatphisher, cooking,
        dating, downloads, drugs, education, finance, forum, gamble,
        government, hacking, health, hobby, homehealth, imagehosting,
        jobsearch, lingerie, music, news, occult, onlinemarketing, politics,
        porn, publicite, radiotv, recreation, religion, remotecontrol,
        shopping, socialnet, spyware, updatesites, urlshortener, violence,
        warez, weapons, webmail, webphone, webradio, webtv
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
        latest: bool = False,
    ) -> list[dict]:
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
            latest (bool): Whether to download latest model versions (default: False)

        Returns:
            list[dict]: Classification results in JSON format with fields:
                - url: Original URL/domain input
                - domain: Parsed domain name
                - text_path: Path to collected HTML file
                - image_path: Path to collected screenshot
                - date_time_collected: When data was collected (ISO format)
                - model_used: "combined/text_image_ml"
                - category: Best prediction (ensemble of text + image)
                - confidence: Confidence score (0-1)
                - reason: None (reasoning field for LLM models)
                - error: Error message if classification failed
                - text_category: Text-only prediction
                - text_confidence: Text confidence
                - image_category: Image-only prediction
                - image_confidence: Image confidence
                - raw_predictions: Full probability distributions

        Raises:
            ValueError: If domains list is empty

        Example:
            >>> classifier = DomainClassifier()
            >>> results = classifier.classify(["cnn.com", "bbc.com"])
            >>> print(f"{results[0]['domain']}: {results[0]['category']} ({results[0]['confidence']:.3f})")
            cnn.com: news (0.876)
        """
        if not domains:
            raise ValueError("domains list cannot be empty")

        # Step 1: Collect content using separated workflow
        collection_data = self.collect_content(
            domains=domains, archive_date=archive_date, use_cache=use_cache
        )

        # Step 2: Perform combined classification on collected data
        return self.classify_from_collection(
            collection_data=collection_data, method="combined", latest=latest
        )

    def classify_by_text(
        self,
        domains: list[str],
        archive_date: str | datetime | None = None,
        use_cache: bool = True,
        latest: bool = False,
    ) -> list[dict]:
        """
        Classify domains using only text content analysis.

        Faster than combined analysis, good for batch processing or when
        screenshots are not needed.

        Args:
            domains (list[str]): List of domain names or URLs to classify
            archive_date (str or datetime, optional): For historical analysis
            use_cache (bool): Whether to reuse cached content (default: True)
            latest (bool): Whether to download latest model versions (default: False)

        Returns:
            list[dict]: Text classification results in JSON format with fields:
                - url: Original URL/domain input
                - domain: Parsed domain name
                - text_path: Path to collected HTML file
                - image_path: Path to collected screenshot (may be None)
                - date_time_collected: When data was collected (ISO format)
                - model_used: "text/shallalist_ml"
                - category: Text classification prediction
                - confidence: Text confidence score (0-1)
                - reason: None (reasoning field for LLM models)
                - error: Error message if classification failed
                - raw_predictions: Full text probability distribution

        Example:
            >>> classifier = DomainClassifier()
            >>> results = classifier.classify_by_text(["wikipedia.org"])
            >>> print(f"{results[0]['domain']}: {results[0]['category']} ({results[0]['confidence']:.3f})")
            wikipedia.org: education (0.823)
        """
        if not domains:
            raise ValueError("domains list cannot be empty")

        # Step 1: Collect content using separated workflow
        collection_data = self.collect_content(
            domains=domains, archive_date=archive_date, use_cache=use_cache
        )

        # Step 2: Perform text classification on collected data
        return self.classify_from_collection(
            collection_data=collection_data, method="text", latest=latest
        )

    def classify_by_images(
        self,
        domains: list[str],
        archive_date: str | datetime | None = None,
        use_cache: bool = True,
        latest: bool = False,
    ) -> list[dict]:
        """
        Classify domains using only homepage screenshot analysis.

        Good for visual content classification, especially when text content
        is minimal or misleading.

        Args:
            domains (list[str]): List of domain names or URLs to classify
            archive_date (str or datetime, optional): For historical analysis
            use_cache (bool): Whether to reuse cached content (default: True)
            latest (bool): Whether to download latest model versions (default: False)

        Returns:
            list[dict]: Image classification results in JSON format with fields:
                - url: Original URL/domain input
                - domain: Parsed domain name
                - text_path: Path to collected HTML file (may be None)
                - image_path: Path to collected screenshot
                - date_time_collected: When data was collected (ISO format)
                - model_used: "image/shallalist_ml"
                - category: Image classification prediction
                - confidence: Image confidence score (0-1)
                - reason: None (reasoning field for LLM models)
                - error: Error message if classification failed
                - raw_predictions: Full image probability distribution

        Example:
            >>> classifier = DomainClassifier()
            >>> results = classifier.classify_by_images(["instagram.com"])
            >>> print(f"{results[0]['domain']}: {results[0]['category']} ({results[0]['confidence']:.3f})")
            instagram.com: socialnet (0.912)
        """
        if not domains:
            raise ValueError("domains list cannot be empty")

        # Step 1: Collect content using separated workflow
        collection_data = self.collect_content(
            domains=domains, archive_date=archive_date, use_cache=use_cache
        )

        # Step 2: Perform image classification on collected data
        return self.classify_from_collection(
            collection_data=collection_data, method="images", latest=latest
        )

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
        from .llm.config import LLMConfig
        from .llm_classifier import LLMClassifier

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
        mode: str = "text",
    ) -> list[dict]:
        """
        Classify domains using LLM analysis.

        Args:
            domains: List of domain names to classify
            custom_instructions: Optional custom classification instructions
            use_cache: Whether to use cached content (default: True)
            mode: LLM mode - "text", "image", or "multimodal" (default: "text")

        Returns:
            list[dict]: LLM classification results in JSON format with fields:
                - url: Original URL/domain input
                - domain: Parsed domain name
                - text_path: Path to collected HTML file
                - image_path: Path to collected screenshot (if applicable)
                - date_time_collected: When data was collected (ISO format)
                - model_used: "text/llm_{provider}_{model}" or similar
                - category: LLM classification prediction
                - confidence: LLM confidence score (0-1)
                - reason: LLM reasoning explanation
                - error: Error message if classification failed

        Raises:
            RuntimeError: If LLM not configured

        Example:
            >>> classifier = DomainClassifier()
            >>> classifier.configure_llm("openai", "gpt-4o", api_key="sk-...")
            >>> results = classifier.classify_by_llm(["cnn.com", "amazon.com"])
            >>> print(f"{results[0]['domain']}: {results[0]['category']} - {results[0]['reason']}")
            cnn.com: news - This domain contains current events and journalism content
        """
        if self._llm_classifier is None:
            raise RuntimeError("LLM not configured. Call configure_llm() first.")

        if not domains:
            raise ValueError("domains list cannot be empty")

        # Step 1: Collect content using separated workflow
        collection_data = self.collect_content(domains=domains, use_cache=use_cache)

        # Step 2: Perform LLM classification on collected data
        return self._llm_classifier.classify_from_data(
            collection_data=collection_data, mode=mode
        )

    def classify_by_llm_multimodal(
        self,
        domains: list[str],
        custom_instructions: str | None = None,
        use_cache: bool = True,
    ) -> list[dict]:
        """
        Classify domains using LLM multimodal analysis (text + screenshots).

        Args:
            domains: List of domain names to classify
            custom_instructions: Optional custom classification instructions
            use_cache: Whether to use cached content (default: True)

        Returns:
            list[dict]: Multimodal LLM classification results in JSON format

        Raises:
            RuntimeError: If LLM not configured

        Example:
            >>> classifier = DomainClassifier()
            >>> classifier.configure_llm("openai", "gpt-4o", api_key="sk-...")
            >>> results = classifier.classify_by_llm_multimodal(["cnn.com"])
            >>> print(f"{results[0]['domain']}: {results[0]['category']} - {results[0]['reason']}")
            cnn.com: news - Based on text content and visual layout typical of news websites
        """
        return self.classify_by_llm(
            domains=domains,
            custom_instructions=custom_instructions,
            use_cache=use_cache,
            mode="multimodal",
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

    def collect_content(
        self,
        domains: list[str],
        archive_date: str | datetime | None = None,
        collection_id: str | None = None,
        use_cache: bool = True,
        batch_size: int = 10,
    ) -> dict:
        """
        Collect website content for domains without performing inference.

        Separates content collection from classification, enabling:
        - Content reuse across multiple models
        - Clear data lineage and inspection
        - Reproducible analysis workflows

        Args:
            domains (list[str]): List of domain names or URLs to collect content for
            archive_date (str or datetime, optional): For historical analysis
            collection_id (str, optional): Identifier for this collection
            use_cache (bool): Whether to use cached content when available
            batch_size (int): Number of domains to process in parallel

        Returns:
            dict: Collection metadata with file paths for downstream inference

        Example:
            >>> classifier = DomainClassifier()
            >>> collection = classifier.collect_content(["cnn.com", "bbc.com"])
            >>> print(collection["domains"][0]["text_path"])
            html/cnn.com.html
        """
        archive_date = self._normalize_archive_date(archive_date)

        # Import DataCollector
        from .data_collector import DataCollector

        # Create data collector with appropriate settings
        collector = DataCollector(cache_dir=self.cache_dir, archive_date=archive_date)

        # Collect data using batch processing for efficiency
        if len(domains) > batch_size:
            return collector.collect_batch(
                domains,
                collection_id=collection_id,
                use_cache=use_cache,
                batch_size=batch_size,
            )
        else:
            return collector.collect(
                domains, collection_id=collection_id, use_cache=use_cache
            )

    def classify_from_collection(
        self,
        collection_data: dict,
        method: str = "combined",
        output_file: str | None = None,
        latest: bool = False,
    ) -> list[dict]:
        """
        Perform inference on previously collected content.

        Args:
            collection_data (dict): Collection metadata from collect_content()
            method (str): Classification method - "text", "images", "combined", or "llm"
            output_file (str, optional): Path to save JSON results
            latest (bool): Whether to use latest model versions (default: False)

        Returns:
            list[dict]: Classification results in JSON format

        Example:
            >>> classifier = DomainClassifier()
            >>> collection = classifier.collect_content(["cnn.com"])
            >>> results = classifier.classify_from_collection(collection, method="text")
            >>> print(results[0]["category"])
            news
        """
        if method not in ["text", "images", "combined", "llm"]:
            raise ValueError("method must be 'text', 'images', 'combined', or 'llm'")

        if method == "text":
            from .text import TextClassifier

            classifier = TextClassifier(cache_dir=self.cache_dir)
            return classifier.classify_from_data(collection_data, output_file, latest)

        elif method == "images":
            from .image import ImageClassifier

            classifier = ImageClassifier(cache_dir=self.cache_dir)
            return classifier.classify_from_data(collection_data, output_file, latest)

        elif method == "combined":
            # Run both text and image classification, then combine
            from .image import ImageClassifier
            from .text import TextClassifier

            text_classifier = TextClassifier(cache_dir=self.cache_dir)
            image_classifier = ImageClassifier(cache_dir=self.cache_dir)

            text_results = text_classifier.classify_from_data(
                collection_data, latest=latest
            )
            image_results = image_classifier.classify_from_data(
                collection_data, latest=latest
            )

            # Combine results using simple ensemble (equal weighting)
            combined_results = []

            # Create lookup for image results
            image_lookup = {r["domain"]: r for r in image_results}

            for text_result in text_results:
                domain = text_result["domain"]
                image_result = image_lookup.get(domain, {})

                # Combine predictions with equal weighting
                text_conf = text_result.get("confidence", 0.0) or 0.0
                image_conf = image_result.get("confidence", 0.0) or 0.0

                if text_conf > 0 and image_conf > 0:
                    # Both models have predictions - use ensemble
                    combined_conf = (text_conf + image_conf) / 2
                    # Use text category as primary (could be more sophisticated)
                    category = text_result.get("category") or image_result.get(
                        "category"
                    )
                elif text_conf > 0:
                    # Only text model has prediction
                    combined_conf = text_conf
                    category = text_result.get("category")
                elif image_conf > 0:
                    # Only image model has prediction
                    combined_conf = image_conf
                    category = image_result.get("category")
                else:
                    # No valid predictions
                    combined_conf = 0.0
                    category = None

                combined_result = {
                    "url": text_result.get("url"),
                    "domain": domain,
                    "text_path": text_result.get("text_path"),
                    "image_path": text_result.get("image_path"),
                    "date_time_collected": text_result.get("date_time_collected"),
                    "model_used": "combined/text_image_ml",
                    "category": category,
                    "confidence": combined_conf,
                    "reason": None,
                    "error": text_result.get("error") or image_result.get("error"),
                    "text_category": text_result.get("category"),
                    "text_confidence": text_conf,
                    "image_category": image_result.get("category"),
                    "image_confidence": image_conf,
                    "raw_predictions": {
                        "text": text_result.get("raw_predictions"),
                        "image": image_result.get("raw_predictions"),
                    },
                }
                combined_results.append(combined_result)

            # Save combined results if requested
            if output_file:
                import json
                from datetime import datetime

                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                output_data = {
                    "inference_timestamp": datetime.utcnow().isoformat() + "Z",
                    "model_used": "combined/text_image_ml",
                    "total_domains": len(combined_results),
                    "successful": len(
                        [r for r in combined_results if r["category"] is not None]
                    ),
                    "failed": len(
                        [r for r in combined_results if r["error"] is not None]
                    ),
                    "results": combined_results,
                }

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2)

            return combined_results

        elif method == "llm":
            if self._llm_classifier is None:
                raise RuntimeError("LLM not configured. Call configure_llm() first.")
            return self._llm_classifier.classify_from_data(
                collection_data, output_file, mode="multimodal"
            )

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
) -> list[dict]:
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
) -> list[dict]:
    """Quick domain classification function.

    Args:
        domains: List of domain names or URLs to classify
        method: Classification method - "combined", "text", or "images"
        archive_date: Optional historical date for archive.org analysis
        cache_dir: Optional cache directory override

    Returns:
        list[dict]: Classification results in JSON format

    Example:
        >>> results = classify_domains(["cnn.com", "github.com"])
        >>> for result in results:
        ...     print(f"{result['domain']}: {result['category']} ({result['confidence']:.3f})")
        cnn.com: news (0.876)
        github.com: computers (0.892)
    """

    current = globals().get("classify_domains")
    if current is not _classify_domains_wrapper:
        return current(
            domains, method=method, archive_date=archive_date, cache_dir=cache_dir
        )

    return _classify_domains_impl(domains, method, archive_date, cache_dir)


# Store original function object for patch detection
_classify_domains_wrapper = classify_domains
