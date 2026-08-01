#!/usr/bin/env python3
"""Modern, intuitive API for piedomains domain classification.

This module provides a clean, class-based interface for domain content
classification with support for text analysis, image analysis, and
historical archive.org snapshots.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import UTC, datetime

# LLM imports happen lazily when needed
from .outcomes import ErrorCode, Stage, annotate, build_report, classify_exception
from .piedomains_logging import bind_context, get_logger

logger = get_logger()


class DomainClassifier:
    """Main interface for domain content classification.

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

        Collection Data Schema (from collect_content)::

            {
                "collection_id": str,       # Unique identifier for collection
                "timestamp": str,           # ISO 8601 collection timestamp
                "config": {
                    "cache_dir": str,       # Cache directory path
                    "archive_date": str,    # Archive.org date (YYYYMMDD) or null
                    "fetcher_type": str,    # "live" or "archive"
                    "max_parallel": int     # Parallel fetch limit
                },
                "domains": [               # List of domain results
                    {
                        "url": str,         # Original input URL/domain
                        "domain": str,      # Parsed domain name
                        "text_path": str,   # HTML file, relative to cache_dir
                        "image_path": str,  # Screenshot, relative to cache_dir
                        "date_time_collected": str,  # ISO 8601 timestamp
                        "fetch_success": bool,  # Whether collection succeeded
                        "cached": bool,     # Whether data came from cache
                        "error": str,       # Error if fetch_success is false
                        "title": str,       # Page title (optional)
                        "meta_description": str  # Meta description (optional)
                    }
                ],
                "summary": {
                    "total_domains": int,   # Total domains requested
                    "successful": int,      # Successfully collected
                    "failed": int           # Failed collections
                }
            }

        Classification Result Schema (from classify methods)::

            [
                {
                    "url": str,             # Original input URL/domain
                    "domain": str,          # Parsed domain name
                    "text_path": str,       # Path to HTML file
                    "image_path": str,      # Path to screenshot
                    "date_time_collected": str,  # ISO 8601 timestamp
                    "model_used": str,      # e.g. "text/shallalist_ml"
                    "category": str,        # Predicted category
                    "confidence": float,    # Confidence score (0.0-1.0)
                    "reason": str,          # LLM reasoning (null for ML models)
                    "error": str,           # Error if classification failed
                    "raw_predictions": dict,  # Full probability distribution

                    # Combined classification specific fields:
                    "text_category": str,   # Text-only prediction
                    "text_confidence": float,   # Text confidence
                    "image_category": str,  # Image-only prediction
                    "image_confidence": float   # Image confidence
                }
            ]

        Supported Categories:
            The 44 categories in :data:`piedomains.constants.classes`:
            adult, alcohol, automobile, dating, downloads, drugs, education,
            finance, fortunetelling, forum, gamble, government, hobby/cooking,
            hobby/games, hobby/gardening, hobby/pets, homestyle, hospitals,
            imagehosting, isp, jobsearch, library, military, movies, music,
            news, politics, radiotv, realestate, recreation/humor,
            recreation/restaurants, recreation/sports, recreation/travel,
            recreation/wellness, religion, science, searchengines, shopping,
            socialnet, urlshortener, weapons, webmail, parked, unavailable.

            Two kinds of absence, for two different reasons.

            Categories describing how a site is hosted or monetised
            (adv, tracker, spyware, redirector) are absent because a page does
            not state them. So are those asking about delivery mechanism or
            legality rather than subject: games split by whether they are played
            online, radio split by whether it is broadcast, downloads split by
            whether the licence permitted them.

            `parked` and `unavailable` are present for the mirror-image reason.
            A domain that resolves to a for-sale page or a server autoindex has
            no site to classify, and saying so is both plainly readable from the
            text and the answer a caller can act on.

            See training/taxonomy.py.
    """

    def __init__(self, cache_dir: str | None = None):
        """Initialize domain classifier.

        Args:
            cache_dir: Directory for caching downloaded content.
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

    def _run(
        self,
        domains: list[str],
        method: str,
        archive_date: str | datetime | None,
        use_cache: bool,
        latest: bool,
    ) -> dict:
        """Collect, classify, annotate and report on a batch of domains.

        Shared by every ``classify*`` entry point so that outcome annotation
        and the run report are produced identically regardless of method.

        Args:
            domains: Domain names or URLs to classify.
            method: One of ``text``, ``images``, ``combined``, ``llm``.
            archive_date: Optional historical date for archive.org analysis.
            use_cache: Whether to reuse cached content.
            latest: Whether to fetch the newest model artifacts.

        Returns:
            dict: ``{"results": [...], "report": {...}}``.

        Raises:
            ValueError: If ``domains`` is empty.
        """
        if not domains:
            raise ValueError("domains list cannot be empty")

        run_id = uuid.uuid4().hex[:12]
        started = datetime.now(UTC)
        bind_context(run_id=run_id)
        logger.info(f"Run {run_id} started: {len(domains)} domains, method={method}")

        try:
            collection_data = self.collect_content(
                domains=domains, archive_date=archive_date, use_cache=use_cache
            )
            results = self.classify_from_collection(
                collection_data=collection_data, method=method, latest=latest
            )
            results = _annotate_results(results, collection_data)
            results = _reconcile_with_requested(results, domains, collection_data)
            report = build_report(
                results,
                run_id=run_id,
                started_at=started,
                finished_at=datetime.now(UTC),
            )
            logger.info(
                f"Run {run_id} finished: {report['classified']}/{report['total']} "
                f"classified, {report['failed']} failed",
                extra={"by_reason": report["by_reason"]},
            )
            return {"results": results, "report": report}
        finally:
            bind_context(run_id=None)

    def classify(
        self,
        domains: list[str],
        archive_date: str | datetime | None = None,
        use_cache: bool = True,
        latest: bool = False,
        use_screenshots: bool = False,
    ) -> dict:
        """Classify domains from their page text.

        **Screenshots are opt-in, and the reason is measured.** Combining the two models
        by calibrated late fusion was fitted on 1,704 held-out paired domains and scored
        on 1,742 more:

        ===================  ========  ========
        model                accuracy  macro-F1
        ===================  ========  ========
        text only               0.794     0.699
        image only              0.429     0.306
        fused (per-class)       0.798     0.700
        ===================  ========  ========

        +0.001 macro-F1 is noise at that sample size, and the fitted text weight is
        0.973 — the optimiser puts almost nothing on the screenshot. So the default does
        not pay for loading a 350MB vision model and running it per domain.

        Pass ``use_screenshots=True`` to fuse anyway. It is honest about its own value:
        without published fusion weights, or with the screenshot model unavailable, it
        returns the text answer rather than falling back to averaging.

        Args:
            domains: List of domain names or URLs to classify
                               e.g., ["google.com", "https://facebook.com/page"]
            archive_date: For historical analysis.
                                                    Format: "YYYYMMDD" or datetime object
            use_cache: Whether to reuse cached content (default: True)
            latest: Whether to download latest model versions (default: False)
            use_screenshots: Fuse the screenshot model in (default: False). See above for
                what it is worth.

        Returns:
            dict: ``{"results": [...], "report": {...}}``. Each result carries ``url``,
            ``domain``, ``text_path``, ``image_path``, ``date_time_collected``,
            ``model_used``, ``category``, ``confidence``, ``raw_predictions``, plus the
            ``status``/``stage``/``error_code``/``retryable`` outcome fields.

        Example:
            >>> classifier = DomainClassifier()
            >>> run = classifier.classify(["cnn.com", "bbc.com"])
            >>> first = run["results"][0]
            >>> print(f"{first['domain']}: {first['category']}")
            cnn.com: news

        """
        method = "combined" if use_screenshots else "text"
        return self._run(domains, method, archive_date, use_cache, latest)

    def classify_by_text(
        self,
        domains: list[str],
        archive_date: str | datetime | None = None,
        use_cache: bool = True,
        latest: bool = False,
    ) -> dict:
        """Classify domains using only text content analysis.

        Faster than combined analysis, good for batch processing or when
        screenshots are not needed.

        Args:
            domains: List of domain names or URLs to classify
            archive_date: For historical analysis
            use_cache: Whether to reuse cached content (default: True)
            latest: Whether to download latest model versions (default: False)

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
        return self._run(domains, "text", archive_date, use_cache, latest)

    def classify_by_images(
        self,
        domains: list[str],
        archive_date: str | datetime | None = None,
        use_cache: bool = True,
        latest: bool = False,
    ) -> dict:
        """Classify domains using only homepage screenshot analysis.

        Good for visual content classification, especially when text content
        is minimal or misleading.

        Args:
            domains: List of domain names or URLs to classify
            archive_date: For historical analysis
            use_cache: Whether to reuse cached content (default: True)
            latest: Whether to download latest model versions (default: False)

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
        return self._run(domains, "images", archive_date, use_cache, latest)

    def configure_llm(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        categories: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Configure LLM for AI-powered domain classification.

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
        """Classify domains using LLM analysis.

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


        Example:
            >>> classifier = DomainClassifier()
            >>> classifier.configure_llm("openai", "gpt-4o", api_key="sk-...")
            >>> results = classifier.classify_by_llm(["cnn.com", "amazon.com"])
            >>> print(f"{results[0]['domain']}: {results[0]['category']} - {results[0]['reason']}")
            cnn.com: news - This domain contains current events and journalism content


        Raises:
            RuntimeError: If the operation cannot be completed in the current state.
            ValueError: If an argument is invalid.
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
        """Classify domains using LLM multimodal analysis (text + screenshots).

        Args:
            domains: List of domain names to classify
            custom_instructions: Optional custom classification instructions
            use_cache: Whether to use cached content (default: True)

        Returns:
            list[dict]: Multimodal LLM classification results in JSON format


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
        """Get LLM usage statistics and cost tracking.

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
        """Collect website content for domains without performing inference.

        Separates content collection from classification, enabling:
        - Content reuse across multiple models
        - Clear data lineage and inspection
        - Reproducible analysis workflows

        Args:
            domains: List of domain names or URLs to collect content for
            archive_date: For historical analysis
            collection_id: Identifier for this collection
            use_cache: Whether to use cached content when available
            batch_size: Number of domains to process in parallel

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
        """Perform inference on previously collected content.

        Args:
            collection_data: Collection metadata from collect_content()
            method: Classification method - "text", "images", "combined", or "llm"
            output_file: Path to save JSON results
            latest: Whether to use latest model versions (default: False)

        Returns:
            list[dict]: Classification results in JSON format

        Example:
            >>> classifier = DomainClassifier()
            >>> collection = classifier.collect_content(["cnn.com"])
            >>> results = classifier.classify_from_collection(collection, method="text")
            >>> print(results[0]["category"])
            news


        Raises:
            RuntimeError: If the operation cannot be completed in the current state.
            ValueError: If an argument is invalid.
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
            return self._classify_combined(collection_data, output_file, latest)

        else:  # method == "llm"; other values rejected by the guard above
            if self._llm_classifier is None:
                raise RuntimeError("LLM not configured. Call configure_llm() first.")
            return self._llm_classifier.classify_from_data(
                collection_data, output_file, mode="multimodal"
            )

    def _classify_combined(
        self, collection_data: dict, output_file: str | None, latest: bool
    ) -> list[dict]:
        """Blend the text and screenshot models into one answer per domain.

        Both models' probabilities are calibrated by their own fitted temperature before
        being mixed, and the mixing weights come from ``fusion.json`` rather than a
        hard-coded average. That is the whole difference from what this used to do: the
        old branch returned the text label every time and averaged a calibrated,
        unnormalized text score against a raw image softmax, so the image model could not
        change an answer, only blur the number attached to it.

        When no fusion weights are published, or the screenshot model is unavailable,
        this returns text-only and says so. It does not fall back to averaging.

        Args:
            collection_data: The collection envelope.
            output_file: Optional path to write JSON results to.
            latest: Whether to re-resolve the models.

        Returns:
            list[dict]: One result row per collected domain.
        """
        from .fusion import fuse_probabilities, load_fusion_weights
        from .text import TextClassifier

        text_classifier = TextClassifier(cache_dir=self.cache_dir)
        text_rows = text_classifier.classify_from_data(collection_data, None, latest)

        try:
            from .image import ImageClassifier, resolve_image_model

            weights = load_fusion_weights(resolve_image_model(latest))
            if weights is None:
                logger.info(
                    "No fusion weights published for the screenshot model; "
                    "returning text-only. Run training/fuse.py to fit them."
                )
                return self._maybe_write(text_rows, output_file)

            image_classifier = ImageClassifier(cache_dir=self.cache_dir)
            image_rows = image_classifier.classify_from_data(
                collection_data, None, latest
            )
        except Exception as e:
            logger.warning(f"Screenshot model unavailable ({e}); returning text-only")
            return self._maybe_write(text_rows, output_file)

        image_by_domain = {r.get("domain"): r for r in image_rows}
        fused: list[dict] = []
        for row in text_rows:
            image_row = image_by_domain.get(row.get("domain"), {})
            text_probs = row.get("raw_predictions")
            image_probs = image_row.get("raw_predictions")

            if not text_probs and not image_probs:
                fused.append(row)
                continue
            try:
                blended = fuse_probabilities(text_probs, image_probs, weights)
            except ValueError as e:
                # A label-order mismatch mislabels everything while looking healthy.
                # Refuse the blend rather than ship a plausible wrong answer.
                logger.error(f"Cannot fuse {row.get('domain')}: {e}")
                fused.append(row)
                continue

            best = max(blended, key=lambda k: blended[k])
            merged = dict(row)
            merged["category"] = best
            merged["confidence"] = blended[best]
            merged["raw_predictions"] = blended
            merged["model_used"] = "combined/text_image"
            merged["modalities"] = [
                m for m, p in (("text", text_probs), ("image", image_probs)) if p
            ]
            fused.append(merged)

        return self._maybe_write(fused, output_file)

    @staticmethod
    def _maybe_write(rows: list[dict], output_file: str | None) -> list[dict]:
        """Write results to disk when a path was given.

        Args:
            rows: Result rows.
            output_file: Where to write, or ``None``.

        Returns:
            list[dict]: The rows, unchanged.
        """
        if output_file:
            import json as _json

            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as handle:
                _json.dump({"results": rows}, handle, indent=2)
        return rows

    def _parse_domain_name(self, url_or_domain: str) -> str:
        """Extract domain name from URL or domain string."""
        # Import here to avoid circular imports
        from .piedomain import Piedomain

        return Piedomain.parse_url_to_domain(url_or_domain)


def _reconcile_with_requested(
    results: list[dict], requested: list[str], collection_data: dict | None = None
) -> list[dict]:
    """Ensure every requested domain has a row, even if the pipeline lost it.

    Domains can drop out between collection and classification -- a fetch that
    fails hard, a batch path that returns fewer rows than it was given. Without
    this, such a domain is absent from both the results and the report, so
    "what is missing?" cannot be answered: the report only ever counted the rows
    it was handed.

    Args:
        results: Rows produced by classification, already annotated.
        requested: The domains the caller asked about, in input order.
        collection_data: The collection envelope, so a synthesized row can carry
            the real fetch verdict (e.g. ``bot_blocked``) instead of ``unknown``.

    Returns:
        list[dict]: One row per requested domain, in the requested order,
        with synthesized failure rows for any that never came back.
    """
    from .piedomain import Piedomain

    fetch_verdict: dict[str, tuple[str, str | None]] = {}
    for entry in (collection_data or {}).get("domains", []):
        if not entry.get("fetch_success"):
            name = entry.get("domain") or entry.get("url") or ""
            fetch_verdict[name] = (
                entry.get("error") or "no result returned by the pipeline",
                entry.get("error_code"),
            )

    by_key: dict[str, dict] = {}
    for row in results:
        for key in (row.get("domain"), row.get("url")):
            if key:
                by_key.setdefault(str(key), row)

    reconciled: list[dict] = []
    seen: set[int] = set()
    for name in requested:
        row = by_key.get(name) or by_key.get(Piedomain.parse_url_to_domain(name))
        if row is not None:
            if id(row) not in seen:
                seen.add(id(row))
                reconciled.append(row)
            continue
        domain = Piedomain.parse_url_to_domain(name)
        message, code = fetch_verdict.get(
            domain,
            fetch_verdict.get(name, ("no result returned by the pipeline", None)),
        )
        reconciled.append(
            annotate(
                {
                    "url": name,
                    "domain": domain,
                    "category": None,
                    "confidence": None,
                    "error": message,
                    "status": "failed",
                    "stage": Stage.FETCH.value,
                    "error_code": code or ErrorCode.UNKNOWN.value,
                }
            )
        )
    # Anything the pipeline returned that we did not ask for is still reported.
    for row in results:
        if id(row) not in seen:
            reconciled.append(row)
            seen.add(id(row))
    return reconciled


def _annotate_results(
    results: list[dict], collection_data: dict | None = None
) -> list[dict]:
    """Attach outcome fields to each result row.

    Failures are attributed to the stage they actually occurred in: if the
    collection step reported ``fetch_success: False`` for a domain, the row is
    marked as a ``fetch`` failure with a code derived from the fetch error,
    otherwise it is treated as an inference-stage failure.

    Args:
        results: Classification result rows.
        collection_data: The collection envelope these rows came from, used to
            recover fetch-stage errors.

    Returns:
        list[dict]: The same rows, annotated in place.
    """
    fetch_errors: dict[str, tuple[str, str | None]] = {}
    provenance: dict[str, tuple[str, str | None]] = {}
    if collection_data:
        for entry in collection_data.get("domains", []):
            key = entry.get("domain") or entry.get("url") or ""
            if not entry.get("fetch_success"):
                fetch_errors[key] = (
                    entry.get("error") or "content fetch failed",
                    entry.get("error_code"),
                )
            else:
                provenance[key] = (
                    entry.get("source") or "live",
                    entry.get("snapshot_timestamp"),
                )

    for row in results:
        key = row.get("domain") or row.get("url") or ""
        if key in provenance:
            row["source"], row["snapshot_timestamp"] = provenance[key]

        if row.get("category") is not None and not row.get("error"):
            annotate(row)
            continue

        if key in fetch_errors:
            message, code = fetch_errors[key]
            row.setdefault("error", message)
            row["stage"] = Stage.FETCH.value
            # Prefer the fetcher's own verdict (e.g. bot_blocked) over guessing
            # from the message text.
            row["error_code"] = code or classify_exception(RuntimeError(message)).value
        elif row.get("error_code"):
            # The classifier already named it (thin_content, empty_text); do not
            # re-derive a worse answer from the message string.
            row["stage"] = row.get("stage") or Stage.INFER.value
        else:
            row["stage"] = row.get("stage") or Stage.INFER.value
            message = str(row.get("error") or "")
            if "no meaningful text" in message.lower():
                row["error_code"] = ErrorCode.EMPTY_TEXT.value
            elif "not found" in message.lower():
                row["error_code"] = ErrorCode.MISSING_INPUT_PATH.value
            elif message:
                row["error_code"] = classify_exception(RuntimeError(message)).value
            else:
                row["error_code"] = ErrorCode.UNKNOWN.value
        row["status"] = "failed"
        annotate(row)
    return results


# Convenience functions for quick access
def _classify_domains_impl(
    domains: list[str],
    method: str = "combined",
    archive_date: str | datetime | None = None,
    cache_dir: str | None = None,
) -> dict:
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
) -> dict:
    """Quick domain classification function.

    Args:
        domains: List of domain names or URLs to classify
        method: Classification method - "combined", "text", or "images"
        archive_date: Optional historical date for archive.org analysis
        cache_dir: Optional cache directory override

    Returns:
        dict: ``{"results": [...], "report": {...}}``. Each result row carries
        ``status``, ``stage`` and ``error_code``; the report aggregates counts
        by reason and names the domains that produced nothing.

    Example:
        >>> run = classify_domains(["cnn.com", "github.com"])
        >>> for result in run["results"]:
        ...     print(f"{result['domain']}: {result['category']} ({result['confidence']:.3f})")
        cnn.com: news (0.876)
        github.com: computers (0.892)
        >>> run["report"]["failed"]
        0
    """
    current = globals().get("classify_domains")
    if current is not None and current is not _classify_domains_wrapper:
        return current(
            domains, method=method, archive_date=archive_date, cache_dir=cache_dir
        )

    return _classify_domains_impl(domains, method, archive_date, cache_dir)


# Store original function object for patch detection
_classify_domains_wrapper = classify_domains
