#!/usr/bin/env python3
"""Data collector for piedomains - separates data collection from inference.

This module provides clean separation between data fetching and classification,
enabling reusability, transparency, and reproducibility.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import get_config
from .fetchers import get_fetcher
from .piedomains_logging import get_logger

logger = get_logger()


class DataCollector:
    """Pure data collection for domain content analysis.

    Separates data fetching from inference, enabling:
    - Data reuse across multiple models
    - Clear data lineage and inspection
    - Reproducible analysis workflows

    Example:
        >>> collector = DataCollector(cache_dir="data")
        >>> data = collector.collect(["cnn.com", "bbc.com"])
        >>> # data is now JSON with file paths for downstream inference
    """

    def __init__(
        self,
        cache_dir: str = "data",
        archive_date: str | None = None,
        max_parallel: int | None = None,
    ):
        """Initialize data collector.

        Args:
            cache_dir: Directory for storing collected data
            archive_date: Date for archive.org snapshots (YYYYMMDD format)
            max_parallel: Maximum parallel fetching operations
        """
        self.config = get_config()
        self.cache_dir = Path(cache_dir)
        self.archive_date = archive_date
        self.max_parallel = max_parallel or self.config.get("max_parallel", 4)

        # Create organized directory structure
        self.html_dir = self.cache_dir / "html"
        self.images_dir = self.cache_dir / "images"
        self.metadata_dir = self.cache_dir / "metadata"

        for dir_path in [self.html_dir, self.images_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize fetcher
        self.fetcher = get_fetcher(
            archive_date=self.archive_date, max_parallel=self.max_parallel
        )

        logger.info(f"Initialized DataCollector with cache_dir: {self.cache_dir}")

    def collect(
        self,
        domains: list[str],
        collection_id: str | None = None,
        use_cache: bool = True,
        save_metadata: bool = True,
    ) -> dict:
        """Collect data for domains and return structured metadata.

        Args:
            domains: List of domain names or URLs to collect data for
            collection_id: Optional identifier for this collection (auto-generated if None)
            use_cache: Whether to use cached data when available
            save_metadata: Whether to save collection metadata to file

        Returns:
            Dictionary with collection metadata and file paths

        Example:
            >>> collector = DataCollector()
            >>> data = collector.collect(["cnn.com", "bbc.com"])
            >>> print(data["domains"][0]["text_path"])
            html/cnn.com.html


        Raises:
            ValueError: If an argument is invalid.
        """
        if not domains:
            raise ValueError("domains list cannot be empty")

        collection_id = collection_id or str(uuid.uuid4())
        collection_start = datetime.now(UTC)

        logger.info(f"Starting data collection for {len(domains)} domains")

        # Collect data from all domains
        collected_domains = []

        for domain in domains:
            domain_data = self._collect_single_domain(domain, use_cache)
            collected_domains.append(domain_data)

        # Create collection metadata
        metadata = {
            "collection_id": collection_id,
            "timestamp": collection_start.isoformat() + "Z",
            "config": {
                "cache_dir": str(self.cache_dir),
                "archive_date": self.archive_date,
                "fetcher_type": "archive" if self.archive_date else "live",
                "max_parallel": self.max_parallel,
            },
            "domains": collected_domains,
            "summary": {
                "total_domains": len(domains),
                "successful": len([d for d in collected_domains if d["fetch_success"]]),
                "failed": len([d for d in collected_domains if not d["fetch_success"]]),
            },
        }

        # Save metadata if requested
        if save_metadata:
            metadata_file = self.metadata_dir / f"collection_{collection_id}.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved collection metadata to {metadata_file}")

        logger.info(f"Data collection complete: {metadata['summary']}")
        return metadata

    def _cache_stem(self, domain_name: str) -> str:
        """Build the cache filename stem for a domain.

        The archive date is part of the key: without it a live fetch and a 2010
        snapshot of the same domain overwrite each other silently.

        Args:
            domain_name: Parsed domain name.

        Returns:
            str: Filename stem, unique per (domain, archive_date).
        """
        if self.archive_date:
            return f"{domain_name}@{self.archive_date}"
        return domain_name

    def _collect_single_domain(self, domain: str, use_cache: bool) -> dict:
        """Collect data for a single domain.

        Args:
            domain: Domain name or URL
            use_cache: Whether to use cached data

        Returns:
            Dictionary with domain collection results
        """
        domain_name = self._parse_domain_name(domain)
        collection_time = datetime.now(UTC)

        # Define file paths
        stem = self._cache_stem(domain_name)
        html_file = self.html_dir / f"{stem}.html"
        image_file = self.images_dir / f"{stem}.png"

        # Check cache first if enabled
        if use_cache and html_file.exists() and image_file.exists():
            logger.info(f"Using cached data for {domain_name}")
            return {
                "url": domain,
                "domain": domain_name,
                "text_path": str(html_file.relative_to(self.cache_dir)),
                "image_path": str(image_file.relative_to(self.cache_dir)),
                "date_time_collected": collection_time.isoformat() + "Z",
                "fetch_success": True,
                "cached": True,
                "error": None,
            }

        # Fetch new data
        try:
            # Ensure domain has protocol for fetcher
            if not domain.startswith(("http://", "https://")):
                fetch_url = f"https://{domain}"
            else:
                fetch_url = domain

            result = self.fetcher.fetch_both(fetch_url, str(image_file))

            if result.success:
                # Save HTML content
                if result.html:
                    with open(html_file, "w", encoding="utf-8") as f:
                        f.write(result.html)

                return {
                    "url": domain,
                    "domain": domain_name,
                    # Same rule as the screenshot below: advertise the HTML only
                    # if it was written. A page that renders but yields almost no
                    # text succeeds with a screenshot and no HTML, and pointing
                    # at the file anyway would report the text failure as a
                    # missing path rather than as thin content.
                    "text_path": (
                        str(html_file.relative_to(self.cache_dir))
                        if result.html
                        else None
                    ),
                    # Only advertise a screenshot the fetcher actually captured.
                    # `screenshot_path` is empty when the capture failed, and
                    # reporting the intended path regardless makes downstream
                    # image classification fail on a missing file.
                    "image_path": (
                        str(image_file.relative_to(self.cache_dir))
                        if result.screenshot_path
                        else None
                    ),
                    "snapshot_timestamp": getattr(result, "snapshot_timestamp", "")
                    or None,
                    # "live" or "archive" — a bot-walled page recovered from
                    # archive.org is not the site as it stands today, and the
                    # caller has to be able to see that.
                    "source": getattr(result, "source", "live"),
                    "date_time_collected": collection_time.isoformat() + "Z",
                    "fetch_success": True,
                    "cached": False,
                    # A fetch can succeed for one modality and not the other, so
                    # carry the reason through rather than reporting None. It is
                    # what tells a caller their text row failed because the page
                    # rendered 11 words, not because a file went missing.
                    "error": result.error or None,
                    "error_code": result.error_code or None,
                    "title": result.title or None,
                    "meta_description": result.meta_description or None,
                }
            else:
                logger.warning(f"Failed to fetch data for {domain}: {result.error}")
                return {
                    "url": domain,
                    "domain": domain_name,
                    "text_path": None,
                    "image_path": None,
                    "date_time_collected": collection_time.isoformat() + "Z",
                    "fetch_success": False,
                    "cached": False,
                    "error": result.error,
                    # Carry the fetcher's own classification through. Without
                    # this a detected bot wall arrives as `unknown`, which
                    # defeats the point of detecting it.
                    "error_code": result.error_code or None,
                }

        except Exception as e:
            logger.error(f"Exception during data collection for {domain}: {e}")
            return {
                "url": domain,
                "domain": domain_name,
                "text_path": None,
                "image_path": None,
                "date_time_collected": collection_time.isoformat() + "Z",
                "fetch_success": False,
                "cached": False,
                "error": str(e),
            }

    def collect_batch(
        self,
        domains: list[str],
        collection_id: str | None = None,
        use_cache: bool = True,
        save_metadata: bool = True,
        batch_size: int = 10,
    ) -> dict:
        """Collect data for large batches of domains with optimized parallel processing.

        Args:
            domains: List of domain names or URLs
            collection_id: Optional identifier for this collection
            use_cache: Whether to use cached data when available
            save_metadata: Whether to save collection metadata to file
            batch_size: Number of domains to process in parallel

        Returns:
            Dictionary with collection metadata and file paths


        Raises:
            ValueError: If an argument is invalid.
        """
        if not domains:
            raise ValueError("domains list cannot be empty")

        collection_id = collection_id or str(uuid.uuid4())
        collection_start = datetime.now(UTC)

        logger.info(f"Starting batch data collection for {len(domains)} domains")

        # Process domains in batches for better resource management
        all_results = []

        for i in range(0, len(domains), batch_size):
            batch = domains[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(domains) - 1) // batch_size + 1}"
            )

            try:
                batch_results = self._collect_batch_parallel(batch, use_cache)
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch {i // batch_size + 1} failed: {e}")
                # Create error entries for failed batch
                for domain in batch:
                    domain_name = self._parse_domain_name(domain)
                    all_results.append(
                        {
                            "url": domain,
                            "domain": domain_name,
                            "text_path": None,
                            "image_path": None,
                            "date_time_collected": datetime.now(UTC).isoformat(),
                            "fetch_success": False,
                            "cached": False,
                            "error": f"Batch processing failed: {e}",
                        }
                    )

        # Create collection metadata
        metadata = {
            "collection_id": collection_id,
            "timestamp": collection_start.isoformat() + "Z",
            "config": {
                "cache_dir": str(self.cache_dir),
                "archive_date": self.archive_date,
                "fetcher_type": "archive" if self.archive_date else "live",
                "max_parallel": self.max_parallel,
                "batch_size": batch_size,
            },
            "domains": all_results,
            "summary": {
                "total_domains": len(domains),
                "successful": len([d for d in all_results if d["fetch_success"]]),
                "failed": len([d for d in all_results if not d["fetch_success"]]),
                "cached": len([d for d in all_results if d.get("cached", False)]),
            },
        }

        # Save metadata if requested
        if save_metadata:
            metadata_file = self.metadata_dir / f"collection_{collection_id}.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved collection metadata to {metadata_file}")

        logger.info(f"Batch data collection complete: {metadata['summary']}")
        return metadata

    def _collect_batch_parallel(
        self, domains: list[str], use_cache: bool
    ) -> list[dict]:
        """Collect data for a batch of domains using parallel fetching.

        Args:
            domains: List of domain names or URLs
            use_cache: Whether to use cached data

        Returns:
            List of domain collection results
        """
        import asyncio

        # Separate domains that need fetching vs. cached
        to_fetch = []
        cached_results = []

        for domain in domains:
            domain_name = self._parse_domain_name(domain)
            stem = self._cache_stem(domain_name)
            html_file = self.html_dir / f"{stem}.html"
            image_file = self.images_dir / f"{stem}.png"

            if use_cache and html_file.exists() and image_file.exists():
                cached_results.append(
                    {
                        "url": domain,
                        "domain": domain_name,
                        "text_path": str(html_file.relative_to(self.cache_dir)),
                        "image_path": str(image_file.relative_to(self.cache_dir)),
                        "date_time_collected": datetime.now(UTC).isoformat(),
                        "fetch_success": True,
                        "cached": True,
                        "error": None,
                    }
                )
            else:
                to_fetch.append(domain)

        # Parallel fetch for uncached domains
        if to_fetch:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                fetch_results = loop.run_until_complete(
                    self.fetcher.fetch_batch(to_fetch, str(self.cache_dir))
                )

                # Process fetch results
                fetched_results = []
                for result in fetch_results:
                    domain_name = self._parse_domain_name(result.url)
                    stem = self._cache_stem(domain_name)
                    collection_time = datetime.now(UTC)

                    if result.success:
                        # Save HTML content
                        html_file = self.html_dir / f"{stem}.html"
                        if result.html:
                            with open(html_file, "w", encoding="utf-8") as f:
                                f.write(result.html)

                        fetched_results.append(
                            {
                                "url": result.url,
                                "domain": domain_name,
                                # Advertise each modality only if it was written;
                                # see the single-domain path for why.
                                "text_path": str(html_file.relative_to(self.cache_dir))
                                if result.html
                                else None,
                                "image_path": str(
                                    Path(result.screenshot_path).relative_to(
                                        self.cache_dir
                                    )
                                )
                                if result.screenshot_path
                                else None,
                                "snapshot_timestamp": result.snapshot_timestamp or None,
                                "source": result.source,
                                "date_time_collected": collection_time.isoformat()
                                + "Z",
                                "fetch_success": True,
                                "cached": False,
                                "error": result.error or None,
                                "error_code": result.error_code or None,
                                "title": result.title or None,
                                "meta_description": result.meta_description or None,
                            }
                        )
                    else:
                        fetched_results.append(
                            {
                                "url": result.url,
                                "domain": domain_name,
                                "text_path": None,
                                "image_path": None,
                                "date_time_collected": collection_time.isoformat()
                                + "Z",
                                "fetch_success": False,
                                "cached": False,
                                "error": result.error,
                                # Same as the single-domain path: without this a
                                # detected bot wall arrives as `unknown`.
                                "error_code": result.error_code or None,
                            }
                        )

                return cached_results + fetched_results

            finally:
                loop.close()
        else:
            return cached_results

    def load_collection(self, collection_id: str) -> dict:
        """Load previously saved collection metadata.

        Args:
            collection_id: ID of the collection to load

        Returns:
            Collection metadata dictionary

        Raises:
            FileNotFoundError: If collection file doesn't exist
        """
        metadata_file = self.metadata_dir / f"collection_{collection_id}.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Collection {collection_id} not found")

        with open(metadata_file, encoding="utf-8") as f:
            return json.load(f)

    def list_collections(self) -> list[dict]:
        """List all available collections.

        Returns:
            List of collection summaries
        """
        collections = []

        for metadata_file in self.metadata_dir.glob("collection_*.json"):
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    metadata = json.load(f)

                collections.append(
                    {
                        "collection_id": metadata["collection_id"],
                        "timestamp": metadata["timestamp"],
                        "total_domains": metadata["summary"]["total_domains"],
                        "successful": metadata["summary"]["successful"],
                        "failed": metadata["summary"]["failed"],
                        "config": metadata["config"],
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load collection {metadata_file}: {e}")

        return sorted(collections, key=lambda x: x["timestamp"], reverse=True)

    def _parse_domain_name(self, url_or_domain: str) -> str:
        """Extract clean domain name from URL or domain string."""
        from .piedomain import Piedomain

        return Piedomain.parse_url_to_domain(url_or_domain)
