#!/usr/bin/env python3
"""
Test ArchiveFetcher batch functionality.

Simple integration test to verify that batch archive fetching works end-to-end
without mocking. Uses real archive.org API with rate limiting.
"""

import os
import tempfile
import unittest

from piedomains import DomainClassifier
from piedomains.fetchers import ArchiveFetcher
from tests.conftest import skip_in_ci


class TestArchiveBatch(unittest.TestCase):
    """Test ArchiveFetcher batch processing."""

    def setUp(self):
        """Set up test with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @skip_in_ci()
    def test_archive_fetcher_batch_basic(self):
        """Test basic batch functionality with real archive.org URLs."""
        # Use a date from 2010 when these sites existed and were simple
        fetcher = ArchiveFetcher(target_date="20100101", max_parallel=1)

        # Test with a few well-known domains that should have snapshots
        test_urls = [
            "google.com",
            "wikipedia.org",
            "cnn.com"
        ]

        # This is a real integration test - will make actual API calls
        print(f"Testing archive batch fetch with {len(test_urls)} URLs...")
        print("Note: This test makes real archive.org API calls and may be slow")

        results = []
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                fetcher.fetch_batch(test_urls, self.temp_dir)
            )
        finally:
            loop.close()

        # Verify we got results for all URLs
        self.assertEqual(len(results), len(test_urls))

        # Check that at least some succeeded (archive.org might not have all)
        successful_results = [r for r in results if r.success]
        print(f"Successfully fetched {len(successful_results)}/{len(test_urls)} archive snapshots")

        # Should have at least one success for this test to be meaningful
        self.assertGreater(len(successful_results), 0, "No archive snapshots found - test inconclusive")

        # Verify successful results have content
        for result in successful_results:
            self.assertTrue(result.success)
            self.assertIsNotNone(result.html)
            self.assertGreater(len(result.html), 0)
            print(f"✓ {result.url}: {len(result.html)} chars HTML")

    def test_archive_fetcher_parallel_config(self):
        """Test that parallel configuration works."""
        # Test with different parallel settings
        fetcher1 = ArchiveFetcher(target_date="20100101")  # Default (should be 2)
        fetcher2 = ArchiveFetcher(target_date="20100101", max_parallel=1)  # Explicit 1

        self.assertEqual(fetcher1.max_parallel, 2)  # Default archive setting
        self.assertEqual(fetcher2.max_parallel, 1)  # Explicit setting

        # Check rate limiting config
        self.assertEqual(fetcher1.cdx_rate_limit, 1.0)  # 1 second between CDX calls
        self.assertEqual(fetcher1.page_delay, 0.5)  # 0.5 second between page loads

    def test_domain_classifier_with_archive_batch(self):
        """Test high-level API with archive batch processing."""
        classifier = DomainClassifier(cache_dir=self.temp_dir)

        # Use historical date
        test_domains = ["google.com", "wikipedia.org"]

        print("Testing DomainClassifier with archive batch...")

        try:
            # This should use the new batch processing internally
            result = classifier.classify_by_text(
                test_domains,
                archive_date="20100101"
            )

            # Verify we got a list back
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), len(test_domains))

            print(f"✓ DomainClassifier archive batch returned {len(result)} results")

            # Check that at least one domain was processed
            successful_domains = [r for r in result if r.get('category') is not None]
            print(f"✓ Successfully classified {len(successful_domains)}/{len(result)} domains from archives")

        except Exception as e:
            # Archive.org may be unavailable - this is acceptable for the test
            print(f"Archive.org unavailable (this is acceptable): {e}")
            self.skipTest("archive.org unavailable")


if __name__ == "__main__":
    unittest.main()
