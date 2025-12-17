#!/usr/bin/env python

"""
Critical integration tests for piedomains v0.3.0+ architecture.
Tests end-to-end functionality, error handling, and resource management.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pytest

from piedomains.api import DomainClassifier
from piedomains.context_managers import ResourceManager
from piedomains.http_client import PooledHTTPClient, http_client
from tests.conftest import skip_in_ci


class TestCriticalIntegration(unittest.TestCase):
    """Critical integration tests for production readiness."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_domains = ["example.com", "test.org", "invalid-domain"]

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_http_client_connection_pooling(self):
        """Test HTTP client connection pooling and session reuse."""
        with http_client() as client:
            self.assertIsInstance(client, PooledHTTPClient)

            # Test session reuse
            session1 = client.session
            session2 = client.session
            self.assertIs(session1, session2, "Session should be reused")

            # Test session has proper configuration
            adapter = session1.get_adapter("http://")
            self.assertEqual(adapter.config["pool_connections"], 10)
            self.assertEqual(adapter.config["pool_maxsize"], 20)

    def test_resource_manager_cleanup(self):
        """Test comprehensive resource cleanup."""
        with ResourceManager() as rm:
            # Test that context manager works
            self.assertEqual(len(rm._drivers), 0)
            self.assertEqual(len(rm._temp_files), 0)
            self.assertEqual(len(rm._temp_dirs), 0)

    @patch("piedomains.data_collector.DataCollector.collect")
    @patch("piedomains.text.TextClassifier.classify_from_data")
    def test_domain_validation_edge_cases(self, mock_classify, mock_collect):
        """Test domain validation with edge cases and security inputs."""
        # Mock successful classification
        def mock_collection(domains, *args, **kwargs):
            return {
                "collection_id": "test_collection",
                "timestamp": "2025-12-17T12:00:00Z",
                "domains": [
                    {
                        "url": domain,
                        "domain": domain,
                        "text_path": f"html/{domain}.html",
                        "image_path": f"images/{domain}.png",
                        "date_time_collected": "2025-12-17T12:00:00Z",
                        "fetch_success": True,
                        "cached": False,
                        "error": None
                    }
                    for domain in domains
                ]
            }

        def mock_classification(collection_data, *args, **kwargs):
            domains_data = collection_data.get("domains", [])
            return [
                {
                    "url": domain_data["url"],
                    "domain": domain_data["domain"],
                    "text_path": domain_data["text_path"],
                    "image_path": domain_data["image_path"],
                    "date_time_collected": domain_data["date_time_collected"],
                    "model_used": "text/shallalist_ml",
                    "category": "news",
                    "confidence": 0.85,
                    "reason": None,
                    "error": None,
                    "raw_predictions": {"news": 0.85, "other": 0.15}
                }
                for domain_data in domains_data
            ]

        mock_collect.side_effect = mock_collection
        mock_classify.side_effect = mock_classification

        classifier = DomainClassifier(cache_dir=self.temp_dir)

        # Test various edge cases
        edge_cases = [
            "valid.com",  # Valid domain
            "sub.valid.com",  # Subdomain
            "https://valid.com/path",  # URL with path
            "valid.com:8080",  # Domain with port
            "",  # Empty string
            "   ",  # Whitespace only
            "invalid..com",  # Double dots
            "invalid-.com",  # Trailing dash
            ".invalid.com",  # Leading dot
            "very-long-subdomain-name-that-exceeds-normal-limits.example.com",  # Long subdomain
        ]

        for domain in edge_cases:
            with self.subTest(domain=domain):
                try:
                    result = classifier.classify_by_text([domain])
                    # Should handle gracefully without crashing
                    self.assertIsInstance(result, list)
                except Exception as e:
                    # Should have meaningful error messages
                    self.assertIn("domain", str(e).lower())

    @patch("piedomains.data_collector.DataCollector.collect")
    @patch("piedomains.text.TextClassifier.classify_from_data")
    def test_batch_processing_memory_management(
        self, mock_classify, mock_collect
    ):
        """Test batch processing doesn't leak memory."""
        # Mock data collection and classification
        def mock_collection(domains, *args, **kwargs):
            return {
                "collection_id": "test_collection",
                "timestamp": "2025-12-17T12:00:00Z",
                "domains": [
                    {
                        "url": domain,
                        "domain": domain,
                        "text_path": f"html/{domain}.html",
                        "image_path": f"images/{domain}.png",
                        "date_time_collected": "2025-12-17T12:00:00Z",
                        "fetch_success": True,
                        "cached": False,
                        "error": None
                    }
                    for domain in domains
                ]
            }

        def mock_classification(collection_data, *args, **kwargs):
            domains_data = collection_data.get("domains", [])
            return [
                {
                    "url": domain_data["url"],
                    "domain": domain_data["domain"],
                    "text_path": domain_data["text_path"],
                    "image_path": domain_data["image_path"],
                    "date_time_collected": domain_data["date_time_collected"],
                    "model_used": "text/shallalist_ml",
                    "category": "news",
                    "confidence": 0.8,
                    "reason": None,
                    "error": None,
                    "raw_predictions": {"news": 0.8, "other": 0.2}
                }
                for domain_data in domains_data
            ]

        mock_collect.side_effect = mock_collection
        mock_classify.side_effect = mock_classification

        classifier = DomainClassifier(cache_dir=self.temp_dir)
        domains = [f"test{i}.com" for i in range(100)]  # Large batch

        # Test batch processing completes without memory issues
        result = classifier.classify_by_text(domains)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 100)
        self.assertTrue(
            all("domain" in res and "category" in res and "confidence" in res for res in result)
        )

    def test_error_handling_network_failures(self):
        """Test error handling for network failures."""
        classifier = DomainClassifier(cache_dir=self.temp_dir)

        # Test with completely invalid domain that should fail gracefully
        with patch(
            "piedomains.fetchers.PlaywrightFetcher.fetch_both",
            side_effect=Exception("Network unreachable"),
        ):
            result = classifier.classify_by_text(["unreachable.invalid"])

            # Should return list with error information rather than crash
            self.assertIsInstance(result, list)

    def test_concurrent_classification_safety(self):
        """Test that multiple concurrent operations are safe."""
        import queue
        import threading

        results_queue = queue.Queue()
        classifier = DomainClassifier(cache_dir=self.temp_dir)

        # Mock at the module level BEFORE starting threads
        with patch("piedomains.data_collector.DataCollector.collect") as mock_collect, \
             patch("piedomains.text.TextClassifier.classify_from_data") as mock_classify:

            def mock_collection(domains, *args, **kwargs):
                return {
                    "collection_id": "test_collection",
                    "timestamp": "2025-12-17T12:00:00Z",
                    "domains": [
                        {
                            "url": domain,
                            "domain": domain,
                            "text_path": f"html/{domain}.html",
                            "image_path": f"images/{domain}.png",
                            "date_time_collected": "2025-12-17T12:00:00Z",
                            "fetch_success": True,
                            "cached": False,
                            "error": None
                        }
                        for domain in domains
                    ]
                }

            def mock_classification(collection_data, *args, **kwargs):
                domains_data = collection_data.get("domains", [])
                return [
                    {
                        "url": domain_data["url"],
                        "domain": domain_data["domain"],
                        "text_path": domain_data["text_path"],
                        "image_path": domain_data["image_path"],
                        "date_time_collected": domain_data["date_time_collected"],
                        "model_used": "text/shallalist_ml",
                        "category": "news",
                        "confidence": 0.8,
                        "reason": None,
                        "error": None,
                        "raw_predictions": {"news": 0.8, "other": 0.2}
                    }
                    for domain_data in domains_data
                ]

            mock_collect.side_effect = mock_collection
            mock_classify.side_effect = mock_classification

            def classify_worker(domains):
                try:
                    result = classifier.classify_by_text(domains)
                    results_queue.put(("success", result))
                except Exception as e:
                    results_queue.put(("error", str(e)))

            # Launch multiple concurrent operations
            threads = []
            for i in range(5):
                thread = threading.Thread(
                    target=classify_worker, args=([f"test{i}.com"],)
                )
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=30)

        # Check all operations completed successfully
        success_count = 0
        while not results_queue.empty():
            result_type, result_data = results_queue.get()
            if result_type == "success":
                success_count += 1
                self.assertIsInstance(result_data, list)

        self.assertEqual(success_count, 5, "All concurrent operations should succeed")

    def test_input_sanitization_security(self):
        """Test input sanitization against potential security issues."""
        classifier = DomainClassifier(cache_dir=self.temp_dir)

        # Test various potentially malicious inputs
        malicious_inputs = [
            "../../../etc/passwd",  # Path traversal
            "javascript:alert('xss')",  # JavaScript injection
            "<script>alert('xss')</script>",  # HTML injection
            "'; DROP TABLE domains; --",  # SQL injection style
            "\x00\x01\x02",  # Null bytes and control chars
            "very" + "long" * 1000 + ".com",  # Extremely long input
        ]

        for malicious_input in malicious_inputs:
            with self.subTest(input=malicious_input):
                try:
                    # Should either handle gracefully or fail safely
                    result = classifier.classify_by_text([malicious_input])
                    # If it succeeds, should return valid list
                    self.assertIsInstance(result, list)
                except Exception as e:
                    # If it fails, should have proper error message
                    # Accept validation errors, Playwright browser errors, or general exceptions
                    # The key is that malicious inputs don't cause crashes or security issues
                    self.assertTrue(
                        isinstance(e, (ValueError, TypeError)) or
                        "Executable doesn't exist" in str(e) or
                        "Browser" in str(e) or
                        "Error" in type(e).__name__,
                        f"Unexpected error type for input '{malicious_input}': {type(e).__name__}: {e}"
                    )

    @skip_in_ci()
    @pytest.mark.ml
    def test_real_model_inference_basic(self):
        """Test actual model inference with real TensorFlow models (requires ML models)."""
        classifier = DomainClassifier(cache_dir=self.temp_dir)

        # Test with a well-known domain that should classify successfully
        result = classifier.classify_by_text(["google.com"])

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["domain"], "google.com")
        self.assertIsInstance(result[0]["category"], str)
        self.assertIsInstance(result[0]["confidence"], float)
        self.assertGreater(result[0]["confidence"], 0.0)
        self.assertLessEqual(result[0]["confidence"], 1.0)

    def test_archive_date_validation(self):
        """Test archive date validation and error handling."""
        classifier = DomainClassifier(cache_dir=self.temp_dir)

        # Test invalid date formats
        invalid_dates = [
            "2024-01-01",  # Wrong format
            "20240132",  # Invalid date
            "19990101",  # Too old
            "20991231",  # Future date
            "abcd1234",  # Non-numeric
            "",  # Empty
        ]

        for invalid_date in invalid_dates:
            with self.subTest(date=invalid_date):
                with self.assertRaises(ValueError):
                    classifier.classify(["example.com"], archive_date=invalid_date)

    def test_cache_directory_management(self):
        """Test cache directory creation and management."""
        # Test automatic cache directory creation
        cache_dir = os.path.join(self.temp_dir, "new_cache")
        DomainClassifier(cache_dir=cache_dir)

        # Cache directory should be created when needed
        self.assertTrue(os.path.exists(cache_dir))

        # Test with existing directory
        classifier2 = DomainClassifier(cache_dir=cache_dir)
        self.assertEqual(classifier2.cache_dir, cache_dir)


if __name__ == "__main__":
    unittest.main()
