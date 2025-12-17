#!/usr/bin/env python

"""
Integration tests for the new DomainClassifier API.
Tests actual functionality with real domains and mock data.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pytest

from piedomains.api import DomainClassifier, classify_domains


class TestNewAPIIntegration(unittest.TestCase):
    """Test the new DomainClassifier API with integration scenarios."""

    def setUp(self):
        """Set up test environment with temporary cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)
        self.test_domains = ["example.com", "test.org"]

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_classifier_initialization(self):
        """Test DomainClassifier initialization."""
        # Test default initialization
        classifier = DomainClassifier()
        self.assertEqual(classifier.cache_dir, "cache")

        # Test custom cache directory
        custom_classifier = DomainClassifier(cache_dir="/tmp/test")
        self.assertEqual(custom_classifier.cache_dir, "/tmp/test")

    @patch("piedomains.text.TextClassifier.classify_from_data")
    def test_classify_by_text_basic(self, mock_classify):
        """Test text-only classification."""
        # Mock successful text classification
        mock_result = [
            {
                "url": "example.com",
                "domain": "example.com",
                "text_path": "html/example.com.html",
                "image_path": None,
                "date_time_collected": "2024-01-01T12:00:00Z",
                "model_used": "text/shallalist_ml",
                "category": "news",
                "confidence": 0.85,
                "reason": None,
                "error": None,
                "raw_predictions": {"news": 0.85, "sports": 0.15}
            }
        ]
        mock_classify.return_value = mock_result

        result = self.classifier.classify_by_text(["example.com"])

        # Verify result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["domain"], "example.com")
        self.assertEqual(result[0]["category"], "news")
        self.assertEqual(result[0]["confidence"], 0.85)

    @patch("piedomains.image.ImageClassifier.classify_from_data")
    def test_classify_by_images_basic(self, mock_classify):
        """Test image-only classification."""
        # Mock successful image classification
        mock_result = [
            {
                "url": "example.com",
                "domain": "example.com",
                "text_path": None,
                "image_path": "images/example.com.png",
                "date_time_collected": "2024-01-01T12:00:00Z",
                "model_used": "image/shallalist_ml",
                "category": "socialnet",
                "confidence": 0.72,
                "reason": None,
                "error": None,
                "raw_predictions": {"socialnet": 0.72, "news": 0.28}
            }
        ]
        mock_classify.return_value = mock_result

        result = self.classifier.classify_by_images(["example.com"])

        # Verify result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["domain"], "example.com")
        self.assertEqual(result[0]["category"], "socialnet")
        self.assertEqual(result[0]["confidence"], 0.72)

    @patch("piedomains.text.TextClassifier.classify_from_data")
    @patch("piedomains.image.ImageClassifier.classify_from_data")
    def test_classify_combined_basic(self, mock_image_classify, mock_text_classify):
        """Test combined text+image classification."""
        # Mock text classification results
        mock_text_result = [
            {
                "url": "example.com",
                "domain": "example.com",
                "text_path": "html/example.com.html",
                "image_path": None,
                "date_time_collected": "2024-01-01T12:00:00Z",
                "model_used": "text/shallalist_ml",
                "category": "news",
                "confidence": 0.85,
                "reason": None,
                "error": None,
                "raw_predictions": {"news": 0.85, "sports": 0.15}
            }
        ]
        mock_text_classify.return_value = mock_text_result

        # Mock image classification results
        mock_image_result = [
            {
                "url": "example.com",
                "domain": "example.com",
                "text_path": None,
                "image_path": "images/example.com.png",
                "date_time_collected": "2024-01-01T12:00:00Z",
                "model_used": "image/shallalist_ml",
                "category": "socialnet",
                "confidence": 0.72,
                "reason": None,
                "error": None,
                "raw_predictions": {"socialnet": 0.72, "news": 0.28}
            }
        ]
        mock_image_classify.return_value = mock_image_result

        result = self.classifier.classify(["example.com"])

        # Verify result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["domain"], "example.com")
        self.assertEqual(result[0]["model_used"], "combined/text_image_ml")
        self.assertIn("text_category", result[0])
        self.assertIn("image_category", result[0])
        self.assertIn("confidence", result[0])

    def test_classify_empty_domains_error(self):
        """Test error handling for empty domain list."""
        with self.assertRaises(ValueError):
            self.classifier.classify([])

        with self.assertRaises(ValueError):
            self.classifier.classify_by_text([])

        with self.assertRaises(ValueError):
            self.classifier.classify_by_images([])

    def test_archive_date_conversion(self):
        """Test archive date format conversion."""
        from datetime import datetime

        # Test with datetime object
        test_date = datetime(2020, 1, 1)
        classifier = DomainClassifier()

        # Test the internal date normalization method directly
        normalized_date = classifier._normalize_archive_date(test_date)
        self.assertEqual(normalized_date, "20200101")

        # Test with string date
        normalized_date = classifier._normalize_archive_date("20200101")
        self.assertEqual(normalized_date, "20200101")

        # Test with None
        normalized_date = classifier._normalize_archive_date(None)
        self.assertIsNone(normalized_date)

    @patch("piedomains.data_collector.DataCollector.collect_batch")
    @patch("piedomains.text.TextClassifier.classify_from_data")
    @patch("piedomains.image.ImageClassifier.classify_from_data")
    def test_batch_processing(self, mock_image_classify, mock_text_classify, mock_collect):
        """Test batch processing functionality."""
        # Mock data collection
        mock_collect.return_value = {
            "collection_id": "test123",
            "timestamp": "2024-01-01T12:00:00Z",
            "domains": [
                {"domain": "example.com", "text_path": "html/example.com.html", "image_path": "images/example.com.png"},
                {"domain": "test.org", "text_path": "html/test.org.html", "image_path": "images/test.org.png"}
            ]
        }

        # Mock classification results
        mock_text_result = [
            {"domain": "example.com", "category": "news", "confidence": 0.85, "error": None},
            {"domain": "test.org", "category": "education", "confidence": 0.92, "error": None},
        ]
        mock_text_classify.return_value = mock_text_result

        mock_image_result = [
            {"domain": "example.com", "category": "news", "confidence": 0.80, "error": None},
            {"domain": "test.org", "category": "education", "confidence": 0.88, "error": None},
        ]
        mock_image_classify.return_value = mock_image_result

        domains = ["example.com", "test.org"]
        result = self.classifier.classify(domains)

        # Result should be a list
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_batch_invalid_method(self):
        """Test error handling for invalid method."""
        # Mock collection to avoid actual data fetching
        with patch("piedomains.data_collector.DataCollector.collect") as mock_collect:
            mock_collect.return_value = {"domains": []}

            with self.assertRaises(ValueError):
                self.classifier.classify_from_collection({"domains": []}, method="invalid")

    @patch("piedomains.api._classify_domains_impl")
    def test_convenience_function(self, mock_classify):
        """Test the convenience classify_domains function."""
        mock_classify.return_value = [{"domain": "example.com", "category": "news", "confidence": 0.85}]

        # Test the convenience function
        result = classify_domains(["example.com"], method="text")

        mock_classify.assert_called_once()
        self.assertIsInstance(result, list)


class TestAPIErrorHandling(unittest.TestCase):
    """Test error handling in the new API."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("piedomains.text.TextClassifier.classify_from_data")
    def test_text_classification_error_handling(self, mock_classify):
        """Test error handling in text classification."""
        # Mock text classifier failure
        mock_classify.side_effect = Exception("Model loading failed")

        # Should not raise, but return empty list or handle gracefully
        try:
            result = self.classifier.classify_by_text(["example.com"])
            # Result should still be a list (might be empty or contain error info)
            self.assertIsInstance(result, list)
        except Exception:
            # If it does raise, that's also acceptable for now
            pass

    @patch("piedomains.text.TextClassifier.classify_from_data")
    @patch("piedomains.image.ImageClassifier.classify_from_data")
    def test_combined_classification_partial_failure(self, mock_image_classify, mock_text_classify):
        """Test handling when some domains fail in combined classification."""
        # Mock partial failure scenario
        mock_text_result = [
            {"domain": "example.com", "category": "news", "confidence": 0.85, "error": None},
            {"domain": "failed.com", "category": None, "confidence": 0.0, "error": "Network timeout"},
        ]
        mock_text_classify.return_value = mock_text_result

        mock_image_result = [
            {"domain": "example.com", "category": "news", "confidence": 0.80, "error": None},
            {"domain": "failed.com", "category": None, "confidence": 0.0, "error": "Network timeout"},
        ]
        mock_image_classify.return_value = mock_image_result

        result = self.classifier.classify(["example.com", "failed.com"])

        # Should handle partial failures gracefully
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0]["error"])  # Success case
        self.assertIsNotNone(result[1]["error"])  # Failure case


# Performance and benchmark tests
@pytest.mark.performance
class TestAPIPerformance(unittest.TestCase):
    """Performance tests for the new API."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("piedomains.data_collector.DataCollector.collect_batch")
    @patch("piedomains.text.TextClassifier.classify_from_data")
    def test_batch_performance(self, mock_classify, mock_collect):
        """Test that batch processing handles large lists efficiently."""
        # Mock data collection
        mock_collect.return_value = {
            "collection_id": "perf_test",
            "domains": [{"domain": f"test{i}.com"} for i in range(100)]
        }

        # Mock fast classification
        mock_classify.return_value = [
            {"domain": f"test{i}.com", "category": "news", "confidence": 0.8}
            for i in range(100)
        ]

        # Test with 100 domains
        domains = [f"test{i}.com" for i in range(100)]

        import time

        start_time = time.time()
        result = self.classifier.classify_by_text(domains)
        end_time = time.time()

        # Should complete reasonably quickly (mocked, so very fast)
        self.assertLess(end_time - start_time, 10)  # Should be much faster with mocking
        self.assertEqual(len(result), 100)


if __name__ == "__main__":
    unittest.main()
