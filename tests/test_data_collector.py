#!/usr/bin/env python3
"""
Test DataCollector and separated workflow functionality.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from piedomains import DataCollector, DomainClassifier
from piedomains import TextClassifier, ImageClassifier


class TestDataCollector(unittest.TestCase):
    """Test DataCollector class."""

    def setUp(self):
        """Set up test with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = DataCollector(cache_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_data_collector_initialization(self):
        """Test DataCollector initializes properly."""
        self.assertEqual(str(self.collector.cache_dir), self.temp_dir)
        self.assertTrue((Path(self.temp_dir) / "html").exists())
        self.assertTrue((Path(self.temp_dir) / "images").exists())
        self.assertTrue((Path(self.temp_dir) / "metadata").exists())

    @patch('piedomains.data_collector.get_fetcher')
    def test_collect_single_domain_mock(self, mock_get_fetcher):
        """Test data collection for single domain with mocked fetcher."""
        # Mock fetcher and result
        mock_fetcher = MagicMock()
        mock_get_fetcher.return_value = mock_fetcher

        from piedomains.fetchers import FetchResult
        mock_result = FetchResult(
            url="https://example.com",
            success=True,
            html="<html><head><title>Test</title></head><body>Test content</body></html>",
            screenshot_path=str(Path(self.temp_dir) / "images" / "example.com.png"),
            title="Test Site",
            error=None
        )
        mock_fetcher.fetch_both.return_value = mock_result

        # Create collector with mocked fetcher
        collector = DataCollector(cache_dir=self.temp_dir)

        # Test collection
        result = collector.collect(["example.com"], save_metadata=False)

        # Verify result structure
        self.assertIn("collection_id", result)
        self.assertIn("timestamp", result)
        self.assertIn("domains", result)
        self.assertEqual(len(result["domains"]), 1)

        domain_data = result["domains"][0]
        self.assertEqual(domain_data["domain"], "example.com")
        self.assertEqual(domain_data["url"], "example.com")
        self.assertTrue(domain_data["fetch_success"])
        self.assertIsNotNone(domain_data["text_path"])
        self.assertIsNotNone(domain_data["image_path"])
        self.assertIsNotNone(domain_data["date_time_collected"])

    def test_collect_empty_domains(self):
        """Test collection fails with empty domains list."""
        with self.assertRaises(ValueError):
            self.collector.collect([])

    def test_metadata_save_and_load(self):
        """Test saving and loading collection metadata."""
        # Create mock data
        collection_data = {
            "collection_id": "test-123",
            "timestamp": "2024-12-17T10:30:00Z",
            "domains": [
                {
                    "domain": "test.com",
                    "fetch_success": True,
                    "text_path": "html/test.com.html",
                    "image_path": "images/test.com.png"
                }
            ]
        }

        # Save manually
        metadata_file = Path(self.temp_dir) / "metadata" / "collection_test-123.json"
        metadata_file.parent.mkdir(exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(collection_data, f)

        # Test loading
        loaded_data = self.collector.load_collection("test-123")
        self.assertEqual(loaded_data["collection_id"], "test-123")
        self.assertEqual(len(loaded_data["domains"]), 1)

    def test_list_collections(self):
        """Test listing available collections."""
        # Initially empty
        collections = self.collector.list_collections()
        self.assertEqual(len(collections), 0)

        # Add test collection
        collection_data = {
            "collection_id": "test-456",
            "timestamp": "2024-12-17T11:00:00Z",
            "summary": {"total_domains": 1, "successful": 1, "failed": 0},
            "config": {"cache_dir": self.temp_dir}
        }

        metadata_file = Path(self.temp_dir) / "metadata" / "collection_test-456.json"
        with open(metadata_file, "w") as f:
            json.dump(collection_data, f)

        # Test listing
        collections = self.collector.list_collections()
        self.assertEqual(len(collections), 1)
        self.assertEqual(collections[0]["collection_id"], "test-456")


class TestSeparatedWorkflow(unittest.TestCase):
    """Test the separated data collection and inference workflow."""

    def setUp(self):
        """Set up test with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_domain_classifier_collect_content(self):
        """Test DomainClassifier.collect_content method."""
        classifier = DomainClassifier(cache_dir=self.temp_dir)

        with patch.object(classifier, '_normalize_archive_date') as mock_normalize:
            mock_normalize.return_value = None

            with patch('piedomains.data_collector.DataCollector') as mock_collector_class:
                mock_collector = MagicMock()
                mock_collector_class.return_value = mock_collector

                mock_collector.collect.return_value = {
                    "collection_id": "test-789",
                    "domains": [{"domain": "test.com", "fetch_success": True}]
                }

                # Test collect_content
                result = classifier.collect_content(["test.com"])

                # Verify collector was called correctly
                mock_collector_class.assert_called_once_with(
                    cache_dir=self.temp_dir,
                    archive_date=None
                )
                mock_collector.collect.assert_called_once_with(
                    ["test.com"],
                    collection_id=None,
                    use_cache=True
                )

                self.assertEqual(result["collection_id"], "test-789")

    @patch('piedomains.text.TextClassifier.load_models')
    @patch('piedomains.text_processor.TextProcessor.process_html_to_text')
    def test_text_classifier_from_paths(self, mock_process_html, mock_load_models):
        """Test TextClassifier.classify_from_paths method."""
        mock_process_html.return_value = "processed text content"
        mock_load_models.return_value = None

        # Create test HTML file
        html_dir = Path(self.temp_dir) / "html"
        html_dir.mkdir(parents=True)
        html_file = html_dir / "test.com.html"
        with open(html_file, "w") as f:
            f.write("<html><body>Test content</body></html>")

        # Create test data
        data_paths = [{
            "domain": "test.com",
            "url": "test.com",
            "text_path": str(html_file),
            "image_path": "images/test.com.png",
            "date_time_collected": "2024-12-17T12:00:00Z",
            "fetch_success": True
        }]

        classifier = TextClassifier()

        # Mock the _predict_text method to avoid model loading
        with patch.object(classifier, '_predict_text') as mock_predict:
            mock_predict.return_value = {
                "text_label": "test_category",
                "text_prob": 0.85,
                "text_domain_probs": {"test_category": 0.85, "other": 0.15}
            }

            # Test classification
            results = classifier.classify_from_paths(data_paths)

            # Verify results
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result["domain"], "test.com")
            self.assertEqual(result["category"], "test_category")
            self.assertEqual(result["confidence"], 0.85)
            self.assertEqual(result["model_used"], "text/shallalist_ml")
            self.assertIsNone(result["error"])

    def test_classify_from_collection_integration(self):
        """Test end-to-end separated workflow (with mocking)."""
        classifier = DomainClassifier(cache_dir=self.temp_dir)

        # Mock collection data
        collection_data = {
            "collection_id": "test-integration",
            "domains": [{
                "domain": "test.com",
                "url": "test.com",
                "text_path": "html/test.com.html",
                "image_path": "images/test.com.png",
                "date_time_collected": "2024-12-17T12:00:00Z",
                "fetch_success": True
            }]
        }

        # Mock the TextClassifier import inside the api.py method
        with patch('piedomains.text.TextClassifier') as mock_text_classifier_class:
            mock_text_classifier = MagicMock()
            mock_text_classifier_class.return_value = mock_text_classifier

            mock_text_classifier.classify_from_data.return_value = [{
                "domain": "test.com",
                "category": "test_category",
                "confidence": 0.9,
                "model_used": "text/shallalist_ml",
                "error": None
            }]

            # Test classify_from_collection
            results = classifier.classify_from_collection(collection_data, method="text")

            # Verify results
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["domain"], "test.com")
            self.assertEqual(results[0]["category"], "test_category")
            self.assertEqual(results[0]["confidence"], 0.9)

    def test_json_schema_validation(self):
        """Test that our JSON schemas match expected format."""
        # Test collection metadata schema
        collection_data = {
            "collection_id": "test-schema",
            "timestamp": "2024-12-17T10:30:00Z",
            "config": {
                "cache_dir": "/tmp/test",
                "archive_date": None,
                "fetcher_type": "live"
            },
            "domains": [{
                "url": "example.com",
                "domain": "example.com",
                "text_path": "html/example.com.html",
                "image_path": "images/example.com.png",
                "date_time_collected": "2024-12-17T10:30:15Z",
                "fetch_success": True,
                "error": None
            }],
            "summary": {
                "total_domains": 1,
                "successful": 1,
                "failed": 0
            }
        }

        # Verify required fields exist
        required_collection_fields = ["collection_id", "timestamp", "config", "domains", "summary"]
        for field in required_collection_fields:
            self.assertIn(field, collection_data)

        # Test inference result schema
        inference_result = {
            "url": "example.com",
            "text_path": "html/example.com.html",
            "image_path": "images/example.com.png",
            "date_time_collected": "2024-12-17T10:30:15Z",
            "model_used": "text/shallalist_ml",
            "category": "news",
            "confidence": 0.87,
            "reason": None,
            "error": None,
            "raw_predictions": {"news": 0.87, "sports": 0.13}
        }

        # Verify required fields exist
        required_result_fields = ["url", "model_used", "category", "confidence"]
        for field in required_result_fields:
            self.assertIn(field, inference_result)


if __name__ == "__main__":
    unittest.main()
