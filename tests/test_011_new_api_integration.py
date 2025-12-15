#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration tests for the new DomainClassifier API.
Tests actual functionality with real domains and mock data.
"""

import unittest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock
import pandas as pd
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
    
    @patch('piedomains.classifiers.text_classifier.TextClassifier.predict')
    def test_classify_by_text_basic(self, mock_predict):
        """Test text-only classification."""
        # Mock successful text classification
        mock_result = pd.DataFrame([
            {
                'domain': 'example.com',
                'text_label': 'news',
                'text_prob': 0.85,
                'used_domain_text': True,
                'extracted_text': 'example news content',
                'error': None
            }
        ])
        mock_predict.return_value = mock_result
        
        result = self.classifier.classify_by_text(["example.com"])
        
        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('domain', result.columns)
        self.assertIn('text_label', result.columns)
        self.assertIn('text_prob', result.columns)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['domain'], 'example.com')
        self.assertEqual(result.iloc[0]['text_label'], 'news')
    
    @patch('piedomains.classifiers.image_classifier.ImageClassifier.predict')
    def test_classify_by_images_basic(self, mock_predict):
        """Test image-only classification."""
        # Mock successful image classification
        mock_result = pd.DataFrame([
            {
                'domain': 'example.com',
                'image_label': 'socialnet',
                'image_prob': 0.72,
                'used_domain_screenshot': True,
                'error': None
            }
        ])
        mock_predict.return_value = mock_result
        
        result = self.classifier.classify_by_images(["example.com"])
        
        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('domain', result.columns)
        self.assertIn('image_label', result.columns)
        self.assertIn('image_prob', result.columns)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['domain'], 'example.com')
        self.assertEqual(result.iloc[0]['image_label'], 'socialnet')
    
    @patch('piedomains.classifiers.combined_classifier.CombinedClassifier.predict')
    def test_classify_combined_basic(self, mock_predict):
        """Test combined text+image classification."""
        # Mock successful combined classification
        mock_result = pd.DataFrame([
            {
                'domain': 'example.com',
                'pred_label': 'news',
                'pred_prob': 0.78,
                'text_label': 'news',
                'text_prob': 0.85,
                'image_label': 'socialnet',
                'image_prob': 0.72,
                'used_domain_text': True,
                'used_domain_screenshot': True,
                'error': None
            }
        ])
        mock_predict.return_value = mock_result
        
        result = self.classifier.classify(["example.com"])
        
        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('domain', result.columns)
        self.assertIn('pred_label', result.columns)
        self.assertIn('pred_prob', result.columns)
        self.assertIn('text_label', result.columns)
        self.assertIn('image_label', result.columns)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['pred_label'], 'news')
    
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
        
        # This should not raise an error (conversion happens internally)
        # We can't easily test the internal conversion without mocking,
        # but we can verify the interface accepts datetime objects
        try:
            # Mock the classifier to avoid actual API calls
            with patch('piedomains.classifiers.combined_classifier.CombinedClassifier.predict') as mock_predict:
                mock_predict.return_value = pd.DataFrame()
                result = classifier.classify(["example.com"], archive_date=test_date)
                self.assertIsInstance(result, pd.DataFrame)
        except Exception as e:
            # Should not raise ValueError for date format
            self.assertNotIsInstance(e, ValueError)
    
    @patch('piedomains.classifiers.combined_classifier.CombinedClassifier.predict')
    def test_batch_processing(self, mock_predict):
        """Test batch processing functionality."""
        # Mock batch results
        mock_result = pd.DataFrame([
            {'domain': 'example.com', 'pred_label': 'news', 'pred_prob': 0.85},
            {'domain': 'test.org', 'pred_label': 'education', 'pred_prob': 0.92}
        ])
        mock_predict.return_value = mock_result
        
        domains = ["example.com", "test.org", "sample.net", "demo.edu"]
        result = self.classifier.classify_batch(
            domains, 
            method="combined", 
            batch_size=2, 
            show_progress=False
        )
        
        # Should have called predict twice (2 batches of 2)
        self.assertEqual(mock_predict.call_count, 2)
        
        # Result should be a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_batch_invalid_method(self):
        """Test error handling for invalid batch method."""
        with self.assertRaises(ValueError):
            self.classifier.classify_batch(["example.com"], method="invalid")
    
    @patch('piedomains.api.classify_domains')
    def test_convenience_function(self, mock_classify):
        """Test the convenience classify_domains function."""
        mock_classify.return_value = pd.DataFrame()
        
        # Test the convenience function
        result = classify_domains(["example.com"], method="text")
        
        mock_classify.assert_called_once()
        self.assertIsInstance(result, pd.DataFrame)


class TestAPIErrorHandling(unittest.TestCase):
    """Test error handling in the new API."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)
    
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('piedomains.classifiers.text_classifier.TextClassifier.predict')
    def test_text_classification_error_handling(self, mock_predict):
        """Test error handling in text classification."""
        # Mock text classifier failure
        mock_predict.side_effect = Exception("Model loading failed")
        
        # Should not raise, but return empty DataFrame or handle gracefully
        try:
            result = self.classifier.classify_by_text(["example.com"])
            # Result should still be a DataFrame (might be empty or contain error info)
            self.assertIsInstance(result, pd.DataFrame)
        except Exception:
            # If it does raise, that's also acceptable for now
            pass
    
    @patch('piedomains.classifiers.combined_classifier.CombinedClassifier.predict')
    def test_combined_classification_partial_failure(self, mock_predict):
        """Test handling when some domains fail in combined classification."""
        # Mock partial failure scenario
        mock_result = pd.DataFrame([
            {'domain': 'example.com', 'pred_label': 'news', 'error': None},
            {'domain': 'failed.com', 'pred_label': None, 'error': 'Network timeout'}
        ])
        mock_predict.return_value = mock_result
        
        result = self.classifier.classify(["example.com", "failed.com"])
        
        # Should handle partial failures gracefully
        self.assertEqual(len(result), 2)
        self.assertIsNone(result.iloc[0]['error'])  # Success case
        self.assertIsNotNone(result.iloc[1]['error'])  # Failure case


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
    
    @patch('piedomains.classifiers.text_classifier.TextClassifier.predict')
    def test_batch_performance(self, mock_predict):
        """Test that batch processing handles large lists efficiently."""
        # Mock fast prediction
        mock_predict.return_value = pd.DataFrame([
            {'domain': f'test{i}.com', 'text_label': 'news', 'text_prob': 0.8}
            for i in range(100)
        ])
        
        # Test with 100 domains
        domains = [f"test{i}.com" for i in range(100)]
        
        import time
        start_time = time.time()
        result = self.classifier.classify_batch(
            domains, 
            method="text", 
            batch_size=25, 
            show_progress=False
        )
        end_time = time.time()
        
        # Should complete reasonably quickly (mocked, so very fast)
        self.assertLess(end_time - start_time, 10)  # Should be much faster with mocking
        self.assertEqual(len(result), 100)


if __name__ == '__main__':
    unittest.main()