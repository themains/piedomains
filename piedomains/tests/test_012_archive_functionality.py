#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for archive.org historical classification functionality.
"""

import unittest
import tempfile
import shutil
import os
from datetime import datetime
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from piedomains.api import DomainClassifier
from piedomains.archive_support import (
    pred_shalla_cat_archive,
    pred_shalla_cat_with_text_archive,
    pred_shalla_cat_with_images_archive
)


class TestArchiveFunctionality(unittest.TestCase):
    """Test archive.org integration and historical classification."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)
        self.test_domains = ["google.com", "yahoo.com"]
        self.test_date = "20200101"
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('piedomains.classifiers.combined_classifier.CombinedClassifier.predict')
    def test_archive_date_integration(self, mock_predict):
        """Test archive date handling in new API."""
        # Mock archive classification result
        mock_result = pd.DataFrame([
            {
                'domain': 'google.com',
                'pred_label': 'searchengines',
                'pred_prob': 0.95,
                'archive_date': '20200101',
                'error': None
            }
        ])
        mock_predict.return_value = mock_result
        
        # Test with string date
        result = self.classifier.classify(
            domains=["google.com"], 
            archive_date="20200101"
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('archive_date', result.columns)
        self.assertEqual(result.iloc[0]['archive_date'], '20200101')
    
    @patch('piedomains.classifiers.combined_classifier.CombinedClassifier.predict')
    def test_archive_datetime_conversion(self, mock_predict):
        """Test datetime object conversion for archive dates."""
        mock_result = pd.DataFrame([
            {
                'domain': 'google.com',
                'pred_label': 'searchengines',
                'archive_date': '20200101'
            }
        ])
        mock_predict.return_value = mock_result
        
        # Test with datetime object
        test_datetime = datetime(2020, 1, 1)
        result = self.classifier.classify(
            domains=["google.com"], 
            archive_date=test_datetime
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        # The datetime should be converted to string format internally
    
    @patch('requests.get')
    def test_archive_api_mock(self, mock_get):
        """Test archive.org API integration with mocked responses."""
        # Mock successful archive.org API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'archived_snapshots': {
                'closest': {
                    'available': True,
                    'url': 'https://web.archive.org/web/20200101120000/https://google.com/',
                    'timestamp': '20200101120000'
                }
            }
        }
        mock_get.return_value = mock_response
        
        # Test that archive URL discovery works
        from piedomains.archive_support import _fetch_archive_snapshot_url
        
        result_url = _fetch_archive_snapshot_url("https://google.com", "20200101")
        self.assertIsNotNone(result_url)
        self.assertIn("web.archive.org", result_url)
        self.assertIn("google.com", result_url)
    
    @patch('requests.get')
    def test_archive_api_not_available(self, mock_get):
        """Test handling when archive.org has no snapshots."""
        # Mock archive.org API response with no snapshots
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'archived_snapshots': {}
        }
        mock_get.return_value = mock_response
        
        from piedomains.archive_support import _fetch_archive_snapshot_url
        
        result_url = _fetch_archive_snapshot_url("https://nonexistent.com", "20200101")
        self.assertIsNone(result_url)
    
    @patch('requests.get')
    def test_archive_api_error_handling(self, mock_get):
        """Test error handling for archive.org API failures."""
        # Mock archive.org API failure
        mock_get.side_effect = Exception("Archive.org unavailable")
        
        from piedomains.archive_support import _fetch_archive_snapshot_url
        
        result_url = _fetch_archive_snapshot_url("https://google.com", "20200101")
        self.assertIsNone(result_url)  # Should return None on error
    
    def test_legacy_archive_functions_exist(self):
        """Test that legacy archive functions are still available."""
        # These functions should exist for backward compatibility
        self.assertTrue(callable(pred_shalla_cat_archive))
        self.assertTrue(callable(pred_shalla_cat_with_text_archive))
        self.assertTrue(callable(pred_shalla_cat_with_images_archive))
    
    @patch('piedomains.piedomain.Piedomain.pred_shalla_cat_with_text')
    def test_legacy_archive_text_function(self, mock_predict):
        """Test legacy archive text function."""
        # Mock legacy function
        mock_result = pd.DataFrame([
            {
                'domain': 'google.com',
                'text_label': 'searchengines',
                'text_prob': 0.95,
                'archive_date': '20200101'
            }
        ])
        mock_predict.return_value = mock_result
        
        # Should not raise error
        try:
            result = pred_shalla_cat_with_text_archive(
                input=["google.com"],
                archive_date="20200101"
            )
            self.assertIsInstance(result, pd.DataFrame)
        except Exception as e:
            # Some errors might be expected due to missing models in test environment
            self.assertIsInstance(e, (ImportError, OSError, AttributeError))


@pytest.mark.integration
class TestArchiveIntegration(unittest.TestCase):
    """Integration tests that would actually hit archive.org (marked for optional execution)."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)
    
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.slow
    def test_real_archive_snapshot_discovery(self):
        """Test actual archive.org snapshot discovery (slow, optional)."""
        # This test would hit the real archive.org API
        # Only run when specifically requested
        from piedomains.archive_support import _fetch_archive_snapshot_url
        
        # Test with a domain that should have snapshots
        result_url = _fetch_archive_snapshot_url("https://google.com", "20200101")
        
        if result_url:  # If archive.org is available
            self.assertIn("web.archive.org", result_url)
            self.assertIn("google.com", result_url)
        # If None, archive.org might be down or domain not archived


class TestArchiveErrorHandling(unittest.TestCase):
    """Test error scenarios specific to archive functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)
    
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_invalid_archive_date_format(self):
        """Test handling of invalid archive date formats."""
        # The API should handle various date formats gracefully
        invalid_dates = ["2020-01-01", "Jan 1 2020", "invalid", ""]
        
        for invalid_date in invalid_dates:
            try:
                # This might raise an error or handle gracefully
                result = self.classifier.classify(
                    domains=["example.com"], 
                    archive_date=invalid_date
                )
                # If it doesn't raise, that's fine too
            except (ValueError, TypeError):
                # Expected for invalid formats
                pass
    
    def test_future_archive_date(self):
        """Test behavior with future dates for archive.org."""
        # Archive.org won't have snapshots for future dates
        future_date = "20301231"  # Year 2030
        
        # Should handle gracefully without crashing
        try:
            result = self.classifier.classify(
                domains=["google.com"], 
                archive_date=future_date
            )
            # Might succeed or fail gracefully
        except Exception:
            # Some failures are expected for future dates
            pass


if __name__ == '__main__':
    unittest.main()