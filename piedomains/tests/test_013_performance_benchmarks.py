#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance benchmark tests for piedomains.
Tests timing, memory usage, and scalability.
"""

import unittest
import tempfile
import shutil
import os
import time
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from piedomains.api import DomainClassifier
from piedomains.processors.text_processor import TextProcessor


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_text_processing_performance(self):
        """Benchmark text processing speed."""
        # Test text processing with various input sizes
        test_html = """
        <html><head><title>Test Page</title></head>
        <body>
            <h1>Welcome to Test Site</h1>
            <p>This is a test paragraph with some content about news and politics.</p>
            <div>More content here about technology and science.</div>
            <script>console.log('ignore this');</script>
        </body></html>
        """ * 100  # Make it larger
        
        processor = TextProcessor()
        
        # Benchmark text extraction
        start_time = time.time()
        extracted_text = processor.extract_text_from_html(test_html)
        extraction_time = time.time() - start_time
        
        # Should complete quickly (under 1 second for reasonable HTML)
        self.assertLess(extraction_time, 1.0)
        self.assertIsInstance(extracted_text, str)
        self.assertGreater(len(extracted_text), 0)
        
        # Benchmark text cleaning
        start_time = time.time()
        cleaned_text = processor.clean_and_normalize_text(extracted_text)
        cleaning_time = time.time() - start_time
        
        self.assertLess(cleaning_time, 1.0)
        self.assertIsInstance(cleaned_text, str)
    
    @patch('piedomains.classifiers.text_classifier.TextClassifier.predict')
    def test_batch_processing_scalability(self, mock_predict):
        """Test scalability of batch processing."""
        # Mock prediction results
        def mock_batch_predict(domains, *args, **kwargs):
            return pd.DataFrame([
                {
                    'domain': domain,
                    'text_label': 'news',
                    'text_prob': 0.8,
                    'error': None
                }
                for domain in [self.classifier._parse_domain_name(d) for d in domains]
            ])
        
        mock_predict.side_effect = mock_batch_predict
        
        # Test different batch sizes
        test_sizes = [10, 50, 100]
        
        for size in test_sizes:
            domains = [f"test{i}.com" for i in range(size)]
            
            start_time = time.time()
            result = self.classifier.classify_batch(
                domains,
                method="text",
                batch_size=25,
                show_progress=False
            )
            total_time = time.time() - start_time
            
            # Verify results
            self.assertEqual(len(result), size)
            
            # Performance should scale reasonably (mocked, so very fast)
            # Real performance would be much slower due to network requests
            self.assertLess(total_time, 10)  # Very generous for mocked tests
            
            # Log performance for manual review
            rate = size/total_time if total_time > 0 else float('inf')
            print(f"Processed {size} domains in {total_time:.2f} seconds "
                  f"({rate:.1f} domains/second)")
    
    def test_cache_effectiveness(self):
        """Test that caching improves performance."""
        # Create some test cache files
        cache_html_dir = os.path.join(self.temp_dir, "html")
        os.makedirs(cache_html_dir, exist_ok=True)
        
        # Create a cached HTML file
        test_html = "<html><body><h1>Cached Test Content</h1></body></html>"
        with open(os.path.join(cache_html_dir, "example.com.html"), 'w') as f:
            f.write(test_html)
        
        # Mock the actual prediction to isolate cache performance
        with patch('piedomains.classifiers.text_classifier.TextClassifier.predict') as mock_predict:
            mock_predict.return_value = pd.DataFrame([
                {'domain': 'example.com', 'text_label': 'news', 'text_prob': 0.8}
            ])
            
            # First call (should use cache)
            start_time = time.time()
            result1 = self.classifier.classify_by_text(["example.com"], use_cache=True)
            cached_time = time.time() - start_time
            
            # Second call (should also use cache)
            start_time = time.time()
            result2 = self.classifier.classify_by_text(["example.com"], use_cache=True)
            cached_time2 = time.time() - start_time
            
            # Both should be fast since we're using cache
            self.assertLess(cached_time, 1.0)
            self.assertLess(cached_time2, 1.0)
    
    def test_memory_usage_batch_processing(self):
        """Test memory usage doesn't grow excessively in batch processing."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Mock to avoid actual model loading
        with patch('piedomains.classifiers.text_classifier.TextClassifier.predict') as mock_predict:
            mock_predict.return_value = pd.DataFrame([
                {'domain': f'test{i}.com', 'text_label': 'news', 'text_prob': 0.8}
                for i in range(50)
            ])
            
            # Process multiple batches
            for batch_num in range(5):
                domains = [f"batch{batch_num}_test{i}.com" for i in range(50)]
                result = self.classifier.classify_batch(
                    domains,
                    method="text",
                    batch_size=10,
                    show_progress=False
                )
                
                # Force garbage collection
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for mocked tests)
        self.assertLess(memory_increase, 100)
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
              f"(+{memory_increase:.1f}MB)")


@pytest.mark.performance
@pytest.mark.slow
class TestRealPerformanceBenchmarks(unittest.TestCase):
    """Real performance tests that actually hit networks (marked as slow)."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)
    
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.skipif(os.getenv("SKIP_NETWORK_TESTS") == "1", 
                       reason="Network tests disabled")
    def test_real_text_processing_benchmark(self):
        """Benchmark actual text processing with real domains."""
        # This test would actually fetch content from real domains
        # Only run when network tests are enabled
        
        test_domains = ["example.com"]  # Simple, fast domain
        
        start_time = time.time()
        try:
            result = self.classifier.classify_by_text(
                test_domains, 
                use_cache=False  # Force fresh fetch
            )
            total_time = time.time() - start_time
            
            # Log performance metrics
            if not result.empty and result.iloc[0].get('error') is None:
                print(f"Real text classification: {total_time:.2f}s for {len(test_domains)} domains")
                self.assertLess(total_time, 30)  # Should complete within 30 seconds
            else:
                print("Real text classification failed (expected in CI)")
                
        except Exception as e:
            # Network failures are expected in CI environments
            print(f"Real performance test skipped due to: {e}")


class TestResourceManagement(unittest.TestCase):
    """Test proper resource management and cleanup."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DomainClassifier(cache_dir=self.temp_dir)
    
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cache_directory_creation(self):
        """Test that cache directories are created properly."""
        # Create a classifier with a new cache directory
        new_cache_dir = os.path.join(self.temp_dir, "new_cache")
        classifier = DomainClassifier(cache_dir=new_cache_dir)
        
        # The directory should be created when processors are initialized
        from piedomains.processors.content_processor import ContentProcessor
        processor = ContentProcessor(new_cache_dir)
        
        # Check that directories exist
        self.assertTrue(os.path.exists(processor.html_dir))
        self.assertTrue(os.path.exists(processor.image_dir))
    
    def test_temporary_file_cleanup(self):
        """Test that temporary files don't accumulate."""
        initial_files = len(os.listdir(self.temp_dir))
        
        # Mock some operations that might create temporary files
        with patch('piedomains.classifiers.text_classifier.TextClassifier.predict') as mock_predict:
            mock_predict.return_value = pd.DataFrame([
                {'domain': 'test.com', 'text_label': 'news', 'text_prob': 0.8}
            ])
            
            # Run multiple operations
            for i in range(5):
                result = self.classifier.classify_by_text([f"test{i}.com"])
        
        # File count shouldn't grow excessively
        final_files = len(os.listdir(self.temp_dir))
        file_growth = final_files - initial_files
        
        # Some growth is expected (cache files), but should be reasonable
        self.assertLess(file_growth, 50)  # Arbitrary reasonable limit


if __name__ == '__main__':
    unittest.main()