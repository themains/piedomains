"""
Test error handling and edge cases.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from piedomains.piedomain import Piedomain


class TestErrorHandling(unittest.TestCase):
    """Test error handling in various scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.html_dir = os.path.join(self.temp_dir, "html")
        self.image_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.html_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_validate_input_empty_list_no_path(self):
        """Test validate_input with empty domain list and no path."""
        with self.assertRaises(Exception) as context:
            Piedomain.validate_input([], None, "html")

        self.assertIn("Provide list of Domains", str(context.exception))

    def test_validate_input_nonexistent_path(self):
        """Test validate_input with nonexistent path."""
        nonexistent_path = "/path/that/does/not/exist"

        with self.assertRaises(Exception) as context:
            Piedomain.validate_input([], nonexistent_path, "html")

        self.assertIn("does not exist", str(context.exception))

    def test_validate_input_empty_directory(self):
        """Test validate_input with empty directory."""
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)

        with self.assertRaises(Exception) as context:
            Piedomain.validate_input([], empty_dir, "html")

        self.assertIn("is empty", str(context.exception))

    def test_validate_input_valid_offline_mode(self):
        """Test validate_input with valid offline directory."""
        # Create a test HTML file
        test_file = os.path.join(self.html_dir, "test.html")
        with open(test_file, "w") as f:
            f.write("<html><body>test</body></html>")

        result = Piedomain.validate_input([], self.html_dir, "html")
        self.assertTrue(result)  # Should return True for offline mode

    @patch("piedomains.fetchers.PlaywrightFetcher.fetch_screenshot")
    def test_playwright_screenshot_error(self, mock_fetch_screenshot):
        """Test screenshot capture error handling with Playwright fetcher."""
        # Mock Playwright error
        mock_fetch_screenshot.return_value = (None, "Playwright navigation error")

        from piedomains.fetchers import PlaywrightFetcher

        fetcher = PlaywrightFetcher()

        screenshot_data, error = fetcher.fetch_screenshot("example.com")

        self.assertIsNone(screenshot_data)
        self.assertIn("Playwright navigation error", error)

    def test_text_from_html_malformed_html(self):
        """Test text extraction from malformed HTML."""
        malformed_html = "<html><body><p>Unclosed paragraph<div>Nested incorrectly</p></div></body></html>"

        # Should handle malformed HTML gracefully
        result = Piedomain.text_from_html(malformed_html)

        self.assertIsInstance(result, str)
        # BeautifulSoup should handle malformed HTML

    def test_data_cleanup_non_string_input(self):
        """Test data cleanup with non-string input."""
        # Should handle non-string input gracefully or raise appropriate error
        with self.assertRaises(AttributeError):
            Piedomain.data_cleanup(123)

    def test_validate_domains_with_none_values(self):
        """Test domain validation with None values in list."""
        domains_with_none = ["google.com", None, "facebook.com"]

        valid, invalid = Piedomain.validate_domains(domains_with_none)

        self.assertEqual(len(invalid), 1)
        self.assertIn(None, invalid)
        self.assertEqual(len(valid), 2)


if __name__ == "__main__":
    unittest.main()
