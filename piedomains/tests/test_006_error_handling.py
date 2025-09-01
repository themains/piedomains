"""
Test error handling and edge cases.
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import shutil
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

    @patch('requests.get')
    def test_extract_htmls_network_error(self, mock_get):
        """Test HTML extraction with network errors."""
        mock_get.side_effect = Exception("Network error")

        domains = ["example.com"]
        errors = Piedomain.extract_htmls(domains, False, self.html_dir)

        self.assertIn("example.com", errors)
        self.assertIn("Network error", str(errors["example.com"]))

    @patch('requests.get')
    def test_extract_htmls_http_error(self, mock_get):
        """Test HTML extraction with HTTP errors."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response

        domains = ["nonexistent.com"]
        errors = Piedomain.extract_htmls(domains, False, self.html_dir)

        self.assertIn("nonexistent.com", errors)

    @patch('piedomains.piedomain.Piedomain.get_driver')
    def test_save_image_webdriver_error(self, mock_get_driver):
        """Test screenshot capture with WebDriver errors."""
        mock_driver = MagicMock()
        mock_driver.get.side_effect = Exception("WebDriver error")
        mock_get_driver.return_value = mock_driver

        success, error_msg = Piedomain.save_image("example.com", self.image_dir)

        self.assertFalse(success)
        self.assertIn("WebDriver error", error_msg)
        mock_driver.quit.assert_called_once()

    @patch('piedomains.piedomain.Piedomain.get_driver')
    def test_save_image_driver_quit_error(self, mock_get_driver):
        """Test screenshot capture with driver quit error."""
        mock_driver = MagicMock()
        mock_driver.get.side_effect = Exception("WebDriver error")
        mock_driver.quit.side_effect = Exception("Quit error")
        mock_get_driver.return_value = mock_driver

        # Should handle quit error gracefully
        success, error_msg = Piedomain.save_image("example.com", self.image_dir)

        self.assertFalse(success)
        self.assertIn("WebDriver error", error_msg)

    def test_extract_image_tensor_invalid_directory(self):
        """Test image tensor extraction with invalid directory."""
        nonexistent_dir = "/path/that/does/not/exist"

        with self.assertRaises(FileNotFoundError):
            Piedomain.extract_image_tensor(True, ["example.com"], nonexistent_dir)

    def test_extract_html_text_invalid_directory(self):
        """Test HTML text extraction with invalid directory."""
        nonexistent_dir = "/path/that/does/not/exist"

        with self.assertRaises(FileNotFoundError):
            Piedomain.extract_html_text(True, ["example.com"], nonexistent_dir)

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

    def test_extract_images_permission_error(self):
        """Test image extraction with permission errors."""
        # Create a directory and remove write permissions
        restricted_dir = os.path.join(self.temp_dir, "restricted")
        os.makedirs(restricted_dir)

        # This test may not work on all systems due to permission handling
        try:
            os.chmod(restricted_dir, 0o444)  # Read-only
            used_screenshots, errors = Piedomain.extract_images(
                ["example.com"], False, restricted_dir
            )
            # May or may not fail depending on system
        finally:
            os.chmod(restricted_dir, 0o755)  # Restore permissions

    def test_validate_domains_with_none_values(self):
        """Test domain validation with None values in list."""
        domains_with_none = ["google.com", None, "facebook.com"]

        valid, invalid = Piedomain.validate_domains(domains_with_none)

        self.assertEqual(len(invalid), 1)
        self.assertIn(None, invalid)
        self.assertEqual(len(valid), 2)


if __name__ == "__main__":
    unittest.main()

