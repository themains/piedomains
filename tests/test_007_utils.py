"""
Test utility functions.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import tarfile
import os
from piedomains import utils


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_is_within_directory_safe_path(self):
        """Test is_within_directory with safe paths."""
        directory = "/safe/directory"
        target = "/safe/directory/file.txt"

        result = utils.is_within_directory(directory, target)
        self.assertTrue(result)

    def test_is_within_directory_unsafe_path(self):
        """Test is_within_directory with path traversal attempt."""
        directory = "/safe/directory"
        target = "/safe/directory/../../../etc/passwd"

        result = utils.is_within_directory(directory, target)
        self.assertFalse(result)

    def test_is_within_directory_same_path(self):
        """Test is_within_directory with identical paths."""
        directory = "/safe/directory"
        target = "/safe/directory"

        result = utils.is_within_directory(directory, target)
        self.assertTrue(result)

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('tarfile.open')
    @patch('os.remove')
    def test_download_file_success(self, mock_remove, mock_tarfile, mock_file, mock_get):
        """Test successful file download and extraction."""
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.content = b"fake tar content"
        mock_get.return_value = mock_response

        # Mock tarfile extraction
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        result = utils.download_file("http://example.com/file.tar.gz", "/target", "file.tar.gz")

        self.assertTrue(result)
        mock_get.assert_called_once_with("http://example.com/file.tar.gz", allow_redirects=True, timeout=10)
        mock_file.assert_called_once_with("/target/file.tar.gz", "wb")
        mock_remove.assert_called_once_with("/target/file.tar.gz")

    @patch('requests.get')
    def test_download_file_http_error(self, mock_get):
        """Test file download with HTTP error."""
        mock_get.side_effect = Exception("HTTP error")

        result = utils.download_file("http://example.com/file.tar.gz", "/target", "file.tar.gz")

        self.assertFalse(result)

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('tarfile.open')
    def test_download_file_extraction_error(self, mock_tarfile, mock_file, mock_get):
        """Test file download with extraction error."""
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.content = b"fake tar content"
        mock_get.return_value = mock_response

        # Mock tarfile extraction error
        mock_tarfile.side_effect = Exception("Extraction error")

        result = utils.download_file("http://example.com/file.tar.gz", "/target", "file.tar.gz")

        self.assertFalse(result)

    def test_safe_extract_safe_members(self):
        """Test safe_extract with safe tar members."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple tar file
            tar_path = os.path.join(temp_dir, "test.tar")
            test_file_path = os.path.join(temp_dir, "test.txt")

            with open(test_file_path, "w") as f:
                f.write("test content")

            with tarfile.open(tar_path, "w") as tar:
                tar.add(test_file_path, arcname="test.txt")

            # Extract to another directory
            extract_dir = os.path.join(temp_dir, "extract")
            os.makedirs(extract_dir)

            with tarfile.open(tar_path, "r") as tar:
                utils.safe_extract(tar, extract_dir)

            # Check if file was extracted
            extracted_file = os.path.join(extract_dir, "test.txt")
            self.assertTrue(os.path.exists(extracted_file))

    def test_safe_extract_path_traversal_attempt(self):
        """Test safe_extract with path traversal attempt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a tar file with path traversal attempt
            tar_path = os.path.join(temp_dir, "malicious.tar")

            with tarfile.open(tar_path, "w") as tar:
                # Create a tarinfo object with malicious path
                info = tarfile.TarInfo(name="../../../etc/passwd")
                info.size = 0
                tar.addfile(info, fileobj=None)

            extract_dir = os.path.join(temp_dir, "extract")
            os.makedirs(extract_dir)

            with tarfile.open(tar_path, "r") as tar:
                with self.assertRaises(Exception) as context:
                    utils.safe_extract(tar, extract_dir)

                self.assertIn("Failed Path Traversal", str(context.exception))

    def test_repo_base_url_environment_variable(self):
        """Test REPO_BASE_URL uses environment variable when set."""
        original_url = utils.REPO_BASE_URL

        try:
            # Test with environment variable
            with patch.dict(os.environ, {'PIEDOMAINS_MODEL_URL': 'http://custom.url/model'}):
                # Reload the module to pick up the environment variable
                import importlib
                importlib.reload(utils)
                self.assertEqual(utils.REPO_BASE_URL, 'http://custom.url/model')
        finally:
            # Restore original
            utils.REPO_BASE_URL = original_url

    def test_repo_base_url_default_value(self):
        """Test REPO_BASE_URL uses default when no environment variable."""
        # Ensure environment variable is not set
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            importlib.reload(utils)
            self.assertEqual(utils.REPO_BASE_URL, "https://dataverse.harvard.edu/api/access/datafile/7081895")


if __name__ == "__main__":
    unittest.main()

