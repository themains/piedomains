"""
Test context managers and resource cleanup.
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import shutil
from piedomains.context_managers import (
    webdriver_context, temporary_directory, file_cleanup,
    error_recovery, batch_progress_tracking, ResourceManager
)


class TestContextManagers(unittest.TestCase):
    """Test context manager functionality."""

    @patch('piedomains.context_managers.Piedomain.get_driver')
    def test_webdriver_context_success(self, mock_get_driver):
        """Test successful WebDriver context usage."""
        mock_driver = MagicMock()
        mock_get_driver.return_value = mock_driver

        with webdriver_context() as driver:
            self.assertEqual(driver, mock_driver)

        mock_driver.quit.assert_called_once()

    @patch('piedomains.context_managers.Piedomain.get_driver')
    def test_webdriver_context_with_exception(self, mock_get_driver):
        """Test WebDriver context with exception during usage."""
        mock_driver = MagicMock()
        mock_get_driver.return_value = mock_driver

        with self.assertRaises(RuntimeError):
            with webdriver_context():
                raise RuntimeError("Test exception")

        mock_driver.quit.assert_called_once()

    @patch('piedomains.context_managers.Piedomain.get_driver')
    def test_webdriver_context_cleanup_error(self, mock_get_driver):
        """Test WebDriver context with cleanup error."""
        mock_driver = MagicMock()
        mock_driver.quit.side_effect = Exception("Cleanup error")
        mock_get_driver.return_value = mock_driver

        # Should not raise exception even if cleanup fails
        with webdriver_context():
            pass

        mock_driver.quit.assert_called_once()

    def test_temporary_directory_context(self):
        """Test temporary directory context manager."""
        created_dir = None

        with temporary_directory(suffix="_test", prefix="test_") as temp_dir:
            created_dir = temp_dir
            self.assertTrue(os.path.exists(temp_dir))
            self.assertIn("test_", os.path.basename(temp_dir))
            self.assertTrue(os.path.basename(temp_dir).endswith("_test"))

        # Directory should be cleaned up after context exit
        self.assertFalse(os.path.exists(created_dir))

    def test_temporary_directory_with_exception(self):
        """Test temporary directory context with exception."""
        created_dir = None

        with self.assertRaises(RuntimeError):
            with temporary_directory() as temp_dir:
                created_dir = temp_dir
                self.assertTrue(os.path.exists(temp_dir))
                raise RuntimeError("Test exception")

        # Directory should still be cleaned up after exception
        self.assertFalse(os.path.exists(created_dir))

    def test_file_cleanup_context(self):
        """Test file cleanup context manager."""
        # Create temporary files
        temp_file1 = tempfile.NamedTemporaryFile(delete=False)
        temp_file2 = tempfile.NamedTemporaryFile(delete=False)
        temp_file1.close()
        temp_file2.close()

        file1_path = temp_file1.name
        file2_path = temp_file2.name

        # Verify files exist
        self.assertTrue(os.path.exists(file1_path))
        self.assertTrue(os.path.exists(file2_path))

        with file_cleanup(file1_path, file2_path):
            pass

        # Files should be cleaned up
        self.assertFalse(os.path.exists(file1_path))
        self.assertFalse(os.path.exists(file2_path))

    def test_file_cleanup_with_nonexistent_file(self):
        """Test file cleanup with nonexistent files."""
        nonexistent_file = "/path/that/does/not/exist.txt"

        # Should not raise exception for nonexistent files
        with file_cleanup(nonexistent_file):
            pass

    def test_error_recovery_success(self):
        """Test error recovery context manager with successful operation."""
        with error_recovery("test_operation", fallback_value="fallback") as result:
            result['result'] = "success_value"

        self.assertTrue(result['success'])
        self.assertIsNone(result['error'])
        self.assertEqual(result['result'], "success_value")

    def test_error_recovery_with_error_no_reraise(self):
        """Test error recovery context manager with error (no reraise)."""
        with error_recovery("test_operation", fallback_value="fallback", reraise=False) as result:
            raise ValueError("Test error")

        self.assertFalse(result['success'])
        self.assertIsInstance(result['error'], ValueError)
        self.assertEqual(result['result'], "fallback")

    def test_error_recovery_with_error_reraise(self):
        """Test error recovery context manager with error (reraise)."""
        with self.assertRaises(ValueError):
            with error_recovery("test_operation", reraise=True):
                raise ValueError("Test error")

    @patch('piedomains.context_managers.logger')
    def test_batch_progress_tracking(self, mock_logger):
        """Test batch progress tracking context manager."""
        with batch_progress_tracking(25, "Test Operation") as update_progress:
            # Update progress multiple times
            update_progress(5)  # Should not log (not multiple of 10)
            update_progress(5)  # Should log (10 total)
            update_progress(10)  # Should log (20 total)
            update_progress(5)  # Should log (25 total, completed)

        # Verify logging calls
        self.assertTrue(mock_logger.info.called)
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]

        # Should have start, progress updates, and completion logs
        self.assertTrue(any("Starting Test Operation" in call for call in log_calls))
        self.assertTrue(any("10/25 completed" in call for call in log_calls))
        self.assertTrue(any("20/25 completed" in call for call in log_calls))
        self.assertTrue(any("25/25 completed" in call for call in log_calls))
        self.assertTrue(any("Completed Test Operation" in call for call in log_calls))

    def test_resource_manager_context(self):
        """Test ResourceManager as context manager."""
        mock_driver = MagicMock()
        temp_dir = tempfile.mkdtemp()
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        try:
            with ResourceManager() as rm:
                rm.add_driver(mock_driver)
                rm.add_temp_directory(temp_dir)
                rm.add_temp_file(temp_file.name)

            # Resources should be cleaned up
            mock_driver.quit.assert_called_once()
            self.assertFalse(os.path.exists(temp_dir))
            self.assertFalse(os.path.exists(temp_file.name))
        finally:
            # Cleanup in case test fails
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def test_resource_manager_cleanup_errors(self):
        """Test ResourceManager handles cleanup errors gracefully."""
        mock_driver = MagicMock()
        mock_driver.quit.side_effect = Exception("Cleanup error")

        with ResourceManager() as rm:
            rm.add_driver(mock_driver)
            rm.add_temp_directory("/nonexistent/directory")
            rm.add_temp_file("/nonexistent/file.txt")

        # Should not raise exceptions even if cleanup fails
        mock_driver.quit.assert_called_once()

    def test_resource_manager_manual_cleanup(self):
        """Test ResourceManager manual cleanup."""
        mock_driver = MagicMock()
        rm = ResourceManager()

        rm.add_driver(mock_driver)
        rm.cleanup_all()

        mock_driver.quit.assert_called_once()

        # Should be able to call cleanup multiple times
        rm.cleanup_all()
        self.assertEqual(mock_driver.quit.call_count, 1)  # No additional calls


if __name__ == "__main__":
    unittest.main()

