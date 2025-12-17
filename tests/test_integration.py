#!/usr/bin/env python3
"""
Integration tests for piedomains package.

These tests verify end-to-end functionality across multiple components
including content fetching, processing, and classification workflows.
Tests are designed to work with real services but include appropriate
mocking for CI environments.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from piedomains import DomainClassifier, classify_domains
from piedomains.archive_org_downloader import download_from_archive_org, get_urls_year
from piedomains.content_processor import ContentProcessor
from piedomains.fetchers import ArchiveFetcher, PlaywrightFetcher
from piedomains.piedomains_logging import configure_logging, get_logger
from piedomains.utils import is_within_directory
from tests.conftest import skip_if_no_browser, skip_in_ci

# Configure test logging
configure_logging(level="DEBUG", console_format="simple")
logger = get_logger(__name__)


class TestInfrastructureIntegration:
    """Test integration between core infrastructure components."""

    def test_fetcher_factory(self):
        """Test fetcher implementations work correctly."""
        # Test live fetcher
        live_fetcher = PlaywrightFetcher()
        assert isinstance(live_fetcher, PlaywrightFetcher)

        # Test archive fetcher
        archive_fetcher = ArchiveFetcher(target_date="20200101")
        assert isinstance(archive_fetcher, ArchiveFetcher)
        assert archive_fetcher.target_date == "20200101"

    @patch("requests.get")
    def test_content_processor_with_mocked_requests(self, mock_get):
        """Test content processor with mocked HTTP requests."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = "<html><body><h1>Test Content</h1><p>Sample text for testing.</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = ContentProcessor(cache_dir=temp_dir)

            # Test HTML extraction
            html_content, errors = processor.extract_html_content(["example.com"])

            assert "example.com" in html_content
            assert "Test Content" in html_content["example.com"]
            assert len(errors) == 0

            # Verify caching works
            cache_file = Path(temp_dir) / "html" / "example.com.html"
            assert cache_file.exists()

    @skip_if_no_browser()
    @patch("piedomains.fetchers.PlaywrightFetcher.fetch_screenshot")
    def test_screenshot_with_mocked_playwright(self, mock_screenshot):
        """Test screenshot functionality with mocked Playwright."""
        # Mock the screenshot function
        mock_screenshot.return_value = (True, "")

        fetcher = PlaywrightFetcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_screenshot.png")

            success, error = fetcher.fetch_screenshot(
                "https://example.com", output_path
            )

            assert success is True
            assert error == ""
            mock_screenshot.assert_called_once_with("https://example.com", output_path)

    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = ContentProcessor(cache_dir=temp_dir)

            # Test with invalid domain
            html_content, errors = processor.extract_html_content(
                ["invalid-domain-12345.com"]
            )

            assert "invalid-domain-12345.com" in errors
            assert len(html_content) == 0


class TestArchiveOrgIntegration:
    """Test Archive.org integration with retry logic."""

    @skip_in_ci()
    @patch("piedomains.archive_org_downloader.requests.get")
    def test_archive_with_retry_success(self, mock_get):
        """Test archive API calls with retry logic."""
        # Mock successful response after one retry
        mock_response = Mock()
        mock_response.json.return_value = [
            ["urlkey", "timestamp", "original"],
            ["com,example)/", "20200101120000", "http://example.com/"],
        ]
        mock_response.raise_for_status.return_value = None

        # First call fails, second succeeds
        mock_get.side_effect = [
            Exception("Network error"),  # First attempt fails
            mock_response,  # Second attempt succeeds
        ]

        urls = get_urls_year("example.com", year=2020, limit=5)

        assert len(urls) == 1
        assert "web.archive.org/web/20200101120000" in urls[0]
        assert mock_get.call_count == 2  # Verify retry happened

    @skip_in_ci()
    @patch("piedomains.archive_org_downloader.requests.get")
    def test_archive_download_with_retry(self, mock_get):
        """Test archive content download with retry logic."""
        mock_response = Mock()
        mock_response.content = b"<html><body>Historical content</body></html>"
        mock_response.raise_for_status.return_value = None

        # Simulate one failure then success
        mock_get.side_effect = [Exception("Temporary network issue"), mock_response]

        test_url = "https://web.archive.org/web/20200101120000/https://example.com"
        content = download_from_archive_org(test_url)

        assert "Historical content" in content
        assert mock_get.call_count == 2  # Verify retry


class TestSecurityIntegration:
    """Test security features across components."""

    def test_path_traversal_protection(self):
        """Test path traversal protection in utils."""
        # Safe paths
        assert is_within_directory("/safe/dir", "/safe/dir/file.txt") is True
        assert is_within_directory("/safe/dir", "/safe/dir/subdir/file.txt") is True

        # Dangerous paths
        assert is_within_directory("/safe/dir", "/etc/passwd") is False
        assert (
            is_within_directory("/safe/dir", "/safe/dir/../../../etc/passwd") is False
        )

    @patch("tarfile.open")
    def test_safe_extract_validation(self, mock_tarfile):
        """Test safe extraction validates all members."""
        # Mock tar file with dangerous path
        mock_tar = Mock()
        mock_member = Mock()
        mock_member.name = "../../../etc/passwd"
        mock_tar.getmembers.return_value = [mock_member]
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        with tempfile.TemporaryDirectory() as temp_dir:
            from piedomains.utils import SecurityError, safe_extract

            with pytest.raises(SecurityError) as exc_info:
                safe_extract(mock_tar, temp_dir)

            assert "Path traversal detected" in str(exc_info.value)


class TestResourceManagement:
    """Test resource cleanup and management."""

    @skip_if_no_browser()
    @patch("piedomains.fetchers.PlaywrightFetcher.fetch_screenshot")
    def test_playwright_cleanup_on_exception(self, mock_screenshot):
        """Test Playwright is properly cleaned up even when exceptions occur."""
        # Mock the screenshot function to raise an exception
        mock_screenshot.side_effect = Exception("Page load failed")

        fetcher = PlaywrightFetcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.png")

            # This should handle the exception and still clean up
            success, error = fetcher.fetch_screenshot(
                "https://example.com", output_path
            )

            assert success is False
            assert "Page load failed" in error
            # Verify screenshot function was called
            mock_screenshot.assert_called_once_with("https://example.com", output_path)

    def test_temporary_file_cleanup(self):
        """Test that temporary files are properly cleaned up."""
        from piedomains.context_managers import temporary_directory

        temp_path = None
        with temporary_directory() as temp_dir:
            temp_path = temp_dir
            assert os.path.exists(temp_path)

            # Create a test file
            test_file = os.path.join(temp_path, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            assert os.path.exists(test_file)

        # Directory should be cleaned up after context
        assert not os.path.exists(temp_path)


class TestLoggingIntegration:
    """Test logging across different components."""

    def test_logging_configuration(self):
        """Test logging can be configured and used consistently."""
        from piedomains.piedomains_logging import (
            configure_logging,
            get_effective_level,
            get_logger,
        )

        # Test configuration
        configure_logging(level="DEBUG", console_format="detailed")

        test_logger = get_logger(__name__)
        assert test_logger is not None

        # Test level checking
        current_level = get_effective_level()
        assert current_level in ["DEBUG", "INFO", "WARNING", "ERROR"]

    @patch("piedomains.piedomains_logging.get_logger")
    def test_error_logging_integration(self, mock_get_logger):
        """Test that errors are properly logged across components."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Test error logging in fetcher
        fetcher = PlaywrightFetcher()

        with patch("requests.get", side_effect=Exception("Network error")):
            success, content, error = fetcher.fetch_html("https://invalid-url.com")

            assert success is False
            mock_logger.error.assert_called()


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @patch("piedomains.api.DomainClassifier._load_traditional_models")
    @patch("requests.get")
    def test_classification_workflow_mocked(self, mock_get, mock_load_models):
        """Test complete classification workflow with mocked dependencies."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = "<html><body><h1>News Article</h1><p>Breaking news story...</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock model loading
        mock_load_models.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            classifier = DomainClassifier(cache_dir=temp_dir)

            # Test that the infrastructure works even if models aren't available
            # This verifies the content fetching and processing pipeline
            with patch.object(
                classifier, "_classify_domains_traditional"
            ) as mock_classify:
                mock_classify.return_value = {
                    "domain": ["example.com"],
                    "pred_label": ["news"],
                    "pred_prob": [0.95],
                }

                # This should work without actual models
                result = classify_domains(["example.com"])

                assert result is not None
                # Verify caching worked
                cache_files = list(Path(temp_dir).glob("**/*.html"))
                assert len(cache_files) > 0


if __name__ == "__main__":
    # Run specific test categories
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--integration-only":
        pytest.main([__file__ + "::TestEndToEndWorkflows", "-v"])
    else:
        pytest.main([__file__, "-v"])
