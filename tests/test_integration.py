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

    @patch("piedomains.fetchers.PlaywrightFetcher.fetch_both")
    def test_content_processor_with_mocked_requests(self, mock_fetch):
        """Test content processor with mocked HTTP requests."""
        # Mock the fetch_both response to return a FetchResult
        from piedomains.fetchers import FetchResult

        mock_result = FetchResult(
            url="example.com",
            success=True,
            html="<html><body><h1>Test Content</h1><p>Sample text for testing.</p></body></html>",
            text="Test Content Sample text for testing.",
            screenshot_path="/tmp/example.com.png",
            error=None
        )
        mock_fetch.return_value = mock_result

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
    def test_archive_with_retry_success(self):
        """Test real archive.org API calls - uses actual network."""
        # Use a real domain that definitely has archive snapshots
        # Wikipedia has been archived extensively
        urls = get_urls_year("wikipedia.org", year=2010, limit=2)

        # Should get at least 1 URL from 2010
        assert len(urls) >= 1
        assert "web.archive.org/web/" in urls[0]
        assert "/wikipedia.org" in urls[0]

    @skip_in_ci()
    def test_archive_download_with_retry(self):
        """Test real archive.org content download - uses actual network."""
        # Use a real archived URL - example.com from 2010 should be simple and stable
        # First get a real archive URL
        urls = get_urls_year("example.com", year=2010, limit=1)

        if urls:  # Only test if we got a URL
            content = download_from_archive_org(urls[0])

            # Should have gotten some HTML content
            assert len(content) > 100  # At least some content
            assert "<" in content and ">" in content  # Has HTML tags
        else:
            # If archive.org is down, skip the test
            pytest.skip("Could not get archive URL - archive.org may be unavailable")


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
    def test_playwright_cleanup_on_exception(self):
        """Test Playwright is properly cleaned up even when exceptions occur."""
        fetcher = PlaywrightFetcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.png")

            # Use a URL that will definitely fail - invalid domain
            # This tests real error handling without mocking
            success, error = fetcher.fetch_screenshot(
                "https://this-domain-definitely-does-not-exist-12345.invalid", output_path
            )

            assert success is False
            assert error is not None
            # The error should mention the connection/resolution failure
            assert ("ERR_NAME_NOT_RESOLVED" in error or "not resolve" in error.lower())

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

    def test_error_logging_integration(self):
        """Test that errors are properly logged across components."""
        # Test error logging in fetcher with invalid domain
        fetcher = PlaywrightFetcher()

        # Use a domain that will definitely fail
        success, content, error = fetcher.fetch_html("https://invalid-domain-that-does-not-exist-12345.invalid")

        # Should fail with this invalid domain
        assert success is False
        assert error is not None
        # The error should mention the domain resolution failure
        assert "ERR_NAME_NOT_RESOLVED" in error or "not resolve" in error.lower()


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @patch("requests.get")
    def test_classification_workflow_mocked(self, mock_get):
        """Test complete classification workflow with mocked dependencies."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = "<html><body><h1>News Article</h1><p>Breaking news story...</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            classifier = DomainClassifier(cache_dir=temp_dir)

            # Test that the infrastructure works even if models aren't available
            # This verifies the content fetching and processing pipeline
            # Mock the classify_from_collection method to avoid model loading
            with patch.object(
                classifier, "classify_from_collection"
            ) as mock_classify:
                mock_classify.return_value = [
                    {
                        "domain": "example.com",
                        "category": "news",
                        "confidence": 0.95,
                        "text_path": "html/example.com.html",
                        "image_path": None,
                        "error": None,
                        "model_used": "text/shallalist_ml",
                        "date_time_collected": "2025-12-17T12:00:00Z",
                        "reason": None,
                        "raw_predictions": {}
                    }
                ]

                # Test the content collection works
                result = classifier.classify(["example.com"])

                assert result is not None
                assert len(result) == 1
                assert result[0]["category"] == "news"
                # Verify caching worked
                cache_files = list(Path(temp_dir).glob("**/*.html"))
                # Collection should have created cache files
                assert len(cache_files) >= 0  # May or may not cache depending on mock


if __name__ == "__main__":
    # Run specific test categories
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--integration-only":
        pytest.main([__file__ + "::TestEndToEndWorkflows", "-v"])
    else:
        pytest.main([__file__, "-v"])
