#!/usr/bin/env python3
"""
Shared test fixtures and configuration for piedomains tests.

This file provides common pytest fixtures, test utilities, and markers
used across the test suite.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def browser_available() -> bool:
    """Check if Playwright browsers are available."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
        return True
    except Exception:
        return False


def skip_if_no_browser():
    """Skip test if Playwright browsers are not available."""
    return pytest.mark.skipif(
        not browser_available(),
        reason="Playwright browsers not available"
    )


def skip_in_ci():
    """Skip test in CI environment."""
    return pytest.mark.skipif(
        os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
        reason="Skipped in CI environment"
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory structure for tests."""
    temp_dir = tempfile.mkdtemp()

    # Create subdirectories
    html_dir = os.path.join(temp_dir, "html")
    image_dir = os.path.join(temp_dir, "images")
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    yield {
        "root": temp_dir,
        "html": html_dir,
        "images": image_dir
    }

    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_classifier():
    """Create a mock DomainClassifier for testing."""
    classifier = MagicMock()

    # Default return values for common methods
    default_result = [
        {
            "url": "test.com",
            "domain": "test.com",
            "text_path": "html/test.com.html",
            "image_path": "images/test.com.png",
            "date_time_collected": "2025-12-17T12:00:00Z",
            "model_used": "combined/text_image_ml",
            "category": "test",
            "confidence": 0.9,
            "reason": None,
            "error": None,
            "raw_predictions": {"test": 0.9, "other": 0.1}
        }
    ]

    classifier.classify.return_value = default_result
    classifier.classify_by_text.return_value = default_result
    classifier.classify_by_images.return_value = default_result

    return classifier


@pytest.fixture
def mock_fetcher():
    """Create a mock fetcher for testing."""
    fetcher = MagicMock()

    # Mock FetchResult
    from piedomains.fetchers import FetchResult
    mock_result = FetchResult(
        url="http://test.com",
        success=True,
        html="<html><head><title>Test</title></head><body>Test content</body></html>",
        screenshot_path="/tmp/test.png",
        error=None
    )

    fetcher.fetch_single.return_value = mock_result
    fetcher.fetch_batch.return_value = [mock_result]
    fetcher.fetch_html.return_value = (True, mock_result.html, None)
    fetcher.fetch_screenshot.return_value = (True, None)

    return fetcher


@pytest.fixture
def sample_domains():
    """Provide a list of sample domains for testing."""
    return [
        "google.com",
        "wikipedia.org",
        "github.com",
        "stackoverflow.com",
        "amazon.com"
    ]


@pytest.fixture
def sample_html_content():
    """Provide sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sample Test Page</title>
        <meta name="description" content="This is a sample page for testing">
        <meta name="keywords" content="test, sample, html">
    </head>
    <body>
        <header>
            <h1>Sample Test Website</h1>
            <nav>
                <ul>
                    <li><a href="#home">Home</a></li>
                    <li><a href="#about">About</a></li>
                    <li><a href="#contact">Contact</a></li>
                </ul>
            </nav>
        </header>

        <main>
            <section>
                <h2>Welcome to our test site</h2>
                <p>This is sample content for testing text processing and classification.</p>
                <p>We provide various services including web development, consulting, and support.</p>
            </section>

            <aside>
                <h3>Latest News</h3>
                <ul>
                    <li>Product launch announcement</li>
                    <li>Company milestone achieved</li>
                    <li>New partnership formed</li>
                </ul>
            </aside>
        </main>

        <footer>
            <p>&copy; 2024 Test Company. All rights reserved.</p>
            <p>Contact us: info@test.com | Phone: (555) 123-4567</p>
        </footer>

        <!-- Test scripts and styles -->
        <script>
            console.log('Test page loaded');
        </script>
        <style>
            body { font-family: Arial, sans-serif; }
            .hidden { display: none; }
        </style>
    </body>
    </html>
    """


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "ml: tests that require ML models")
    config.addinivalue_line("markers", "integration: integration tests that may use real services")
    config.addinivalue_line("markers", "slow: tests that take a long time to run")
    config.addinivalue_line("markers", "archive: tests that use archive.org")
    config.addinivalue_line("markers", "llm: tests that require LLM API access")


# Test environment setup
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables and cleanup."""
    # Set test-friendly environment variables
    original_env = {}
    test_env_vars = {
        "PIEDOMAINS_LOG_LEVEL": "WARNING",  # Reduce logging noise in tests
        "PLAYWRIGHT_HEADLESS": "true",      # Force headless mode in tests
    }

    # Set test environment
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def skip_if_no_models():
    """Skip test if ML models are not available."""
    def _skip_if_no_models():
        try:
            from piedomains.config import get_config
            config = get_config()
            model_dir = Path(config.get("model_cache_dir", "model"))
            if not (model_dir / "shallalist").exists():
                pytest.skip("ML models not available")
        except Exception:
            pytest.skip("Unable to check for ML models")

    return _skip_if_no_models


@pytest.fixture
def skip_if_no_network():
    """Skip test if network is not available."""
    def _skip_if_no_network():
        try:
            import requests
            requests.get("https://httpbin.org/get", timeout=5)
        except Exception:
            pytest.skip("Network not available")

    return _skip_if_no_network
