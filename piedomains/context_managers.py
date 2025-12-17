"""
Context managers for resource cleanup and management.
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

from .fetchers import PlaywrightFetcher
from .piedomains_logging import get_logger

logger = get_logger()


@contextmanager
def webdriver_context():
    """
    DEPRECATED: Use PlaywrightFetcher context manager instead.

    This function is maintained for backward compatibility.
    """
    logger.warning(
        "webdriver_context is deprecated. Use PlaywrightFetcher context manager instead."
    )

    # Return a mock-like object that has a quit method for compatibility
    class DeprecatedWebDriverStub:
        def quit(self):
            pass

    stub = DeprecatedWebDriverStub()
    try:
        yield stub
    finally:
        stub.quit()


@contextmanager
def playwright_context() -> Generator[PlaywrightFetcher, None, None]:
    """
    Context manager for PlaywrightFetcher instances.

    Yields:
        PlaywrightFetcher: Playwright fetcher instance

    Ensures proper cleanup of Playwright resources.
    """
    fetcher = None
    try:
        fetcher = PlaywrightFetcher()
        yield fetcher
    except Exception as e:
        logger.error(f"Playwright error: {e}")
        raise
    finally:
        if fetcher:
            try:
                fetcher.cleanup()
                logger.debug("Playwright fetcher cleaned up successfully")
            except Exception as cleanup_error:
                logger.warning(f"Error during Playwright cleanup: {cleanup_error}")


@contextmanager
def temporary_directory(
    suffix: str = "", prefix: str = "piedomains_"
) -> Generator[str, None, None]:
    """
    Context manager for temporary directories.

    Args:
        suffix (str): Directory name suffix
        prefix (str): Directory name prefix

    Yields:
        str: Path to temporary directory

    Ensures cleanup of temporary directories.
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
        logger.debug(f"Created temporary directory: {temp_dir}")
        yield temp_dir
    except Exception as e:
        logger.error(f"Temporary directory error: {e}")
        raise
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Error cleaning up temporary directory {temp_dir}: {cleanup_error}"
                )


@contextmanager
def file_cleanup(*file_paths: str) -> Generator[None, None, None]:
    """
    Context manager for file cleanup.

    Args:
        *file_paths: Paths to files that should be cleaned up

    Ensures cleanup of specified files after context exits.
    """
    try:
        yield
    finally:
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up file: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(
                        f"Error cleaning up file {file_path}: {cleanup_error}"
                    )


@contextmanager
def error_recovery(
    operation_name: str, fallback_value=None, reraise: bool = False
) -> Generator:
    """
    Context manager for error recovery with logging.

    Args:
        operation_name (str): Name of the operation for logging
        fallback_value: Value to return on error (if not reraising)
        reraise (bool): Whether to reraise exceptions

    Yields:
        dict: Dictionary with 'success', 'error', 'result' keys
    """
    result = {"success": False, "error": None, "result": fallback_value}

    try:
        logger.debug(f"Starting operation: {operation_name}")
        yield result
        result["success"] = True
        logger.debug(f"Operation completed successfully: {operation_name}")
    except Exception as e:
        result["error"] = e
        logger.error(f"Operation failed: {operation_name} - {str(e)}")

        if reraise:
            raise
        else:
            result["result"] = fallback_value


@contextmanager
def batch_progress_tracking(
    total_items: int, operation_name: str = "Processing"
) -> Generator:
    """
    Context manager for tracking batch processing progress.

    Args:
        total_items (int): Total number of items to process
        operation_name (str): Name of the operation

    Yields:
        callable: Function to update progress
    """
    processed = {"count": 0}

    def update_progress(increment: int = 1):
        processed["count"] += increment
        if processed["count"] % 10 == 0 or processed["count"] == total_items:
            logger.info(
                f"{operation_name}: {processed['count']}/{total_items} completed "
                f"({processed['count'] / total_items * 100:.1f}%)"
            )

    try:
        logger.info(f"Starting {operation_name}: {total_items} items")
        yield update_progress
        logger.info(
            f"Completed {operation_name}: {processed['count']}/{total_items} processed"
        )
    except Exception as e:
        logger.error(
            f"{operation_name} failed after processing {processed['count']}/{total_items} items: {e}"
        )
        raise


class ResourceManager:
    """
    Resource manager for tracking and cleaning up resources.
    """

    def __init__(self):
        self._drivers = []
        self._temp_dirs = []
        self._temp_files = []

    def add_driver(self, driver):
        """Add a WebDriver/fetcher instance for cleanup (deprecated)."""
        logger.warning(
            "add_driver is deprecated. Use add_fetcher for Playwright fetchers."
        )
        self._drivers.append(driver)

    def add_fetcher(self, fetcher):
        """Add a PlaywrightFetcher instance for cleanup."""
        self._drivers.append(fetcher)  # Reuse the same list for compatibility

    def add_temp_directory(self, path: str):
        """Add a temporary directory for cleanup."""
        self._temp_dirs.append(path)

    def add_temp_file(self, path: str):
        """Add a temporary file for cleanup."""
        self._temp_files.append(path)

    def cleanup_all(self):
        """Clean up all tracked resources."""
        # Clean up WebDriver/fetcher instances
        for driver_or_fetcher in self._drivers:
            try:
                # Try Playwright fetcher cleanup first
                if hasattr(driver_or_fetcher, "cleanup"):
                    driver_or_fetcher.cleanup()
                    logger.debug("Playwright fetcher cleaned up")
                # Fallback to WebDriver quit for backward compatibility
                elif hasattr(driver_or_fetcher, "quit"):
                    driver_or_fetcher.quit()
                    logger.debug("WebDriver cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up driver/fetcher: {e}")
        self._drivers.clear()

        # Clean up temporary files
        for file_path in self._temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Error cleaning up temp file {file_path}: {e}")
        self._temp_files.clear()

        # Clean up temporary directories
        for dir_path in self._temp_dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    logger.debug(f"Cleaned up temp directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory {dir_path}: {e}")
        self._temp_dirs.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()
