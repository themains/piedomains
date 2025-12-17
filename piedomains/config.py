"""
Configuration management for piedomains.
"""

from __future__ import annotations

import os

from .piedomains_logging import get_logger

logger = get_logger()


class Config:
    """Configuration class for piedomains settings."""

    # Default configuration values
    DEFAULT_CONFIG = {
        # Network timeouts
        "http_timeout": 10,
        # Playwright settings
        "playwright_timeout": 30000,  # milliseconds
        "playwright_headless": True,
        "playwright_viewport": {"width": 1280, "height": 1024},
        # Parallel processing
        "max_parallel": 4,
        # Block heavy resources
        "block_media": True,
        "block_resources": ["media", "video", "font", "websocket", "manifest"],
        # Retry settings
        "max_retries": 3,
        "retry_delay": 1,  # seconds
        # Model settings
        "model_cache_dir": None,  # Will use default if None
        "image_size": (254, 254),
        # Batch processing
        "batch_size": 50,  # For processing large numbers of domains
        "parallel_workers": 4,  # For concurrent processing
        # File settings
        "html_extension": ".html",
        "image_extension": ".png",
        # User agent for HTTP requests
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        # Legacy WebDriver settings for backward compatibility
        "webdriver_timeout": 30,
        "webdriver_window_size": "1280,1024",
        # Logging
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # Content validation and security settings
        "enable_content_validation": True,  # Enable content-type and security checks
        "content_safety_mode": "moderate",  # strict, moderate, permissive (default: moderate for better UX)
        "max_content_length": 10 * 1024 * 1024,  # 10MB max download size
        "sandbox_mode_required": False,  # Force sandbox execution for risky content
        "validate_domain_extensions": False,  # If True, validate domain TLDs (usually False for better UX)
        # Allowed content types (MIME types)
        "allowed_content_types": [
            "text/html",
            "application/xhtml+xml",
            "application/xml",
            "text/xml",
            "text/plain",
        ],
        # Dangerous file extensions to block by default
        # Note: These are for file downloads in URL paths, not domain extensions
        # Examples: "site.com/malware.exe" (blocked), "cnn.com" (allowed)
        "blocked_extensions": [
            ".exe",
            ".msi",
            ".scr",
            ".bat",
            ".cmd",
            ".com",
            ".pif",
            ".vbs",
            ".jar",
            ".app",
            ".dmg",
            ".pkg",
            ".deb",
            ".rpm",
            ".run",
            ".bin",
            ".elf",
            ".so",
            ".dll",
            ".dylib",
        ],
        # Suspicious URL patterns for detecting non-HTML content
        # These patterns match file downloads in paths, not domain names
        "suspicious_url_patterns": [
            r".*\/[^\/]*\.(exe|msi|scr|bat|cmd|pif|vbs|jar)(\?.*)?$",  # Executable files in path
            r".*\.com\/.*\.(exe|msi|scr|bat|cmd|pif|vbs|jar)(\?.*)?$",  # Executable after .com domain
            r".*\/download\/.*\.(zip|rar|7z|tar\.gz|tgz)(\?.*)?$",  # Archive downloads
            r".*\/attachment\/.*",  # Forced download attachments
            r".*[?&](download|attachment)=.*",  # Download parameters
        ],
        # Content-Length thresholds by content type
        "content_length_limits": {
            "text/html": 5 * 1024 * 1024,  # 5MB for HTML
            "application/pdf": 50 * 1024 * 1024,  # 50MB for PDF
            "default": 10 * 1024 * 1024,  # 10MB default
        },
    }

    def __init__(self, config_dict: dict[str, any] | None = None):
        """
        Initialize configuration.

        Args:
            config_dict (Dict[str, Any]): Optional configuration overrides
        """
        self._config = self.DEFAULT_CONFIG.copy()

        # Override with environment variables
        self._load_from_environment()

        # Override with provided config
        if config_dict:
            self._config.update(config_dict)

    def _load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "PIEDOMAINS_HTTP_TIMEOUT": ("http_timeout", int),
            "PIEDOMAINS_WEBDRIVER_TIMEOUT": ("webdriver_timeout", int),
            "PIEDOMAINS_PAGE_LOAD_TIMEOUT": ("page_load_timeout", int),
            "PIEDOMAINS_MAX_RETRIES": ("max_retries", int),
            "PIEDOMAINS_RETRY_DELAY": ("retry_delay", float),
            "PIEDOMAINS_SCREENSHOT_WAIT": ("screenshot_wait_time", int),
            "PIEDOMAINS_WINDOW_SIZE": ("webdriver_window_size", str),
            "PIEDOMAINS_BATCH_SIZE": ("batch_size", int),
            "PIEDOMAINS_PARALLEL_WORKERS": ("parallel_workers", int),
            "PIEDOMAINS_USER_AGENT": ("user_agent", str),
            "PIEDOMAINS_LOG_LEVEL": ("log_level", str),
            "PIEDOMAINS_ENABLE_CONTENT_VALIDATION": (
                "enable_content_validation",
                lambda x: x.lower() in ("true", "1", "yes"),
            ),
            "PIEDOMAINS_CONTENT_SAFETY_MODE": ("content_safety_mode", str),
            "PIEDOMAINS_MAX_CONTENT_LENGTH": ("max_content_length", int),
            "PIEDOMAINS_SANDBOX_MODE_REQUIRED": (
                "sandbox_mode_required",
                lambda x: x.lower() in ("true", "1", "yes"),
            ),
        }

        for env_var, (config_key, type_func) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    self._config[config_key] = type_func(env_value)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid value for {env_var}: {env_value}. Using default. Error: {e}"
                    )

    def get(self, key: str, default=None):
        """
        Get configuration value.

        Args:
            key (str): Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._config.get(key, default)

    def set(self, key: str, value: any):
        """
        Set configuration value.

        Args:
            key (str): Configuration key
            value (any): Configuration value
        """
        self._config[key] = value

    def update(self, config_dict: dict[str, any]):
        """
        Update multiple configuration values.

        Args:
            config_dict (dict[str, any]): Configuration updates
        """
        self._config.update(config_dict)

    def to_dict(self) -> dict[str, any]:
        """
        Get configuration as dictionary.

        Returns:
            dict[str, any]: Configuration dictionary
        """
        return self._config.copy()

    @property
    def http_timeout(self) -> int:
        """HTTP request timeout in seconds."""
        return self._config["http_timeout"]

    @property
    def webdriver_timeout(self) -> int:
        """WebDriver timeout in seconds."""
        return self._config["webdriver_timeout"]

    @property
    def page_load_timeout(self) -> int:
        """Page load timeout in seconds."""
        return self._config["page_load_timeout"]

    @property
    def max_retries(self) -> int:
        """Maximum number of retries for failed operations."""
        return self._config["max_retries"]

    @property
    def retry_delay(self) -> float:
        """Delay between retries in seconds."""
        return self._config["retry_delay"]

    @property
    def screenshot_wait_time(self) -> int:
        """Wait time after loading page before screenshot."""
        return self._config["screenshot_wait_time"]

    @property
    def webdriver_window_size(self) -> str:
        """WebDriver window size."""
        return self._config["webdriver_window_size"]

    @property
    def batch_size(self) -> int:
        """Batch size for processing domains."""
        return self._config["batch_size"]

    @property
    def parallel_workers(self) -> int:
        """Number of parallel workers."""
        return self._config["parallel_workers"]

    @property
    def user_agent(self) -> str:
        """User agent string for HTTP requests."""
        return self._config["user_agent"]

    @property
    def image_size(self) -> tuple:
        """Image size for model input."""
        return self._config["image_size"]

    @property
    def enable_content_validation(self) -> bool:
        """Whether content validation is enabled."""
        return self._config["enable_content_validation"]

    @property
    def content_safety_mode(self) -> str:
        """Content safety mode: strict, moderate, or permissive."""
        return self._config["content_safety_mode"]

    @property
    def max_content_length(self) -> int:
        """Maximum content length to download."""
        return self._config["max_content_length"]

    @property
    def sandbox_mode_required(self) -> bool:
        """Whether sandbox mode is required for risky content."""
        return self._config["sandbox_mode_required"]

    @property
    def allowed_content_types(self) -> list:
        """List of allowed MIME types."""
        return self._config["allowed_content_types"]

    @property
    def blocked_extensions(self) -> list:
        """List of blocked file extensions."""
        return self._config["blocked_extensions"]

    @property
    def suspicious_url_patterns(self) -> list:
        """List of regex patterns for suspicious URLs."""
        return self._config["suspicious_url_patterns"]

    @property
    def content_length_limits(self) -> dict:
        """Content length limits by content type."""
        return self._config["content_length_limits"]


# Global configuration instance
_global_config = None


def get_config() -> Config:
    """
    Get global configuration instance.

    Returns:
        Config: Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """
    Set global configuration instance.

    Args:
        config (Config): Configuration instance to set as global
    """
    global _global_config
    _global_config = config


def configure(**kwargs):
    """
    Configure global settings.

    Args:
        **kwargs: Configuration key-value pairs
    """
    config = get_config()
    config.update(kwargs)
