"""Configuration management for piedomains."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from .piedomains_logging import get_logger

logger = get_logger()


class Config:
    """Configuration class for piedomains settings."""

    # Default configuration values
    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        # Network timeouts
        "http_timeout": 10,
        # Playwright settings
        "playwright_timeout": 30000,  # milliseconds
        "playwright_headless": True,
        "playwright_viewport": {"width": 1280, "height": 1024},
        # Device pixel ratio for screenshots. 1 because the image model reads
        # 224px: capturing 4x the pixels only to discard them would quadruple
        # every cached file and risk train/serve skew against a corpus captured
        # at 1x. Raise it when the screenshot is the deliverable rather than
        # model input.
        "screenshot_scale": 1,
        # Wait for the DOM, settle briefly, then race a capped network-quiet
        # window. `networkidle` alone loses whole sites (see fetchers.py).
        "settle_ms": 1500,
        "network_quiet_ms": 3000,
        # Minimum usable tokens before a label is meaningful.
        "min_tokens": 30,
        # "trafilatura" | "legacy". Measured across 188 cached homepages, both through
        # the minimal cleaner:
        #
        #   legacy       median 1,349 tokens, 0/188 under the floor
        #   trafilatura  median   251 tokens, 1/188 under the floor
        #
        # trafilatura keeps ~29% of the words and costs one page. What it removes is
        # boilerplate, and that matters much more now that term frequency survives
        # cleaning: on deadspin.com the legacy extractor yields 264 gambling tokens
        # (7.2% of the page) against trafilatura's 7 (1.1%). Under the old deduplicating
        # cleaner both collapsed to one; with frequency restored, 264 would dominate --
        # which is how a sports site came to be classified `gamble` at 0.98.
        #
        # It also fits the model. At max_length=256 legacy's extra ~1,100 words are
        # truncated away regardless, and truncation keeps the *first* 256 tokens: nav,
        # header and cookie banners. trafilatura's median fits with nothing dropped.
        "extractor": "trafilatura",
        # "minimal" | "legacy". Minimal extracts, collapses whitespace and lowercases,
        # and nothing else. Legacy is what shipped through v0.11.0: it deduplicated
        # tokens twice, sorted them alphabetically and stripped every non-ASCII
        # character, so the model saw an alphabetised set of words with no term
        # frequency, no order and no non-Latin script -- 73% of words discarded across
        # 14 real pages. Harmless for the 2022 bag-of-embeddings model, actively wrong
        # for a contextual multilingual encoder. Kept so v0.11.0 stays reproducible.
        "text_cleaning": "minimal",
        # Legacy-only. Keep only words in NLTK's `words` corpus. Off, because it discards
        # 39.8% of tokens on *English* pages -- it is a Webster's-era list with
        # no brand names and no inflected forms, so bbc.com loses `america`,
        # `american`, `accuses` and `acclaimed` along with `afrique`. It also
        # made the model multilingual in name only. Kept as a flag so v0.8.0's
        # numbers can be reproduced.
        "filter_non_english": False,
        # Parallel processing
        "max_parallel": 4,
        # Archive.org settings. Rate limiting, retries and backoff are handled
        # by the wayback session rather than hand-rolled sleeps.
        "archive_max_parallel": 2,  # Concurrent snapshot fetches
        "archive_window_days": 365,  # How far from the target date to search
        "archive_search_rate": 1,  # CDX calls per second
        "archive_memento_rate": 4,  # Memento (content) calls per second
        "archive_retries": 3,  # Retries before giving up
        "archive_backoff": 2,  # Exponential backoff base, seconds
        "archive_render_settle_ms": 1500,  # Settle before an archived screenshot
        "archive_screenshot_timeout": 15000,  # Screenshot timeout, ms
        # Recover bot-walled pages from archive.org. DataDome and Cloudflare
        # fingerprint headless Chromium itself, so this is the only way to get
        # those pages without evading anyone.
        "archive_fallback": True,
        # How stale a capture may be and still stand in for the live page.
        # Corpus builds can widen this; the live API should not.
        "archive_max_age_days": 365,
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
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
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

    def __init__(self, config_dict: dict[str, Any] | None = None):
        """Initialize configuration.

        Args:
            config_dict: Optional configuration overrides
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
            "PIEDOMAINS_EXTRACTOR": ("extractor", str),
            "PIEDOMAINS_FILTER_NON_ENGLISH": (
                "filter_non_english",
                lambda x: x.lower() in ("true", "1", "yes"),
            ),
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

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Any: Configuration value, or ``default`` when the key is absent.
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value

    def update(self, config_dict: dict[str, Any]):
        """Update multiple configuration values.

        Args:
            config_dict: Configuration updates
        """
        self._config.update(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            dict[str, Any]: Configuration dictionary
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
    """Get global configuration instance.

    Returns:
        Config: Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """Set global configuration instance.

    Args:
        config: Configuration instance to set as global
    """
    global _global_config
    _global_config = config


def configure(**kwargs: Any):
    """Configure global settings.

    Args:
        **kwargs: Configuration key-value pairs
    """
    config = get_config()
    config.update(kwargs)
