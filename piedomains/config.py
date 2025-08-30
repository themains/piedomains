"""
Configuration management for piedomains.
"""

import os
from typing import Dict, Any


class Config:
    """Configuration class for piedomains settings."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        # Network timeouts
        "http_timeout": 10,
        "webdriver_timeout": 30,
        "page_load_timeout": 30,
        
        # Retry settings
        "max_retries": 3,
        "retry_delay": 1,  # seconds
        
        # WebDriver settings
        "screenshot_wait_time": 5,  # seconds
        "webdriver_window_size": "1280,1024",
        
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
        
        # Logging
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    
    def __init__(self, config_dict: Dict[str, Any] = None):
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
        }
        
        for env_var, (config_key, type_func) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    self._config[config_key] = type_func(env_value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {env_value}. Using default. Error: {e}")
    
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
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key (str): Configuration key
            value (Any): Configuration value
        """
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """
        Update multiple configuration values.
        
        Args:
            config_dict (Dict[str, Any]): Configuration updates
        """
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
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