"""
HTTP client with connection pooling and session management for improved performance.
"""

import requests
import time
from typing import Dict, Optional, Any
from contextlib import contextmanager
from .config import get_config
from .logging import get_logger

logger = get_logger()


class PooledHTTPClient:
    """HTTP client with connection pooling and session reuse."""
    
    def __init__(self):
        self._session = None
        self._config = get_config()
    
    @property 
    def session(self) -> requests.Session:
        """Get or create HTTP session with connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,  # Number of connection pools
                pool_maxsize=20,      # Max connections per pool
                max_retries=0         # We handle retries manually
            )
            adapter.config.update({
                "pool_connections": 10,
                "pool_maxsize": 20,
            })
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
            
            # Set default headers
            self._session.headers.update({
                "User-Agent": self._config.user_agent,
                "Accept-Language": "en-US,en;q=0.9"
            })
            
            logger.debug("Created HTTP session with connection pooling")
            
        return self._session
    
    def get(self, url: str, timeout: Optional[float] = None, **kwargs) -> requests.Response:
        """
        Perform HTTP GET with retry logic and connection pooling.
        
        Args:
            url (str): URL to fetch
            timeout (float): Request timeout (uses config default if None)
            **kwargs: Additional arguments passed to requests.get
            
        Returns:
            requests.Response: HTTP response
            
        Raises:
            requests.exceptions.RequestException: On final failure after retries
        """
        if timeout is None:
            timeout = self._config.http_timeout
            
        last_exception = None
        
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self.session.get(
                    url, 
                    timeout=timeout, 
                    allow_redirects=True,
                    **kwargs
                )
                response.raise_for_status()
                return response
                
            except (requests.exceptions.RequestException, IOError) as e:
                last_exception = e
                if attempt < self._config.max_retries:
                    wait_time = self._config.retry_delay * (2 ** attempt)
                    logger.debug(f"Retrying HTTP GET for {url} in {wait_time}s (attempt {attempt + 1}/{self._config.max_retries + 1})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP GET failed for {url} after {self._config.max_retries + 1} attempts: {e}")
                    raise last_exception
    
    def close(self):
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
            logger.debug("HTTP session closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Global instance for reuse across the module
_global_client = None


@contextmanager
def http_client():
    """
    Context manager for getting a pooled HTTP client.
    
    Yields:
        PooledHTTPClient: HTTP client with connection pooling
    """
    global _global_client
    
    if _global_client is None:
        _global_client = PooledHTTPClient()
    
    try:
        yield _global_client
    except Exception:
        # On error, close and recreate client
        if _global_client:
            _global_client.close()
            _global_client = None
        raise


def get_http_client() -> PooledHTTPClient:
    """
    Get the global HTTP client instance.
    
    Returns:
        PooledHTTPClient: Global HTTP client with connection pooling
    """
    global _global_client
    
    if _global_client is None:
        _global_client = PooledHTTPClient()
    
    return _global_client


def close_global_client():
    """Close the global HTTP client."""
    global _global_client
    
    if _global_client:
        _global_client.close()
        _global_client = None