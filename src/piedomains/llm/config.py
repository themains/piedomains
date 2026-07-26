"""LLM configuration for domain classification."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..piedomains_logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM-based classification.

    Attributes:
        provider: LLM provider (e.g., 'openai', 'anthropic', 'google')
        model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro')
        api_key: API key for the provider
        base_url: Optional base URL for custom endpoints
        max_tokens: Maximum tokens for response
        temperature: Temperature for response generation
        categories: List of classification categories
        cost_limit_usd: Maximum cost limit in USD
        usage_tracking: Whether to track API usage
    """

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 500
    temperature: float = 0.1
    categories: list[str] | None = None
    cost_limit_usd: float = 10.0
    usage_tracking: bool = True

    def __post_init__(self) -> None:
        """Validate and set defaults after initialization."""
        # Auto-detect API key from environment if not provided
        if self.api_key is None:
            self.api_key = self._get_api_key_from_env()

        # Set default categories if none provided
        if self.categories is None:
            self.categories = self._get_default_categories()

        # Validate configuration
        self._validate()

    def _get_api_key_from_env(self) -> str | None:
        """Get API key from environment variables."""
        env_vars = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "azure": ["AZURE_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
            "together": ["TOGETHER_API_KEY"],
            "replicate": ["REPLICATE_API_TOKEN"],
        }

        for var in env_vars.get(self.provider, []):
            if key := os.getenv(var):
                return key

        # Generic fallback
        return os.getenv("LLM_API_KEY")

    def _get_default_categories(self) -> list[str]:
        """Get default classification categories."""
        return [
            "news",
            "shopping",
            "social",
            "educational",
            "entertainment",
            "technology",
            "finance",
            "health",
            "government",
            "sports",
            "travel",
            "food",
            "business",
            "science",
            "politics",
        ]

    def _validate(self) -> None:
        """Validate the configuration."""
        if not self.provider:
            raise ValueError("Provider is required")

        if not self.model:
            raise ValueError("Model is required")

        if not self.api_key:
            logger.warning(
                f"No API key found for provider '{self.provider}'. "
                "Set via parameter or environment variable."
            )

        if not self.categories:
            raise ValueError("Categories list cannot be empty")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    def to_litellm_params(self) -> dict[str, Any]:
        """Convert to litellm parameters."""
        params = {
            "model": f"{self.provider}/{self.model}",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        if self.base_url:
            params["api_base"] = self.base_url

        return params

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> LLMConfig:
        """Create LLMConfig from dictionary."""
        return cls(**config_dict)
