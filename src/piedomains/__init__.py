"""Piedomains: Domain content classification library.

This module provides lazy imports to avoid dependency issues when
optional dependencies (like playwright) are not installed.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # The git tag is the version; uv-dynamic-versioning bakes it into the
    # distribution metadata at build time. No version string lives in source.
    __version__ = version("piedomains")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"


def __getattr__(name):
    """Lazy import handler for piedomains modules."""
    match name:
        case "DomainClassifier":
            from .api import DomainClassifier

            return DomainClassifier
        case "classify_domains":
            from .api import classify_domains

            return classify_domains
        case "DataCollector":
            from .data_collector import DataCollector

            return DataCollector
        case "TextClassifier":
            from .text import TextClassifier

            return TextClassifier
        case "ImageClassifier":
            from .image import ImageClassifier

            return ImageClassifier
        case "LLMClassifier":
            from .llm_classifier import LLMClassifier

            return LLMClassifier
        case "LLMConfig":
            from .llm.config import LLMConfig

            return LLMConfig
        case _:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "DataCollector",
    "DomainClassifier",
    "ImageClassifier",
    "LLMClassifier",
    "LLMConfig",
    "TextClassifier",
    "classify_domains",
]
