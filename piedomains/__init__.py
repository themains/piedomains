"""
Piedomains: Domain content classification library.

This module provides lazy imports to avoid dependency issues when
optional dependencies (like playwright) are not installed.
"""


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
    "DomainClassifier",
    "DataCollector",
    "classify_domains",
    "LLMConfig",
    "TextClassifier",
    "ImageClassifier",
    "LLMClassifier",
]
