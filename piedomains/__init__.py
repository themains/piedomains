"""
Piedomains: Domain content classification library.

This module provides lazy imports to avoid dependency issues when
optional dependencies (like playwright) are not installed.
"""


def __getattr__(name):
    """Lazy import handler for piedomains modules."""
    if name == "DomainClassifier":
        from .api import DomainClassifier

        return DomainClassifier
    elif name == "classify_domains":
        from .api import classify_domains

        return classify_domains
    elif name == "DataCollector":
        from .data_collector import DataCollector

        return DataCollector
    elif name == "TextClassifier":
        from .text import TextClassifier

        return TextClassifier
    elif name == "ImageClassifier":
        from .image import ImageClassifier

        return ImageClassifier
    elif name == "LLMClassifier":
        from .llm_classifier import LLMClassifier

        return LLMClassifier
    elif name == "LLMConfig":
        from .llm.config import LLMConfig

        return LLMConfig
    else:
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
