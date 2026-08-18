"""Piedomains: Domain content classification library.

This module provides lazy imports to avoid dependency issues when
optional dependencies (like playwright) are not installed.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Declared for type checkers only; the runtime bindings come from the
    # __getattr__ below, which is what keeps TensorFlow and Playwright off the
    # import path until something actually asks for them. Without these,
    # pyright reports every name in __all__ as unsupported -- seven findings
    # for a module that works.
    from .api import DomainClassifier, classify_domains
    from .data_collector import DataCollector
    from .image import ImageClassifier
    from .llm.config import LLMConfig
    from .llm_classifier import LLMClassifier
    from .text import TextClassifier

try:
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
