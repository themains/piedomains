#!/usr/bin/env python3
"""Text processing utilities for domain content analysis.

Handles HTML content extraction, text cleaning, and preprocessing.
"""

import re
import string

from bs4 import BeautifulSoup
from bs4.element import Comment

from .constants import most_common_words
from .piedomains_logging import get_logger

logger = get_logger()

# Global variables for NLTK data - will be initialized when needed
words = None
stop_words = None


def _initialize_nltk():
    """Initialize NLTK data with proper error handling."""
    global words, stop_words

    if words is not None and stop_words is not None:
        return  # Already initialized

    try:
        import nltk  # pyright: ignore[reportMissingImports]

        # Download required NLTK data
        nltk.download("stopwords", quiet=True)
        nltk.download("words", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)

        # Import and initialize corpora
        from nltk.corpus import (  # pyright: ignore[reportMissingImports]
            stopwords,
        )

        words = set(nltk.corpus.words.words())
        stop_words = set(stopwords.words("english"))

    except Exception as e:
        logger.warning(f"NLTK initialization failed: {e}")
        # Fallback to basic word sets if NLTK fails
        words = set()
        stop_words = set()


class TextProcessor:
    """Handles text extraction and cleaning from HTML content."""

    @staticmethod
    def extract_text_from_html(html_content: str) -> str:
        """Extract clean, visible text from HTML content.

        Args:
            html_content: Raw HTML content

        Returns:
            str: Cleaned visible text content
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text and filter out comments
            texts = soup.find_all(string=True)
            visible_texts = []

            for element in texts:
                parent = element.parent
                if (
                    parent is not None
                    and parent.name
                    not in [
                        "style",
                        "script",
                        "meta",
                        "title",
                    ]
                    and not isinstance(element, Comment)
                ):
                    text = element.strip()
                    if text:
                        visible_texts.append(text)

            return " ".join(visible_texts)

        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""

    @staticmethod
    def clean_and_normalize_text(text: str) -> str:
        """Clean and normalize text data for model input.

        Removes numbers, punctuation, stopwords and common terms. Dictionary
        filtering is available but off by default -- see the note below, and
        ``filter_non_english`` in :mod:`piedomains.config`.

        Args:
            text: Raw text to clean

        Returns:
            str: Cleaned, deduplicated tokens joined by spaces.


        Raises:
            AttributeError: If the input is not of the expected type.
        """
        if not isinstance(text, str):
            raise AttributeError("Input must be a string")

        # Initialize NLTK data if needed
        _initialize_nltk()

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Split into tokens and remove duplicates
        tokens = list(set(text.split()))

        # Remove punctuation from each token
        table = str.maketrans("", "", string.punctuation)
        tokens = [w.translate(table) for w in tokens]

        # Convert to lowercase and filter alphabetic only
        tokens = [w.lower() for w in tokens if w.isalpha()]

        # Remove non-ASCII characters
        tokens = [w for w in tokens if w.isascii()]

        # Optional dictionary filter, off by default.
        #
        # This was the single biggest destroyer of signal in the pipeline. NLTK's
        # `words` corpus is a Webster's-era dictionary with no brand names and no
        # inflected forms, so it drops exactly the most discriminative tokens --
        # `spotify`, `quora`, `facebook`, `instagram`, plus ordinary web
        # vocabulary like `download`, `email`, `employers`, `cookies`. Measured
        # across 20 evaluation pages it discards 39.8% of all tokens and pushes
        # 2 of them under the token floor, on *English* pages; `bbc.com` alone
        # loses `america`, `american`, `accuses` and `acclaimed`.
        #
        # It also made a multilingual encoder pointless by construction, since
        # anything not in an English dictionary was removed before the model saw
        # it. Kept behind the flag so v0.8.0's numbers stay reproducible.
        from .config import get_config

        if words and get_config().get("filter_non_english", False):
            tokens = [w for w in tokens if w in words]

        # Filter out stop words
        if stop_words:
            tokens = [w for w in tokens if w not in stop_words]

        # Filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]

        # Remove most common generic words
        tokens = [w for w in tokens if w not in most_common_words]

        # Remove duplicates again and sort for consistency
        tokens = sorted(set(tokens))

        return " ".join(tokens)

    @classmethod
    def process_html_to_text(cls, html_content: str) -> str:
        """Complete pipeline: extract text from HTML and clean it.

        Args:
            html_content: Raw HTML content

        Returns:
            str: Clean, processed text ready for model input
        """
        raw_text = cls.extract_text(html_content)
        return cls.clean_and_normalize_text(raw_text)

    @staticmethod
    def extract_with_trafilatura(html_content: str) -> str:
        """Extract main content using trafilatura.

        trafilatura powers FineWeb and RefinedWeb and tops the WCXB benchmark on
        articles. It is **not** the default here, because measured on this
        project's own pages it is worse: 16 of 33 fall under the token floor
        against 14 of 33 for the legacy cleaner. It discards navigation and tile
        text as boilerplate, which is correct for an article and wrong for a
        homepage, where that chrome carries most of the site-level signal.

        Kept reachable because the corpus work in PR 5 classifies deep pages as
        well as homepages, where the trade-off reverses.

        Args:
            html_content: Raw HTML content.

        Returns:
            str: Extracted main content, or ``""`` when nothing was found.
        """
        # trafilatura ships py.typed but pyright does not resolve it from this
        # lazily-imported position; the runtime import is exercised by tests.
        import trafilatura  # pyright: ignore[reportMissingImports]

        try:
            return (
                trafilatura.extract(
                    html_content,
                    include_comments=False,
                    include_tables=True,
                    favor_recall=True,
                )
                or ""
            )
        except Exception as e:  # never let an extractor failure kill a fetch
            logger.warning(f"trafilatura extraction failed: {e}")
            return ""

    @classmethod
    def extract_text(cls, html_content: str) -> str:
        """Extract page text using the configured extractor.

        Defaults to the legacy visible-text walk, because the shipped TensorFlow
        model was *trained* on its output -- switching wholesale would break
        train/serve parity. Set ``extractor = "trafilatura"`` in config (or train
        a model on its output) to use the better one.

        Args:
            html_content: Raw HTML content.

        Returns:
            str: Extracted text, falling back to the legacy walk if the
            configured extractor returns nothing.
        """
        from .config import get_config

        which = get_config().get("extractor", "legacy")
        if which == "trafilatura":
            text = cls.extract_with_trafilatura(html_content)
            # Fall back rather than lose the page: trafilatura is precision-first
            # and returns nothing on some layouts the legacy walk still handles.
            return text or cls.extract_text_from_html(html_content)
        return cls.extract_text_from_html(html_content)
