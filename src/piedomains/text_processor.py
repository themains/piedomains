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

#: Characters that, alone, make a token carry no meaning. Kept as a frozenset because
#: this runs per token over the whole corpus.
_PUNCTUATION = frozenset(string.punctuation + "|\u2013\u2014\u2022\u00b7\u2026")

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
        """Normalise page text for the model.

        **What this used to do, and why it was wrong.** Until v0.12.0 this deduplicated
        tokens twice, sorted them alphabetically, and dropped every non-ASCII character.
        The model therefore received an alphabetised *set* of words -- the stored training
        text is literally ``"accueil adresse alfonso aller anciens animation ans
        archives..."``. Across 14 real cached pages that discarded **73% of all words**,
        and `asahi.com` kept 2.7% of its own.

        Three things were lost, and each matters more than it sounds:

        * **Term frequency.** 200 mentions of "sports" and one sportsbook advert in the
          footer weighed exactly the same. That is the mechanism behind `deadspin.com`
          being classified `gamble` at 0.98 confidence, and 13% of predictions on
          Tranco-top-100k domains landing in blockable categories.
        * **Word order**, to an alphabetical sort -- which nullifies the entire reason to
          use a contextual encoder.
        * **Every non-Latin script**, which made a *multilingual* model multilingual in
          name only.

        None of it was unreasonable for the model this pipeline was built for: a
        ``GlobalAveragePooling1D`` bag-of-embeddings is order-invariant by construction.
        The preprocessing simply was not revisited when the model became mmBERT.

        So this now extracts, collapses whitespace and lowercases. Nothing else. A
        transformer wants sentences.

        Set ``text_cleaning="legacy"`` in the config to restore the old behaviour, which
        is required to reproduce v0.11.0 and earlier.

        Args:
            text: Raw text to clean

        Returns:
            str: Normalised text, order and frequency intact.

        Raises:
            AttributeError: If the input is not of the expected type.
        """
        if not isinstance(text, str):
            raise AttributeError("Input must be a string")

        from .config import get_config

        if get_config().get("text_cleaning", "minimal") == "legacy":
            return TextProcessor._clean_legacy(text)

        # Collapse runs of whitespace so the tokenizer is not fed page layout, and
        # lowercase because the model is uncased.
        #
        # Standalone punctuation is *kept* by default, which is not what I expected.
        # Table pipes and layout dashes are 4.8% of tokens and 40% of the p99 page, so
        # dropping them looked obviously right. Trained both ways on otherwise identical
        # corpora, dropping them measured worse: Curlie agreement 0.543 -> 0.523,
        # held-out macro-F1 0.7267 -> 0.7134, and deadspin.com went back to `gamble`.
        # Structural punctuation evidently says something about what kind of page this is.
        #
        # `strip_punctuation=True` restores the other behaviour. Do not flip the default
        # without retraining: the model is fitted to whichever form it was shown.
        collapsed = re.sub(r"\s+", " ", text).strip().lower()
        if not get_config().get("strip_punctuation", False):
            return collapsed
        return " ".join(
            word
            for word in collapsed.split()
            if not all(c in _PUNCTUATION for c in word)
        )

    @staticmethod
    def _clean_legacy(text: str) -> str:
        """The pre-0.12.0 cleaner, kept so earlier numbers stay reproducible.

        Deduplicates, sorts alphabetically and strips non-ASCII. See
        :meth:`clean_and_normalize_text` for why none of that survives by default.

        Args:
            text: Raw text to clean

        Returns:
            str: Cleaned, deduplicated, alphabetically sorted tokens.
        """
        _initialize_nltk()

        text = re.sub(r"\d+", "", text)
        tokens = list(set(text.split()))

        table = str.maketrans("", "", string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        tokens = [w.lower() for w in tokens if w.isalpha()]
        tokens = [w for w in tokens if w.isascii()]

        from .config import get_config

        if words and get_config().get("filter_non_english", False):
            tokens = [w for w in tokens if w in words]
        if stop_words:
            tokens = [w for w in tokens if w not in stop_words]
        tokens = [word for word in tokens if len(word) > 1]
        tokens = [w for w in tokens if w not in most_common_words]

        return " ".join(sorted(set(tokens)))

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
        articles, and it is the default here — see ``config.py`` for the
        measurement. An earlier note in this file said the opposite, that it was
        worse at 16 of 33 pages under the token floor against the legacy
        cleaner's 14. That comparison was invalid: it counted trafilatura's raw
        words against the legacy path's *cleaned* tokens, and the legacy cleaner
        deduplicated, so the two sides were not the same quantity.

        What it does discard is navigation and tile chrome. That costs
        site-level signal on a homepage and removes a great deal of noise: on
        ``deadspin.com`` it cuts gambling tokens from 260 (7.1% of the page) to
        7 (1.1%), which only started to matter once the cleaner stopped
        collapsing repeated words to one.

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
