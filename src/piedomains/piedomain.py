"""Legacy prediction engine.

Only the static URL/domain validators are used in production; the extraction
and inference helpers are superseded by ``fetchers``, ``text`` and ``image``.
"""

import os
import re
import string
from urllib.parse import urlparse

from bs4 import BeautifulSoup

# from selenium import webdriver  # Removed - using Playwright now
from .base import Base
from .constants import most_common_words
from .piedomains_logging import get_logger

logger = get_logger()

# NLTK corpora live in text_processor, which is the cleaner the live pipeline
# uses. This module previously kept a second copy with a *different* failure
# fallback -- a 30-word stoplist here versus an empty set there -- so when NLTK
# was unavailable the two cleaners silently produced different text. These names
# are kept because the legacy validators and their tests still read them.
words = None
stop_words = None


def _initialize_nltk():
    """Initialize NLTK data by delegating to the live text processor.

    Mirrors ``text_processor``'s module globals into this module so the two
    cleaners can never diverge under an NLTK failure.
    """
    global words, stop_words

    from . import text_processor

    text_processor._initialize_nltk()
    words = text_processor.words
    stop_words = text_processor.stop_words


"""
    Piedomain class
    This class is used to predict the category of a given url
"""


class Piedomain(Base):
    """Legacy engine retained for its input validators."""

    MODELFN = "model/shallalist"
    model_file_name = "shallalist_v5_model.tar.gz"
    weights_loaded = False
    img_width = 254
    img_height = 254

    @staticmethod
    def parse_url_to_domain(url: str) -> str:
        """Extract domain name from a URL.

        Args:
            url: Full URL or domain name

        Returns:
            str: Domain name extracted from URL
        """
        if not url or not isinstance(url, str):
            return url

        # If it's already just a domain (no protocol), return as-is
        if not url.startswith(("http://", "https://")):
            # Check if it looks like a URL with path but no protocol
            if "/" in url and "." in url.split("/")[0]:
                return url.split("/")[0]
            return url

        # Parse full URL to extract domain
        parsed = urlparse(url)
        return parsed.netloc

    @staticmethod
    def validate_url_or_domain(url_or_domain: str) -> bool:
        """Validate if input is a valid URL or domain name.

        Args:
            url_or_domain: URL or domain name to validate

        Returns:
            bool: True if valid URL or domain, False otherwise
        """
        if not url_or_domain or not isinstance(url_or_domain, str):
            return False

        # Extract domain part for validation
        domain = Piedomain.parse_url_to_domain(url_or_domain)
        return Piedomain.validate_domain_name(domain)

    @staticmethod
    def validate_domain_name(domain: str) -> bool:
        """Validate if a domain name is properly formatted.

        Args:
            domain: Domain name to validate

        Returns:
            bool: True if domain is valid, False otherwise
        """
        if not domain or not isinstance(domain, str):
            return False

        # Remove protocol if present
        if domain.startswith(("http://", "https://")):
            parsed = urlparse(domain)
            domain = parsed.netloc

        # Remove trailing slash and path
        domain = domain.split("/")[0]

        # Check for invalid characters (spaces, special chars except hyphen and dot)
        if " " in domain or any(c in domain for c in "!@#$%^&*()+=[]{}|\\:\";'<>?/"):
            return False

        # Must contain at least one dot to be a valid domain
        if "." not in domain:
            return False

        # Check for consecutive dots
        if ".." in domain:
            return False

        # Cannot start or end with dot or hyphen
        if (
            domain.startswith(".")
            or domain.endswith(".")
            or domain.startswith("-")
            or domain.endswith("-")
        ):
            return False

        # Check length
        if len(domain) > 253:
            return False

        # Validate each part of the domain
        parts = domain.split(".")
        for part in parts:
            if not part or len(part) > 63:
                return False
            if part.startswith("-") or part.endswith("-"):
                return False
            if not re.match(r"^[a-zA-Z0-9\-]+$", part):
                return False

        return True

    @classmethod
    def validate_domains(cls, domains: list) -> tuple[list, list]:
        """Validate a list of domain names and separate valid from invalid.

        Args:
            domains: List of domain names to validate

        Returns:
            tuple: (valid_domains, invalid_domains)
        """
        valid_domains = []
        invalid_domains = []

        for domain in domains:
            if cls.validate_domain_name(domain):
                # Normalize domain (remove protocol, trailing slash, etc.)
                if domain.startswith(("http://", "https://")):
                    parsed = urlparse(domain)
                    domain = parsed.netloc
                domain = domain.split("/")[0]
                valid_domains.append(domain)
            else:
                invalid_domains.append(domain)

        return valid_domains, invalid_domains

    @classmethod
    def validate_urls_or_domains(cls, urls_or_domains: list) -> tuple[list, list, dict]:
        """Validate a list of URLs or domains and separate valid from invalid.

        Args:
            urls_or_domains: List of URLs or domain names to validate

        Returns:
            tuple: (valid_inputs, invalid_inputs, url_to_domain_map)
        """
        valid_inputs = []
        invalid_inputs = []
        url_to_domain_map = {}

        for input_item in urls_or_domains:
            if cls.validate_url_or_domain(input_item):
                domain = cls.parse_url_to_domain(input_item)
                valid_inputs.append(input_item)
                url_to_domain_map[input_item] = domain
            else:
                invalid_inputs.append(input_item)

        return valid_inputs, invalid_inputs, url_to_domain_map

    @classmethod
    def text_from_html(cls, text: str) -> str:
        """Extract clean text content from HTML.

        Args:
            text: Raw HTML content

        Returns:
            str: Cleaned text with unique lowercase words
        """
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        result = " ".join(
            list({t.lower().strip() for t in text.split() if t.strip().isalpha()})
        )
        return result

    @classmethod
    def data_cleanup(cls, s: str) -> str:
        """Clean and normalize text data for model input.

        Args:
            s: Raw text string

        Returns:
            str: Cleaned text with English words only, no stopwords or common terms


        Raises:
            AttributeError: If the input is not of the expected type.
        """
        if not isinstance(s, str):
            raise AttributeError("Input must be a string")

        # Initialize NLTK data if needed
        _initialize_nltk()

        # remove numbers
        s = re.sub(r"\d+", "", s)
        # remove duplicates
        tokens = list(set(s.split()))
        # remove punctuation from each token
        table = str.maketrans("", "", string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove non alpha first
        tokens = [w.lower() for w in tokens if w.isalpha()]
        # remove non ascii
        tokens = [w.lower() for w in tokens if w.isascii()]
        # remove non english words (only if words corpus is available)
        if words:
            tokens = [w for w in tokens if w in words]
        # filter out stop words (only if the corpus is available)
        if stop_words:
            tokens = [w for w in tokens if w not in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        # remove most common words
        tokens = [w for w in tokens if w not in most_common_words]
        return " ".join(w for w in tokens)

    @classmethod
    def validate_input(cls, input: list, path: str, type: str) -> bool:
        """Validate input parameters for prediction functions.

        Args:
            input: List of URLs or domain names
            path: Path to HTML or image files
            type: Input type - 'html' or 'image'

        Returns:
            bool: True if operating in offline mode (using local files only)

        Raises:
            Exception: If neither URLs/domains nor valid path provided
        """
        pth = "html_path" if type == "html" else "image_path"

        offline = False
        # if input is empty
        if len(input) == 0:
            # if path is None, raise exception
            if path is None:
                raise Exception(
                    f"Provide list of Domains, or for offline provide {pth}"
                )
            else:
                # if path is not None, check if it exists and is not empty
                if not os.path.exists(path):
                    raise Exception(f"{path} does not exist")
                if len(os.listdir(path)) == 0:
                    raise Exception(f"{path} is empty")
                else:
                    offline = True
        return offline
