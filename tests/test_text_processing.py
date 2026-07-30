"""
Test text processing and HTML parsing functionality.
"""

import unittest

from piedomains.piedomain import Piedomain
from piedomains.text_processor import TextProcessor


class TestTextProcessing(unittest.TestCase):
    """Test text extraction and cleaning functions."""

    def test_text_from_html_basic(self):
        """Test basic HTML text extraction."""
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Welcome to Test Site</h1>
            <p>This is a test paragraph with some content.</p>
            <div>Additional content here</div>
        </body>
        </html>
        """

        result = Piedomain.text_from_html(html_content)

        self.assertIsInstance(result, str)
        self.assertIn("welcome", result.lower())
        self.assertIn("test", result.lower())
        self.assertIn("content", result.lower())

    def test_text_from_html_with_scripts_and_styles(self):
        """Test HTML text extraction ignoring scripts and styles."""
        html_content = """
        <html>
        <head>
            <title>Test</title>
            <script>var x = 1;</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <h1>Main Content</h1>
            <script>alert('hello');</script>
        </body>
        </html>
        """

        result = Piedomain.text_from_html(html_content)

        # Should extract text but not script/style content
        self.assertIn("main", result.lower())
        self.assertIn("content", result.lower())
        # Scripts and styles should be ignored by BeautifulSoup's get_text()

    def test_data_cleanup_removes_numbers(self):
        """Test that data cleanup removes numbers."""
        text_with_numbers = "test123 content456 more789 text"
        result = Piedomain.data_cleanup(text_with_numbers)

        self.assertNotIn("123", result)
        self.assertNotIn("456", result)
        self.assertNotIn("789", result)

    def test_data_cleanup_removes_punctuation(self):
        """Test that data cleanup removes punctuation."""
        text_with_punct = "hello, world! this is a test."
        result = Piedomain.data_cleanup(text_with_punct)

        self.assertNotIn(",", result)
        self.assertNotIn("!", result)
        self.assertNotIn(".", result)

    def test_data_cleanup_removes_stopwords(self):
        """Test that data cleanup removes English stopwords."""
        text_with_stopwords = "the quick brown fox and jumps in the lazy dog"
        result = Piedomain.data_cleanup(text_with_stopwords)

        # Common stopwords should be removed
        self.assertNotIn(" the ", " " + result + " ")
        self.assertNotIn(" and ", " " + result + " ")
        self.assertNotIn(" in ", " " + result + " ")
        # Content words should remain
        self.assertIn("quick", result)
        self.assertIn("brown", result)
        self.assertIn("fox", result)

    def test_data_cleanup_removes_short_words(self):
        """Test that data cleanup removes single character words."""
        text_with_short = "a big test i o u"
        result = Piedomain.data_cleanup(text_with_short)

        # Single characters should be removed
        result_words = result.split()
        for word in result_words:
            self.assertGreater(len(word), 1)

    def test_data_cleanup_lowercase(self):
        """Test that data cleanup converts to lowercase."""
        text_mixed_case = "This IS Mixed CASE Text"
        result = Piedomain.data_cleanup(text_mixed_case)

        self.assertEqual(result, result.lower())

    def test_data_cleanup_removes_duplicates(self):
        """Test that data cleanup removes duplicate words."""
        text_with_duplicates = "test test content content more test"
        result = Piedomain.data_cleanup(text_with_duplicates)

        words = result.split()
        unique_words = set(words)

        # Should have same number of unique words as total words
        self.assertEqual(len(words), len(unique_words))

    def test_data_cleanup_filters_non_english(self):
        """Test that data cleanup attempts to filter non-English words."""
        # This test may be limited by the NLTK words corpus availability
        text_mixed = "computer test français deutsche invalid"
        result = Piedomain.data_cleanup(text_mixed)

        # Should contain recognizable English words that aren't stopwords
        self.assertIn("computer", result)
        self.assertIn("test", result)

    def test_data_cleanup_empty_input(self):
        """Test data cleanup with empty input."""
        result = Piedomain.data_cleanup("")
        self.assertEqual(result, "")

    def test_data_cleanup_only_numbers_and_punct(self):
        """Test data cleanup with only numbers and punctuation."""
        result = Piedomain.data_cleanup("123!@#456$%^")
        self.assertEqual(result, "")


class TestDictionaryFilterDefault(unittest.TestCase):
    """The NLTK dictionary filter must stay off unless asked for.

    It discards 39.8% of tokens on *English* pages -- NLTK's `words` corpus is a
    Webster's-era list with no brand names and no inflected forms, so it drops
    `spotify`, `facebook`, `download` and `email` along with every non-English
    word. Leaving it on also made the multilingual encoder pointless by
    construction. The shipped model is trained with it off, so a silent flip
    back would be a train/serve skew, not just a quality regression.
    """

    def test_off_by_default(self):
        from piedomains.config import get_config

        self.assertFalse(get_config().get("filter_non_english"))

    def test_brand_names_survive(self):
        """The tokens that identify a site are exactly what the filter ate."""
        from piedomains.text_processor import TextProcessor

        text = "spotify facebook instagram download email"
        result = TextProcessor.clean_and_normalize_text(text)
        for token in ("spotify", "facebook", "instagram", "download", "email"):
            self.assertIn(token, result, f"{token!r} was dropped")

    def test_flag_still_restores_the_old_behaviour(self):
        """v0.8.0's numbers have to stay reproducible.

        Both flags are needed now: the dictionary filter is part of the legacy cleaner,
        and minimal cleaning ignores it entirely. That is the point of minimal cleaning.
        """
        from piedomains.config import get_config
        from piedomains.text_processor import TextProcessor

        config = get_config()
        original = (config.get("filter_non_english"), config.get("text_cleaning"))
        try:
            config.set("text_cleaning", "legacy")
            config.set("filter_non_english", True)
            result = TextProcessor.clean_and_normalize_text("spotify computer")
            self.assertIn("computer", result)
            self.assertNotIn("spotify", result)
        finally:
            config.set("filter_non_english", original[0])
            config.set("text_cleaning", original[1])

    def test_minimal_cleaning_ignores_the_dictionary_filter(self):
        """It discarded 39.8% of tokens on English pages; minimal must not honour it."""
        from piedomains.config import get_config
        from piedomains.text_processor import TextProcessor

        config = get_config()
        original = (config.get("filter_non_english"), config.get("text_cleaning"))
        try:
            config.set("text_cleaning", "minimal")
            config.set("filter_non_english", True)
            self.assertIn(
                "spotify", TextProcessor.clean_and_normalize_text("spotify computer")
            )
        finally:
            config.set("filter_non_english", original[0])
            config.set("text_cleaning", original[1])


if __name__ == "__main__":
    unittest.main()


class TestMinimalCleaning(unittest.TestCase):
    """The representation the model actually receives.

    Until v0.12.0 the cleaner deduplicated twice, sorted alphabetically and dropped every
    non-ASCII character, so a contextual multilingual encoder was fed an alphabetised set
    of ASCII words. Across 14 real cached pages that discarded 73% of all words. These
    tests pin each defect so none can come back.
    """

    def setUp(self):
        from piedomains.config import get_config

        self.config = get_config()
        self.original = self.config.get("text_cleaning")
        self.config.set("text_cleaning", "minimal")

    def tearDown(self):
        self.config.set("text_cleaning", self.original)

    def test_term_frequency_survives(self):
        """One sportsbook advert must not weigh as much as the whole page.

        This is the mechanism behind deadspin.com being classified `gamble` at 0.98:
        with deduplication, 200 mentions of sport and 3 of casino were both worth 1.
        """
        from collections import Counter

        page = (
            "<html><body>" + "sports " * 200 + "casino bet odds " * 3 + "</body></html>"
        )
        counts = Counter(TextProcessor.process_html_to_text(page).split())
        self.assertEqual(counts["sports"], 200)
        self.assertEqual(counts["casino"], 3)

    def test_word_order_survives(self):
        """Sorting alphabetically nullifies the reason to use a contextual encoder."""
        page = "<html><body>zebra apple monkey banana</body></html>"
        tokens = TextProcessor.process_html_to_text(page).split()
        self.assertEqual(tokens, ["zebra", "apple", "monkey", "banana"])
        self.assertNotEqual(tokens, sorted(tokens))

    def test_non_latin_scripts_survive(self):
        """mmBERT is multilingual; an ASCII filter made that true in name only."""
        for label, body in (
            ("japanese", "ニュース 速報 経済"),
            ("cyrillic", "новости спорт погода"),
            ("arabic", "أخبار رياضة طقس"),
        ):
            with self.subTest(script=label):
                out = TextProcessor.process_html_to_text(
                    f"<html><body>{body}</body></html>"
                )
                self.assertTrue(out.strip(), f"{label} was stripped")

    def test_legacy_is_still_reachable(self):
        """v0.11.0 and earlier cannot be reproduced without it."""
        self.config.set("text_cleaning", "legacy")
        page = (
            "<html><body>" + "sports " * 200 + "casino bet odds " * 3 + "</body></html>"
        )
        tokens = TextProcessor.process_html_to_text(page).split()
        self.assertEqual(tokens, sorted(tokens))
        self.assertEqual(len(tokens), len(set(tokens)))
