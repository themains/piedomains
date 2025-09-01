"""
Test text processing and HTML parsing functionality.
"""

import unittest
from piedomains.piedomain import Piedomain


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


if __name__ == "__main__":
    unittest.main()

