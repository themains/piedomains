#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for URL support functionality.
"""

import unittest
from piedomains.piedomain import Piedomain


class TestURLSupport(unittest.TestCase):
    """Test URL parsing and validation functions."""

    def test_parse_url_to_domain(self):
        """Test URL to domain parsing."""
        test_cases = [
            ("https://example.com", "example.com"),
            ("http://sub.example.org", "sub.example.org"),
            ("https://example.com/path/to/page", "example.com"),
            ("https://example.com:8080/path", "example.com:8080"),
            ("example.com", "example.com"),
            ("sub.example.com/path", "sub.example.com"),
            ("just-a-domain.com", "just-a-domain.com"),
        ]

        for url, expected_domain in test_cases:
            with self.subTest(url=url):
                result = Piedomain.parse_url_to_domain(url)
                self.assertEqual(result, expected_domain)

    def test_parse_url_to_domain_edge_cases(self):
        """Test URL parsing edge cases."""
        edge_cases = [
            ("", ""),
            (None, None),
            ("https://", ""),
            ("just-text-no-domain", "just-text-no-domain"),
        ]

        for url, expected in edge_cases:
            with self.subTest(url=url):
                result = Piedomain.parse_url_to_domain(url)
                self.assertEqual(result, expected)

    def test_validate_url_or_domain(self):
        """Test URL and domain validation."""
        valid_inputs = [
            "google.com",
            "https://example.org",
            "http://sub.example.com",
            "https://test.co.uk/path/to/page",
            "example.net/",
        ]

        for input_item in valid_inputs:
            with self.subTest(input=input_item):
                self.assertTrue(Piedomain.validate_url_or_domain(input_item))

    def test_validate_url_or_domain_invalid(self):
        """Test invalid URL and domain validation."""
        invalid_inputs = [
            "",
            None,
            "invalid..domain.com",
            "https://spaces in domain.com",
            "just-text-without-dot",
            "https://special!chars@domain.com",
        ]

        for input_item in invalid_inputs:
            with self.subTest(input=input_item):
                self.assertFalse(Piedomain.validate_url_or_domain(input_item))

    def test_validate_urls_or_domains(self):
        """Test batch URL/domain validation."""
        mixed_inputs = [
            "google.com",
            "https://example.org/path",
            "invalid..domain",
            "https://facebook.com",
            "",
            "http://twitter.com/user"
        ]

        valid, invalid, url_map = Piedomain.validate_urls_or_domains(mixed_inputs)

        self.assertEqual(len(valid), 4)
        self.assertEqual(len(invalid), 2)
        self.assertIn("google.com", valid)
        self.assertIn("https://example.org/path", valid)
        self.assertIn("https://facebook.com", valid)
        self.assertIn("http://twitter.com/user", valid)
        self.assertIn("invalid..domain", invalid)
        self.assertIn("", invalid)

        # Check URL to domain mapping
        self.assertEqual(url_map["google.com"], "google.com")
        self.assertEqual(url_map["https://example.org/path"], "example.org")
        self.assertEqual(url_map["https://facebook.com"], "facebook.com")
        self.assertEqual(url_map["http://twitter.com/user"], "twitter.com")


if __name__ == "__main__":
    unittest.main()

