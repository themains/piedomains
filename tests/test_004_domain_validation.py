"""
Test domain validation functionality.
"""

import unittest
from piedomains.piedomain import Piedomain


class TestDomainValidation(unittest.TestCase):
    """Test domain name validation functions."""

    def test_validate_domain_name_valid(self):
        """Test valid domain names."""
        valid_domains = [
            "google.com",
            "sub.example.org",
            "test-domain.co.uk",
            "example123.net",
            "a.b",
            "very-long-subdomain.example-domain.com"
        ]

        for domain in valid_domains:
            with self.subTest(domain=domain):
                self.assertTrue(Piedomain.validate_domain_name(domain))

    def test_validate_domain_name_invalid(self):
        """Test invalid domain names."""
        invalid_domains = [
            "",
            None,
            "invalid..domain.com",
            ".invalid.com",
            "invalid-.com",
            "-invalid.com",
            "toolong" + "a" * 250 + ".com",
            "spaces in domain.com",
            "special!chars@domain.com",
            "just-a-string-without-dot"
        ]

        for domain in invalid_domains:
            with self.subTest(domain=domain):
                self.assertFalse(Piedomain.validate_domain_name(domain))

    def test_validate_domain_name_with_protocol(self):
        """Test domain validation with HTTP/HTTPS protocols."""
        domains_with_protocol = [
            "http://google.com",
            "https://example.org",
            "https://sub.example.com/path"
        ]

        for domain in domains_with_protocol:
            with self.subTest(domain=domain):
                self.assertTrue(Piedomain.validate_domain_name(domain))

    def test_validate_domains_list(self):
        """Test validation of domain lists."""
        mixed_domains = [
            "google.com",
            "invalid..domain",
            "facebook.com",
            "",
            "twitter.com"
        ]

        valid, invalid = Piedomain.validate_domains(mixed_domains)

        self.assertEqual(len(valid), 3)
        self.assertEqual(len(invalid), 2)
        self.assertIn("google.com", valid)
        self.assertIn("facebook.com", valid)
        self.assertIn("twitter.com", valid)
        self.assertIn("invalid..domain", invalid)
        self.assertIn("", invalid)

    def test_validate_domains_normalization(self):
        """Test domain normalization during validation."""
        domains_to_normalize = [
            "https://example.com",
            "http://test.org/path",
            "example.net/"
        ]

        valid, invalid = Piedomain.validate_domains(domains_to_normalize)

        self.assertEqual(len(valid), 3)
        self.assertEqual(len(invalid), 0)
        self.assertIn("example.com", valid)
        self.assertIn("test.org", valid)
        self.assertIn("example.net", valid)


if __name__ == "__main__":
    unittest.main()

