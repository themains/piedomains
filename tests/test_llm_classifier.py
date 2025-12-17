#!/usr/bin/env python

"""
Tests for LLM-based domain classification.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from piedomains.api import DomainClassifier
from piedomains.llm.config import LLMConfig
from piedomains.llm.prompts import get_classification_prompt, get_multimodal_prompt
from piedomains.llm.response_parser import parse_llm_response


class TestLLMConfig(unittest.TestCase):
    """Test LLM configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = LLMConfig(provider="openai", model="gpt-4o", api_key="test-key")

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-4o")
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_tokens, 500)
        self.assertIsInstance(config.categories, list)
        self.assertGreater(len(config.categories), 5)

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
            temperature=0.2,
            max_tokens=1000,
            categories=["news", "shopping", "tech"],
        )

        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.categories, ["news", "shopping", "tech"])

    def test_validation_errors(self):
        """Test configuration validation."""
        with self.assertRaises(ValueError):
            LLMConfig(provider="", model="gpt-4o", api_key="test-key")

        with self.assertRaises(ValueError):
            LLMConfig(provider="openai", model="", api_key="test-key")

        with self.assertRaises(ValueError):
            LLMConfig(
                provider="openai", model="gpt-4o", api_key="test-key", max_tokens=0
            )

    def test_litellm_params(self):
        """Test conversion to litellm parameters."""
        config = LLMConfig(provider="openai", model="gpt-4o", api_key="test-key")
        params = config.to_litellm_params()

        self.assertEqual(params["model"], "openai/gpt-4o")
        self.assertEqual(params["api_key"], "test-key")
        self.assertEqual(params["temperature"], 0.1)
        self.assertEqual(params["max_tokens"], 500)


class TestPrompts(unittest.TestCase):
    """Test prompt generation."""

    def test_text_classification_prompt(self):
        """Test text-only classification prompt."""
        prompt = get_classification_prompt(
            domain="example.com",
            content="This is news content about current events.",
            categories=["news", "shopping", "tech"],
        )

        self.assertIn("example.com", prompt)
        self.assertIn("news content", prompt)
        self.assertIn("news, shopping, tech", prompt)
        self.assertIn("JSON", prompt)
        self.assertIn("category", prompt)
        self.assertIn("confidence", prompt)

    def test_multimodal_prompt(self):
        """Test multimodal classification prompt."""
        prompt = get_multimodal_prompt(
            domain="example.com",
            content="Shopping website content",
            categories=["news", "shopping", "tech"],
            has_screenshot=True,
        )

        self.assertIn("example.com", prompt)
        self.assertIn("Shopping website", prompt)
        self.assertIn("screenshot", prompt)
        self.assertIn("visual", prompt)

    def test_content_truncation(self):
        """Test content truncation for long texts."""
        long_content = "test " * 10000  # Very long content
        prompt = get_classification_prompt(
            domain="example.com",
            content=long_content,
            categories=["news"],
            max_content_length=1000,
        )

        self.assertIn("truncated", prompt)
        self.assertLess(len(prompt), len(long_content) + 2000)


class TestResponseParser(unittest.TestCase):
    """Test LLM response parsing."""

    def test_parse_valid_json_response(self):
        """Test parsing valid JSON response."""
        response = """
        {
            "category": "news",
            "confidence": 0.95,
            "reasoning": "The website contains news articles and current events."
        }
        """

        result = parse_llm_response(response)

        self.assertEqual(result["category"], "news")
        self.assertEqual(result["confidence"], 0.95)
        self.assertIn("news articles", result["reasoning"])

    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown."""
        response = """
        Here's my analysis:

        ```json
        {
            "category": "shopping",
            "confidence": 0.88,
            "reasoning": "E-commerce site with product listings."
        }
        ```
        """

        result = parse_llm_response(response)

        self.assertEqual(result["category"], "shopping")
        self.assertEqual(result["confidence"], 0.88)

    def test_parse_invalid_response(self):
        """Test handling of invalid responses."""
        with self.assertRaises(ValueError):
            parse_llm_response("")

        with self.assertRaises(ValueError):
            parse_llm_response("This is not JSON at all")

    def test_confidence_validation(self):
        """Test confidence value validation."""
        # Test out-of-range confidence
        response = """{"category": "news", "confidence": 1.5}"""
        result = parse_llm_response(response)

        self.assertEqual(result["category"], "news")
        self.assertEqual(result["confidence"], 1.0)  # Should be clamped

    def test_missing_fields(self):
        """Test handling of missing required fields."""
        # Missing category
        with self.assertRaises(ValueError):
            parse_llm_response('{"confidence": 0.9}')


class TestDomainClassifierLLM(unittest.TestCase):
    """Test LLM integration with DomainClassifier."""

    def setUp(self):
        """Set up test environment."""
        self.classifier = DomainClassifier()

    def test_configure_llm(self):
        """Test LLM configuration."""
        self.classifier.configure_llm(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            categories=["news", "tech"],
        )

        self.assertIsNotNone(self.classifier._llm_config)
        self.assertIsNotNone(self.classifier._llm_classifier)
        self.assertEqual(self.classifier._llm_config.provider, "openai")
        self.assertEqual(self.classifier._llm_config.categories, ["news", "tech"])

    def test_llm_not_configured_error(self):
        """Test error when LLM not configured."""
        with self.assertRaises(RuntimeError):
            self.classifier.classify_by_llm(["example.com"])

        with self.assertRaises(RuntimeError):
            self.classifier.classify_by_llm_multimodal(["example.com"])

    def test_usage_stats_not_configured(self):
        """Test usage stats when LLM not configured."""
        stats = self.classifier.get_llm_usage_stats()
        self.assertIsNone(stats)

    @patch("piedomains.llm.litellm")
    def test_classify_by_llm_mock(self, mock_litellm):
        """Test LLM classification with mocked response."""
        # Mock litellm response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "category": "news",
            "confidence": 0.9,
            "reasoning": "Contains news content"
        }
        """
        mock_response.usage.total_tokens = 100
        mock_litellm.completion.return_value = mock_response

        # Configure LLM
        self.classifier.configure_llm(
            provider="openai", model="gpt-4o", api_key="test-key"
        )

        # Mock data collection and classification to avoid actual network calls
        with patch("piedomains.data_collector.DataCollector.collect") as mock_collect:
            mock_collect.return_value = {
                "collection_id": "test_collection",
                "timestamp": "2025-12-17T12:00:00Z",
                "domains": [
                    {
                        "url": "example.com",
                        "domain": "example.com",
                        "text_path": "html/example.com.html",
                        "image_path": "images/example.com.png",
                        "date_time_collected": "2025-12-17T12:00:00Z",
                        "fetch_success": True,
                        "cached": False,
                        "error": None
                    }
                ]
            }

            result = self.classifier.classify_by_llm(["example.com"])

            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)

            # Check that litellm.completion was called
            mock_litellm.completion.assert_called()


@pytest.mark.slow
class TestLLMIntegration(unittest.TestCase):
    """Integration tests for LLM functionality (marked as slow)."""

    def test_full_pipeline_mock(self):
        """Test full LLM pipeline with mocked components."""
        classifier = DomainClassifier()

        # This would normally require real API keys
        # In real usage: classifier.configure_llm("openai", "gpt-4o", api_key="sk-...")

        # For now, just test that the methods exist and have correct signatures
        self.assertTrue(hasattr(classifier, "configure_llm"))
        self.assertTrue(hasattr(classifier, "classify_by_llm"))
        self.assertTrue(hasattr(classifier, "classify_by_llm_multimodal"))
        self.assertTrue(hasattr(classifier, "get_llm_usage_stats"))


if __name__ == "__main__":
    unittest.main()
