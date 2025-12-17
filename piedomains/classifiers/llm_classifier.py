"""LLM-based domain classification using modern language models."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import litellm
import pandas as pd
from PIL import Image

from ..llm.config import LLMConfig
from ..llm.prompts import (
    get_batch_prompt,
    get_classification_prompt,
    get_custom_prompt,
    get_multimodal_prompt,
)
from ..llm.response_parser import parse_batch_response, parse_llm_response
from ..piedomains_logging import get_logger

logger = get_logger(__name__)


class LLMClassifier:
    """LLM-based domain classifier using multiple AI providers.

    This classifier leverages modern language models through the litellm
    library to classify domains based on text content and/or screenshots.

    Attributes:
        config: LLM configuration settings
        usage_stats: Dictionary tracking API usage and costs
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLM classifier.

        Args:
            config: LLM configuration object

        Raises:
            ValueError: If configuration is invalid
        """

        self.config = config
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "errors": 0,
        }

        # Test the configuration
        self._test_connection()

    def _test_connection(self) -> None:
        """Test LLM connection and configuration."""
        if not self.config.api_key:
            logger.warning(
                f"No API key configured for {self.config.provider}. "
                "Classification will fail without valid credentials."
            )
            return

        try:
            # Simple test request
            params = self.config.to_litellm_params()
            params["max_tokens"] = 5  # Override for test (use minimal tokens)

            litellm.completion(
                **params,
                messages=[{"role": "user", "content": "Hello"}],
            )
            logger.info(f"LLM connection test successful for {self.config.provider}")
        except Exception as e:
            logger.warning(f"LLM connection test failed: {e}")

    def classify_text(
        self,
        domains: list[str],
        content_dict: dict[str, str],
        custom_instructions: str | None = None,
    ) -> pd.DataFrame:
        """Classify domains based on text content only.

        Args:
            domains: List of domain names
            content_dict: Dictionary mapping domains to their text content
            custom_instructions: Optional custom classification instructions

        Returns:
            DataFrame with classification results
        """
        results = []

        for domain in domains:
            try:
                content = content_dict.get(domain, "")

                if custom_instructions:
                    prompt = get_custom_prompt(
                        domain=domain,
                        content=content,
                        categories=self.config.categories,
                        custom_instructions=custom_instructions,
                        has_screenshot=False,
                    )
                else:
                    prompt = get_classification_prompt(
                        domain=domain,
                        content=content,
                        categories=self.config.categories,
                    )

                result = self._make_request(prompt, domain)
                results.append(result)

            except Exception as e:
                logger.error(f"Error classifying {domain}: {e}")
                results.append(self._create_error_result(domain, str(e)))

        return pd.DataFrame(results)

    def classify_multimodal(
        self,
        domains: list[str],
        content_dict: dict[str, str] | None = None,
        screenshot_dict: dict[str, str | Path] | None = None,
        custom_instructions: str | None = None,
    ) -> pd.DataFrame:
        """Classify domains using both text and image data.

        Args:
            domains: List of domain names
            content_dict: Dictionary mapping domains to text content
            screenshot_dict: Dictionary mapping domains to screenshot paths
            custom_instructions: Optional custom classification instructions

        Returns:
            DataFrame with classification results
        """
        results = []
        content_dict = content_dict or {}
        screenshot_dict = screenshot_dict or {}

        for domain in domains:
            try:
                content = content_dict.get(domain, "")
                screenshot_path = screenshot_dict.get(domain)
                has_screenshot = screenshot_path is not None

                if custom_instructions:
                    prompt = get_custom_prompt(
                        domain=domain,
                        content=content,
                        categories=self.config.categories,
                        custom_instructions=custom_instructions,
                        has_screenshot=has_screenshot,
                    )
                else:
                    prompt = get_multimodal_prompt(
                        domain=domain,
                        content=content,
                        categories=self.config.categories,
                        has_screenshot=has_screenshot,
                    )

                # Prepare messages with image if available
                messages = [{"role": "user", "content": prompt}]

                if has_screenshot and screenshot_path:
                    image_content = self._encode_image(screenshot_path)
                    if image_content:
                        # Update message to include image
                        messages[0]["content"] = [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_content}},
                        ]

                result = self._make_request_with_messages(messages, domain)
                results.append(result)

            except Exception as e:
                logger.error(f"Error classifying {domain}: {e}")
                results.append(self._create_error_result(domain, str(e)))

        return pd.DataFrame(results)

    def classify_batch(
        self, domains_data: list[dict[str, Any]], batch_size: int = 5
    ) -> pd.DataFrame:
        """Classify multiple domains in batches for cost efficiency.

        Args:
            domains_data: List of dictionaries with domain and content data
            batch_size: Number of domains to process per batch

        Returns:
            DataFrame with classification results
        """
        all_results = []

        # Process in batches
        for i in range(0, len(domains_data), batch_size):
            batch = domains_data[i : i + batch_size]

            try:
                prompt = get_batch_prompt(batch, self.config.categories)
                response = self._make_llm_request(prompt)

                # Parse batch response
                batch_results = parse_batch_response(response)

                # Ensure we have results for all domains in batch
                for j, domain_data in enumerate(batch):
                    domain = domain_data.get("domain", f"unknown_{i + j}")

                    if j < len(batch_results):
                        result = batch_results[j]
                        result["domain"] = domain
                        all_results.append(result)
                    else:
                        all_results.append(
                            self._create_error_result(
                                domain, "Missing in batch response"
                            )
                        )

            except Exception as e:
                logger.error(f"Error processing batch {i}-{i + batch_size}: {e}")

                # Add error results for all domains in failed batch
                for domain_data in batch:
                    domain = domain_data.get("domain", f"unknown_{i}")
                    all_results.append(self._create_error_result(domain, str(e)))

        return pd.DataFrame(all_results)

    def _make_request(self, prompt: str, domain: str) -> dict[str, Any]:
        """Make a simple text request to the LLM."""
        response_text = self._make_llm_request(prompt)
        result = parse_llm_response(response_text)
        result["domain"] = domain
        return result

    def _make_request_with_messages(
        self, messages: list[dict[str, Any]], domain: str
    ) -> dict[str, Any]:
        """Make a request with custom message format (for multimodal)."""
        try:
            start_time = time.time()

            response = litellm.completion(
                **self.config.to_litellm_params(), messages=messages
            )

            # Update usage stats
            self._update_usage_stats(response, time.time() - start_time)

            response_text = response.choices[0].message.content
            result = parse_llm_response(response_text)
            result["domain"] = domain

            return result

        except Exception as e:
            self.usage_stats["errors"] += 1
            raise e

    def _make_llm_request(self, prompt: str) -> str:
        """Make a request to the LLM and return response text."""
        try:
            start_time = time.time()

            response = litellm.completion(
                **self.config.to_litellm_params(),
                messages=[{"role": "user", "content": prompt}],
            )

            # Update usage stats
            self._update_usage_stats(response, time.time() - start_time)

            return response.choices[0].message.content

        except Exception as e:
            self.usage_stats["errors"] += 1
            raise e

    def _update_usage_stats(self, response: Any, duration: float) -> None:
        """Update usage statistics."""
        self.usage_stats["total_requests"] += 1

        if hasattr(response, "usage") and response.usage:
            if hasattr(response.usage, "total_tokens"):
                self.usage_stats["total_tokens"] += response.usage.total_tokens

            # Rough cost estimation (varies by model)
            # These are approximate rates as of 2024
            cost_per_1k_tokens = {
                "gpt-4": 0.03,
                "gpt-4o": 0.005,
                "claude-3": 0.015,
                "gemini-pro": 0.001,
            }

            model_key = next(
                (k for k in cost_per_1k_tokens if k in self.config.model.lower()),
                "gpt-4",  # default
            )

            estimated_cost = (
                self.usage_stats["total_tokens"] / 1000 * cost_per_1k_tokens[model_key]
            )
            self.usage_stats["estimated_cost_usd"] = estimated_cost

        # Check cost limits
        if (
            self.config.cost_limit_usd > 0
            and self.usage_stats["estimated_cost_usd"] > self.config.cost_limit_usd
        ):
            logger.warning(
                f"Cost limit ${self.config.cost_limit_usd} exceeded! "
                f"Current cost: ${self.usage_stats['estimated_cost_usd']:.4f}"
            )

    def _encode_image(self, image_path: str | Path) -> str | None:
        """Encode image to base64 data URL."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None

            # Open and resize image if needed
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if too large (most models have limits)
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                img_data = buffer.getvalue()

                b64_data = base64.b64encode(img_data).decode("utf-8")
                return f"data:image/jpeg;base64,{b64_data}"

        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    def _create_error_result(self, domain: str, error_msg: str) -> dict[str, Any]:
        """Create an error result dictionary."""
        return {
            "domain": domain,
            "category": "unknown",
            "confidence": 0.0,
            "reasoning": f"Error: {error_msg}",
            "error": error_msg,
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats.copy()

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "errors": 0,
        }
