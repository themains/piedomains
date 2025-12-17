"""LLM-based domain classification using modern language models."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import litellm
from PIL import Image

from .llm.config import LLMConfig
from .llm.response_parser import parse_llm_response
from .piedomains_logging import get_logger

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

    def classify_text(self, domains: list[str], content_dict: dict[str, str]):
        """
        Classify domains using text content with LLM.

        Args:
            domains: List of domain names
            content_dict: Dict mapping domain to text content

        Returns:
            DataFrame with columns: category, confidence, reasoning
        """
        results = []
        for domain in domains:
            text_content = content_dict.get(domain, "")
            if not text_content.strip():
                results.append(
                    {
                        "domain": domain,
                        "category": "unknown",
                        "confidence": 0.0,
                        "reasoning": "No text content available",
                    }
                )
                continue

            try:
                # Create prompt for text classification
                from .llm.prompts import get_classification_prompt

                categories = self.config.categories or [
                    "news",
                    "shopping",
                    "social",
                    "tech",
                    "finance",
                    "education",
                ]
                prompt = get_classification_prompt(domain, text_content, categories)

                # Make LLM request
                response_text = self._make_llm_request(prompt)

                # Parse LLM response
                from .llm.response_parser import parse_llm_response

                parsed = parse_llm_response(response_text)

                results.append(
                    {
                        "domain": domain,
                        "category": parsed.get("category", "unknown"),
                        "confidence": parsed.get("confidence", 0.0),
                        "reasoning": parsed.get("reasoning", ""),
                    }
                )

            except Exception as e:
                logger.error(f"LLM classification failed for {domain}: {e}")
                self.usage_stats["errors"] += 1
                results.append(
                    {
                        "domain": domain,
                        "category": "unknown",
                        "confidence": 0.0,
                        "reasoning": f"Error: {e}",
                    }
                )

        return results

    def classify_multimodal(
        self,
        domains: list[str],
        content_dict: dict[str, str],
        screenshot_dict: dict[str, str],
    ):
        """
        Classify domains using text content and screenshots with LLM.

        Args:
            domains: List of domain names
            content_dict: Dict mapping domain to text content
            screenshot_dict: Dict mapping domain to screenshot path

        Returns:
            DataFrame with columns: category, confidence, reasoning
        """
        results = []
        for domain in domains:
            text_content = content_dict.get(domain, "")
            screenshot_path = screenshot_dict.get(domain, "")

            try:
                # Encode screenshot to base64
                image_data_url = None
                if screenshot_path:
                    image_data_url = self._encode_image(screenshot_path)

                if not text_content.strip() and not image_data_url:
                    results.append(
                        {
                            "domain": domain,
                            "category": "unknown",
                            "confidence": 0.0,
                            "reasoning": "No text or image content available",
                        }
                    )
                    continue

                # Create multimodal prompt
                from .llm.prompts import get_multimodal_prompt

                categories = self.config.categories or [
                    "news",
                    "shopping",
                    "social",
                    "tech",
                    "finance",
                    "education",
                ]
                prompt = get_multimodal_prompt(
                    domain,
                    text_content,
                    categories,
                    has_screenshot=bool(image_data_url),
                )

                # Make multimodal LLM request with image
                if image_data_url:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_data_url},
                                },
                            ],
                        }
                    ]
                    response = self._make_request_with_messages(messages, domain)
                    response_text = response.get("content", "")
                else:
                    # Fall back to text-only if no image
                    response_text = self._make_llm_request(prompt)

                # Parse LLM response
                from .llm.response_parser import parse_llm_response

                parsed = parse_llm_response(response_text)

                results.append(
                    {
                        "domain": domain,
                        "category": parsed.get("category", "unknown"),
                        "confidence": parsed.get("confidence", 0.0),
                        "reasoning": parsed.get("reasoning", ""),
                    }
                )

            except Exception as e:
                logger.error(f"Multimodal LLM classification failed for {domain}: {e}")
                self.usage_stats["errors"] += 1
                results.append(
                    {
                        "domain": domain,
                        "category": "unknown",
                        "confidence": 0.0,
                        "reasoning": f"Error: {e}",
                    }
                )

        return results

    def classify_from_paths(
        self,
        data_paths: list[dict],
        output_file: str = None,
        mode: str = "text",
        cache_dir: str = "cache",
    ) -> list[dict]:
        """
        Classify domains using files from collected data paths.

        Args:
            data_paths: List of dicts with domain data containing text_path/image_path, domain, etc.
            output_file: Optional path to save JSON results
            mode: Classification mode - "text", "image", or "multimodal"

        Returns:
            List of classification result dictionaries (JSON format)

        Example:
            >>> classifier = LLMClassifier(config)
            >>> data = [{"domain": "cnn.com", "text_path": "html/cnn.com.html", ...}]
            >>> results = classifier.classify_from_paths(data, mode="text")
            >>> print(results[0]["category"])
            news
        """
        results = []

        for domain_data in data_paths:
            domain = domain_data.get("domain")
            text_path = domain_data.get("text_path")
            image_path = domain_data.get("image_path")

            result = {
                "url": domain_data.get("url", domain),
                "domain": domain,
                "text_path": text_path,
                "image_path": image_path,
                "date_time_collected": domain_data.get("date_time_collected"),
                "model_used": f"{mode}/llm_{self.config.provider}_{self.config.model}",
                "category": None,
                "confidence": None,
                "raw_predictions": None,
                "reason": None,
                "error": None,
            }

            if not domain:
                result["error"] = "Missing domain"
                results.append(result)
                continue

            try:
                if mode == "text":
                    if not text_path:
                        result["error"] = "Missing text_path for text mode"
                    else:
                        # Load HTML from file path
                        try:
                            # Resolve path relative to cache directory if it's not absolute
                            import os

                            if not os.path.isabs(text_path):
                                resolved_text_path = os.path.join(cache_dir, text_path)
                            else:
                                resolved_text_path = text_path

                            with open(resolved_text_path, encoding="utf-8") as f:
                                html_content = f.read()

                            # Process HTML to text
                            from .text_processor import TextProcessor

                            text_content = TextProcessor.process_html_to_text(
                                html_content
                            )

                            # Get LLM classification
                            content_dict = {domain: text_content}
                            df_result = self.classify_text([domain], content_dict)

                            if not df_result.empty:
                                row = df_result.iloc[0]
                                result["category"] = row.get("category")
                                result["confidence"] = row.get("confidence", 0.0)
                                result["reason"] = row.get("reasoning")
                        except FileNotFoundError:
                            result["error"] = f"HTML file not found: {text_path}"

                elif mode == "image":
                    if not image_path:
                        result["error"] = "Missing image_path for image mode"
                    else:
                        # Classify using image only (with minimal text context)
                        screenshot_dict = {domain: image_path}
                        content_dict = {
                            domain: f"Homepage of {domain}"
                        }  # Minimal text context
                        df_result = self.classify_multimodal(
                            [domain], content_dict, screenshot_dict
                        )

                        if not df_result.empty:
                            row = df_result.iloc[0]
                            result["category"] = row.get("category")
                            result["confidence"] = row.get("confidence", 0.0)
                            result["reason"] = row.get("reasoning")

                elif mode == "multimodal":
                    if not text_path or not image_path:
                        result["error"] = (
                            "Missing text_path or image_path for multimodal mode"
                        )
                    else:
                        # Load HTML and use both text and image
                        try:
                            # Resolve paths relative to cache directory if they're not absolute
                            import os

                            if not os.path.isabs(text_path):
                                resolved_text_path = os.path.join(cache_dir, text_path)
                            else:
                                resolved_text_path = text_path

                            if not os.path.isabs(image_path):
                                resolved_image_path = os.path.join(
                                    cache_dir, image_path
                                )
                            else:
                                resolved_image_path = image_path

                            with open(resolved_text_path, encoding="utf-8") as f:
                                html_content = f.read()

                            from .text_processor import TextProcessor

                            text_content = TextProcessor.process_html_to_text(
                                html_content
                            )

                            content_dict = {domain: text_content}
                            screenshot_dict = {domain: resolved_image_path}
                            df_result = self.classify_multimodal(
                                [domain], content_dict, screenshot_dict
                            )

                            if not df_result.empty:
                                row = df_result.iloc[0]
                                result["category"] = row.get("category")
                                result["confidence"] = row.get("confidence", 0.0)
                                result["reason"] = row.get("reasoning")
                        except FileNotFoundError as e:
                            result["error"] = f"File not found: {e}"
                else:
                    result["error"] = f"Invalid mode: {mode}"

            except Exception as e:
                result["error"] = f"Classification error: {e}"

            results.append(result)

        # Save results if output file specified
        if output_file:
            # Create results directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Add metadata
            output_data = {
                "inference_timestamp": datetime.utcnow().isoformat() + "Z",
                "model_used": f"{mode}/llm_{self.config.provider}_{self.config.model}",
                "total_domains": len(data_paths),
                "successful": len([r for r in results if r["category"] is not None]),
                "failed": len([r for r in results if r["error"] is not None]),
                "usage_stats": self.get_usage_stats(),
                "results": results,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Saved LLM classification results to {output_file}")

        return results

    def classify_from_data(
        self, collection_data: dict, output_file: str = None, mode: str = "text"
    ) -> list[dict]:
        """
        Classify domains using collection metadata from DataCollector.

        Args:
            collection_data: Collection metadata dict from DataCollector.collect()
            output_file: Optional path to save JSON results
            mode: Classification mode - "text", "image", or "multimodal"

        Returns:
            List of classification result dictionaries (JSON format)

        Example:
            >>> from piedomains import DataCollector
            >>> collector = DataCollector()
            >>> data = collector.collect(["cnn.com"])
            >>> classifier = LLMClassifier(config)
            >>> results = classifier.classify_from_data(data, mode="multimodal")
        """
        # Extract domain data from collection metadata
        domains_data = collection_data.get("domains", [])

        # Filter based on mode requirements
        if mode == "text":
            valid_domains = [
                d for d in domains_data if d.get("fetch_success") and d.get("text_path")
            ]
        elif mode == "image":
            valid_domains = [
                d
                for d in domains_data
                if d.get("fetch_success") and d.get("image_path")
            ]
        elif mode == "multimodal":
            valid_domains = [
                d
                for d in domains_data
                if d.get("fetch_success") and d.get("text_path") and d.get("image_path")
            ]
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'text', 'image', or 'multimodal'"
            )

        if not valid_domains:
            logger.warning(f"No valid domains found for {mode} mode in collection")
            return []

        # Extract cache_dir from collection config for path resolution
        cache_dir = collection_data.get("config", {}).get("cache_dir", "cache")

        return self.classify_from_paths(valid_domains, output_file, mode, cache_dir)
