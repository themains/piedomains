"""Response parsing utilities for LLM classification results."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def parse_llm_response(response_text: str) -> dict[str, Any]:
    """Parse LLM response into structured classification result.

    Args:
        response_text: Raw response text from LLM

    Returns:
        Dictionary with parsed classification data

    Raises:
        ValueError: If response cannot be parsed
    """
    if not response_text:
        raise ValueError("Empty response text")

    # Try to extract JSON from the response
    json_data = _extract_json(response_text)

    if json_data:
        return _validate_classification_result(json_data)

    # Fallback: try to parse structured text response
    return _parse_text_response(response_text)


def parse_batch_response(response_text: str) -> list[dict[str, Any]]:
    """Parse batch LLM response into list of classification results.

    Args:
        response_text: Raw batch response text from LLM

    Returns:
        List of dictionaries with parsed classification data

    Raises:
        ValueError: If response cannot be parsed
    """
    if not response_text:
        raise ValueError("Empty response text")

    # Try to extract JSON array from the response
    json_data = _extract_json(response_text)

    if isinstance(json_data, list):
        results = []
        for item in json_data:
            try:
                results.append(_validate_classification_result(item))
            except ValueError as e:
                logger.warning(f"Skipping invalid batch item: {e}")
                continue
        return results

    # If not a list, try to parse as single item
    if isinstance(json_data, dict):
        return [_validate_classification_result(json_data)]

    raise ValueError("Could not parse batch response as JSON array or object")


def _extract_json(text: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Extract JSON from text response."""
    # Clean the text
    text = text.strip()

    # Try to find JSON block markers
    json_patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r'(\{[^{}]*"category"[^{}]*\})',
        r'(\[[^[\]]*\{[^{}]*"category"[^{}]*\}[^[\]]*\])',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                continue

    # Try to parse the entire response as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON-like structure without quotes around keys
    try:
        # Simple regex to add quotes around unquoted keys
        fixed_text = re.sub(r"(\w+):", r'"\1":', text)
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass

    return None


def _validate_classification_result(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a classification result."""
    if not isinstance(data, dict):
        raise ValueError("Classification result must be a dictionary")

    # Required fields
    if "category" not in data:
        raise ValueError("Classification result missing 'category' field")

    category = str(data["category"]).strip().lower()
    if not category:
        raise ValueError("Category cannot be empty")

    # Optional fields with defaults
    confidence = data.get("confidence", 0.5)
    reasoning = data.get("reasoning", "No reasoning provided")

    # Validate confidence
    try:
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            logger.warning(f"Confidence {confidence} out of range [0,1], clamping")
            confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        logger.warning(f"Invalid confidence value: {confidence}, using 0.5")
        confidence = 0.5

    return {
        "category": category,
        "confidence": confidence,
        "reasoning": str(reasoning).strip(),
        "raw_response": data,
    }


def _parse_text_response(text: str) -> dict[str, Any]:
    """Parse non-JSON text response."""
    text = text.strip()

    # Look for category mentions
    category_patterns = [
        r"category[:\s]+([a-zA-Z_]+)",
        r"classified as[:\s]+([a-zA-Z_]+)",
        r"categorize.*as[:\s]+([a-zA-Z_]+)",
    ]

    category = None
    for pattern in category_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            category = match.group(1).strip().lower()
            break

    if not category:
        # Last resort: look for common category names
        common_categories = [
            "news",
            "shopping",
            "social",
            "educational",
            "entertainment",
            "technology",
            "finance",
            "health",
            "government",
            "sports",
        ]
        for cat in common_categories:
            if cat in text.lower():
                category = cat
                break

    if not category:
        raise ValueError(f"Could not extract category from response: {text[:200]}")

    # Look for confidence
    confidence_patterns = [
        r"confidence[:\s]+([0-9.]+)",
        r"([0-9.]+)\s*confidence",
    ]

    confidence = 0.5
    for pattern in confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                confidence = float(match.group(1))
                confidence = max(0.0, min(1.0, confidence))
                break
            except ValueError:
                continue

    return {
        "category": category,
        "confidence": confidence,
        "reasoning": "Parsed from text response",
        "raw_response": text,
    }
