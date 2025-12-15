"""Prompt templates for LLM-based domain classification."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_classification_prompt(
    domain: str, content: str, categories: list[str], max_content_length: int = 8000
) -> str:
    """Generate classification prompt for text-only analysis.

    Args:
        domain: Domain name to classify
        content: Extracted text content from the domain
        categories: List of available categories
        max_content_length: Maximum length of content to include

    Returns:
        Formatted prompt string
    """
    # Truncate content if too long
    if len(content) > max_content_length:
        content = content[:max_content_length] + "... [truncated]"

    categories_str = ", ".join(categories)

    prompt = f"""You are a domain classification expert. Analyze the provided domain and its content to determine the most appropriate category.

Domain: {domain}

Content from the domain:
{content}

Available categories: {categories_str}

Please classify this domain into ONE of the provided categories. Consider:
1. The domain name itself
2. The actual content and topics discussed
3. The primary purpose and audience of the website

Respond with a JSON object containing:
- "category": the most appropriate category from the list
- "confidence": a number between 0.0 and 1.0 indicating your confidence
- "reasoning": a brief explanation of why you chose this category

Example response:
{{
    "category": "news",
    "confidence": 0.95,
    "reasoning": "The domain contains current news articles, breaking news updates, and journalistic content covering various topics."
}}

Your response:"""

    return prompt


def get_multimodal_prompt(
    domain: str,
    content: str | None = None,
    categories: list[str] | None = None,
    has_screenshot: bool = False,
    max_content_length: int = 6000,
) -> str:
    """Generate classification prompt for multimodal analysis (text + image).

    Args:
        domain: Domain name to classify
        content: Extracted text content (optional)
        categories: List of available categories
        has_screenshot: Whether a screenshot image is provided
        max_content_length: Maximum length of content to include

    Returns:
        Formatted prompt string
    """
    if categories is None:
        categories = [
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

    categories_str = ", ".join(categories)

    prompt = f"""You are a domain classification expert. Analyze the provided domain and classify it appropriately.

Domain: {domain}

Available categories: {categories_str}"""

    if has_screenshot:
        prompt += "\n\nI've provided a screenshot of the website's homepage. Analyze the visual elements, layout, and design."

    if content:
        # Truncate content if too long (shorter for multimodal to leave room for image)
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"

        prompt += f"\n\nText content from the domain:\n{content}"

    prompt += """

Please classify this domain into ONE of the provided categories. Consider:
1. The domain name itself
2. The visual design and layout (if screenshot provided)
3. The actual text content (if provided)
4. The apparent purpose and target audience

Respond with a JSON object containing:
- "category": the most appropriate category from the list
- "confidence": a number between 0.0 and 1.0 indicating your confidence
- "reasoning": a brief explanation considering both visual and textual elements

Example response:
{
    "category": "shopping",
    "confidence": 0.92,
    "reasoning": "The website displays product listings, shopping cart functionality, and e-commerce layout typical of online retail sites."
}

Your response:"""

    return prompt


def get_custom_prompt(
    domain: str,
    content: str | None = None,
    categories: list[str] | None = None,
    custom_instructions: str | None = None,
    has_screenshot: bool = False,
) -> str:
    """Generate a custom classification prompt.

    Args:
        domain: Domain name to classify
        content: Extracted text content (optional)
        categories: List of available categories
        custom_instructions: Custom classification instructions
        has_screenshot: Whether a screenshot image is provided

    Returns:
        Formatted prompt string
    """
    base_prompt = get_multimodal_prompt(
        domain=domain,
        content=content,
        categories=categories,
        has_screenshot=has_screenshot,
    )

    if custom_instructions:
        # Insert custom instructions before the final response format
        parts = base_prompt.split("Please classify this domain")
        if len(parts) == 2:
            base_prompt = (
                parts[0]
                + f"\nAdditional instructions: {custom_instructions}\n\n"
                + "Please classify this domain"
                + parts[1]
            )

    return base_prompt


def get_batch_prompt(domains_data: list[dict[str, Any]], categories: list[str]) -> str:
    """Generate a batch classification prompt for multiple domains.

    Args:
        domains_data: List of dictionaries with domain info
        categories: List of available categories

    Returns:
        Formatted batch prompt string
    """
    categories_str = ", ".join(categories)

    prompt = f"""You are a domain classification expert. Analyze and classify multiple domains.

Available categories: {categories_str}

Domains to classify:
"""

    for i, domain_data in enumerate(domains_data, 1):
        domain = domain_data.get("domain", "unknown")
        content = domain_data.get("content", "")

        # Limit content length for batch processing
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"

        prompt += f"\n{i}. Domain: {domain}\n"
        if content:
            prompt += f"   Content: {content}\n"

    prompt += """
For each domain, classify it into ONE of the provided categories.

Respond with a JSON array containing objects for each domain:
[
    {
        "domain": "domain1.com",
        "category": "category_name",
        "confidence": 0.85,
        "reasoning": "brief explanation"
    },
    ...
]

Your response:"""

    return prompt
