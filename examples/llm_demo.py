#!/usr/bin/env python3
"""
Demo script for LLM-based domain classification using piedomains.

This example shows how to use modern AI models for domain classification
with support for custom categories and multimodal analysis.
"""

import os

from piedomains import DomainClassifier


def main():
    """Demonstrate LLM classification features."""

    # Initialize classifier
    classifier = DomainClassifier()

    # Example 1: Configure LLM with OpenAI
    print("🤖 Configuring LLM for domain classification...")

    # Option 1: Use environment variable for API key
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

    # Option 2: Pass API key directly (not recommended for production)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return

    classifier.configure_llm(
        provider="openai",
        model="gpt-4o",  # Multimodal model
        api_key=api_key,
        categories=[
            "news",
            "shopping",
            "social",
            "technology",
            "entertainment",
            "education",
        ],
        temperature=0.1,  # Low temperature for consistent results
        max_tokens=300,
    )

    # Example 2: Text-only LLM classification
    print("\n📝 Testing LLM text-only classification...")
    domains = ["cnn.com", "amazon.com", "github.com"]

    try:
        result = classifier.classify_by_llm(domains)
        print("Results:")
        for _, row in result.iterrows():
            print(
                f"  {row['domain']}: {row['category']} (confidence: {row['confidence']:.2f})"
            )
            print(f"    Reasoning: {row['reasoning']}")
            print()
    except Exception as e:
        print(f"❌ Text classification failed: {e}")

    # Example 3: Multimodal classification (text + screenshots)
    print("\n🖼️  Testing LLM multimodal classification...")
    domains = ["reddit.com"]  # Single domain for demo

    try:
        result = classifier.classify_by_llm_multimodal(domains)
        print("Multimodal Results:")
        for _, row in result.iterrows():
            print(
                f"  {row['domain']}: {row['category']} (confidence: {row['confidence']:.2f})"
            )
            print(f"    Reasoning: {row['reasoning']}")
            print()
    except Exception as e:
        print(f"❌ Multimodal classification failed: {e}")

    # Example 4: Custom instructions
    print("\n🎯 Testing custom classification instructions...")
    custom_instructions = """
    Focus specifically on whether this website is suitable for children.
    Consider content safety, educational value, and age-appropriateness.
    Use categories: child-safe, teen-appropriate, adult-content, unknown
    """

    try:
        result = classifier.classify_by_llm(
            ["khanacademy.org", "youtube.com"], custom_instructions=custom_instructions
        )
        print("Custom Classification Results:")
        for _, row in result.iterrows():
            print(
                f"  {row['domain']}: {row['category']} (confidence: {row['confidence']:.2f})"
            )
            print(f"    Reasoning: {row['reasoning']}")
            print()
    except Exception as e:
        print(f"❌ Custom classification failed: {e}")

    # Example 5: Usage statistics
    print("\n📊 LLM Usage Statistics:")
    stats = classifier.get_llm_usage_stats()
    if stats:
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Estimated cost: ${stats['estimated_cost_usd']:.4f}")
        if stats["errors"] > 0:
            print(f"  Errors: {stats['errors']}")

    print("\n✅ Demo completed!")


def demo_different_providers():
    """Demo different LLM providers."""
    classifier = DomainClassifier()

    providers = [
        {"provider": "openai", "model": "gpt-4o", "env_var": "OPENAI_API_KEY"},
        {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "env_var": "ANTHROPIC_API_KEY",
        },
        {"provider": "google", "model": "gemini-1.5-pro", "env_var": "GOOGLE_API_KEY"},
    ]

    print("🔄 Testing different LLM providers...")

    for config in providers:
        api_key = os.getenv(config["env_var"])
        if api_key:
            print(f"\n🧪 Testing {config['provider']}/{config['model']}...")

            try:
                classifier.configure_llm(
                    provider=config["provider"],
                    model=config["model"],
                    api_key=api_key,
                    categories=["news", "tech", "social"],
                )

                result = classifier.classify_by_llm(["techcrunch.com"])
                if not result.empty:
                    row = result.iloc[0]
                    print(
                        f"  Result: {row['category']} (confidence: {row['confidence']:.2f})"
                    )

            except Exception as e:
                print(f"  ❌ Failed: {e}")
        else:
            print(f"\n⏭️  Skipping {config['provider']} (no API key)")


if __name__ == "__main__":
    print("🚀 piedomains LLM Classification Demo")
    print("=" * 40)

    main()

    # Uncomment to test different providers
    # demo_different_providers()
