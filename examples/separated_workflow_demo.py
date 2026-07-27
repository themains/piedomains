#!/usr/bin/env python3
"""
Demo of the new separated data collection and inference workflow.

This example shows how to:
1. Collect data for domains (HTML, screenshots)
2. Perform inference separately using different classifiers
3. Compare results across different approaches

Usage:
    python examples/separated_workflow_demo.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from piedomains import (
        DataCollector,
        DomainClassifier,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("This demo requires the piedomains package to be installed.")
    print("Run: pip install -e . from the project root")
    sys.exit(1)


def demo_separated_workflow():
    """Demonstrate the separated workflow."""
    print("🔧 Piedomains Separated Workflow Demo")
    print("=" * 50)

    # Test domains
    domains = ["example.com", "httpbin.org"]

    # Create data collector
    print("\n📦 Step 1: Data Collection")
    collector = DataCollector(cache_dir="demo_data")

    try:
        # Collect data
        print(f"Collecting data for {len(domains)} domains...")
        collection_data = collector.collect(domains, use_cache=True)

        print("✅ Collection complete!")
        print(f"   Collection ID: {collection_data['collection_id']}")
        print(f"   Total domains: {collection_data['summary']['total_domains']}")
        print(f"   Successful: {collection_data['summary']['successful']}")
        print(f"   Failed: {collection_data['summary']['failed']}")

        # Show collected files
        print("\n📁 Collected Files:")
        for domain_data in collection_data["domains"]:
            if domain_data["fetch_success"]:
                print(f"   {domain_data['domain']}:")
                print(f"     HTML: {domain_data['text_path']}")
                print(f"     Image: {domain_data['image_path']}")
            else:
                print(f"   {domain_data['domain']}: FAILED - {domain_data['error']}")

        # Step 2: Classification using different approaches
        print("\n🧠 Step 2: Classification")

        # Use DomainClassifier for high-level API
        classifier = DomainClassifier()

        print("\n🔤 Text Classification:")
        try:
            text_results = classifier.classify_from_collection(
                collection_data, method="text"
            )
            for result in text_results:
                if result["error"]:
                    print(f"   {result['domain']}: ERROR - {result['error']}")
                else:
                    print(
                        f"   {result['domain']}: {result['category']} ({result['confidence']:.3f})"
                    )
        except Exception as e:
            print(f"   Text classification failed: {e}")

        print("\n🖼️  Image Classification:")
        try:
            image_results = classifier.classify_from_collection(
                collection_data, method="images"
            )
            for result in image_results:
                if result["error"]:
                    print(f"   {result['domain']}: ERROR - {result['error']}")
                else:
                    print(
                        f"   {result['domain']}: {result['category']} ({result['confidence']:.3f})"
                    )
        except Exception as e:
            print(f"   Image classification failed: {e}")

        # Step 3: Data reuse demonstration
        print("\n♻️  Step 3: Data Reuse")
        print("The same collected data can now be used with different models:")
        print("- Try different ML model versions")
        print("- Compare with LLM-based classification")
        print("- Run ensemble methods")
        print("- Export for external analysis")

        # Show JSON output example
        print("\n📄 JSON Output Example:")
        if collection_data["domains"]:
            example_domain = collection_data["domains"][0]
            example_result = {
                "url": example_domain["url"],
                "text_path": example_domain.get("text_path"),
                "image_path": example_domain.get("image_path"),
                "date_time_collected": example_domain["date_time_collected"],
                "model_used": "text/shallalist_ml",
                "category": "example_category",
                "confidence": 0.85,
                "reason": None,
                "error": None,
            }
            print(json.dumps(example_result, indent=2))

        print("\n✅ Demo completed successfully!")
        print("\nKey Benefits:")
        print("- 📊 Clean data lineage and transparency")
        print("- ♻️  Reuse collected data with multiple models")
        print("- 🔍 Easy debugging and inspection")
        print("- 🌐 JSON format works with any tool")
        print("- 🔬 Perfect for research and experimentation")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nThis is expected if:")
        print("- Dependencies are missing (PIL, TensorFlow, etc.)")
        print("- Network is unavailable")
        print("- ML models aren't downloaded")
        print("\nThe demo shows the API structure even without full functionality.")


def demo_json_schemas():
    """Show the JSON schemas used in the separated workflow."""
    print("\n📋 JSON Schema Reference")
    print("=" * 30)

    print("\n1️⃣ Collection Metadata Schema:")
    collection_schema = {
        "collection_id": "uuid-string",
        "timestamp": "2024-12-17T10:30:00Z",
        "config": {
            "cache_dir": "/path/to/cache",
            "archive_date": "20200101",  # or null for live
            "fetcher_type": "archive",  # or "live"
        },
        "domains": [
            {
                "url": "example.com",
                "domain": "example.com",
                "text_path": "html/example.com.html",
                "image_path": "images/example.com.png",
                "date_time_collected": "2024-12-17T10:30:15Z",
                "fetch_success": True,
                "error": None,
            }
        ],
        "summary": {"total_domains": 1, "successful": 1, "failed": 0},
    }
    print(json.dumps(collection_schema, indent=2))

    print("\n2️⃣ Inference Results Schema:")
    results_schema = [
        {
            "url": "example.com",
            "text_path": "html/example.com.html",
            "image_path": "images/example.com.png",
            "date_time_collected": "2024-12-17T10:30:15Z",
            "model_used": "text/shallalist_ml",  # or "image/shallalist_ml", "llm/openai_gpt4", etc.
            "category": "news",
            "confidence": 0.87,
            "reason": None,  # LLM reasoning (if applicable)
            "error": None,
            "raw_predictions": {
                "news": 0.87,
                "sports": 0.13,
            },  # full probability distribution
        }
    ]
    print(json.dumps(results_schema, indent=2))


if __name__ == "__main__":
    demo_separated_workflow()
    demo_json_schemas()
