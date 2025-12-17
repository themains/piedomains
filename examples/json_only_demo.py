#!/usr/bin/env python3
"""
Demo of the new JSON-only classification architecture.

This example shows the clean JSON API that replaces DataFrames.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from piedomains import DataCollector, DomainClassifier
except ImportError as e:
    print(f"Import error: {e}")
    print("This demo requires the piedomains package to be installed.")
    sys.exit(1)


def demo_json_api():
    """Demonstrate the new JSON-only API."""
    print("🚀 Piedomains JSON-Only Architecture Demo")
    print("=" * 50)

    # Test domains
    domains = ["example.com", "httpbin.org"]

    # Create classifier
    classifier = DomainClassifier(cache_dir="demo_cache")

    print(f"\n🔤 Testing JSON API with {len(domains)} domains...")
    print("Domains:", domains)

    try:
        # Test the new JSON-only classify method
        results = classifier.classify(domains)

        print("\n✅ Classification complete!")
        print(f"Result type: {type(results)}")
        print(f"Number of results: {len(results)}")

        print("\n📊 Results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Domain: {result.get('domain', 'unknown')}")
            print(f"   URL: {result.get('url', 'unknown')}")
            print(f"   Category: {result.get('category', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
            print(f"   Model Used: {result.get('model_used', 'unknown')}")
            print(
                f"   Data Collection Time: {result.get('date_time_collected', 'unknown')}"
            )
            print(f"   Text Path: {result.get('text_path', 'none')}")
            print(f"   Image Path: {result.get('image_path', 'none')}")

            if result.get("error"):
                print(f"   ❌ Error: {result['error']}")
            else:
                print("   ✅ Success")

        # Test different classification methods
        print("\n🔤 Testing text-only classification...")
        text_results = classifier.classify_by_text(domains)
        print(f"Text results: {len(text_results)} domains")
        for result in text_results:
            print(
                f"   {result['domain']}: {result.get('category', 'error')} "
                f"({result.get('confidence', 0):.3f}) - {result.get('model_used', 'unknown')}"
            )

        print("\n🖼️  Testing image-only classification...")
        image_results = classifier.classify_by_images(domains)
        print(f"Image results: {len(image_results)} domains")
        for result in image_results:
            print(
                f"   {result['domain']}: {result.get('category', 'error')} "
                f"({result.get('confidence', 0):.3f}) - {result.get('model_used', 'unknown')}"
            )

        # Show JSON structure
        print("\n📋 JSON Schema Example:")
        if results:
            example_result = results[0]
            import json

            print(json.dumps(example_result, indent=2))

        print("\n✅ Demo completed successfully!")
        print("\nKey improvements:")
        print("- 🗂️  Pure JSON output (no pandas dependency)")
        print("- 🔄 Unified data collection → inference pipeline")
        print("- 📁 Clear data file paths for debugging")
        print("- ♻️  Data reuse across multiple classification approaches")
        print("- 🌐 Language-agnostic JSON format")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        print("\nThis is expected if:")
        print("- Dependencies are missing")
        print("- Network is unavailable")
        print("- ML models aren't downloaded")
        print("\nThe demo shows the API structure even without full functionality.")


def demo_separated_workflow():
    """Show the separated data collection and inference workflow."""
    print("\n" + "=" * 50)
    print("🔧 Separated Data Collection & Inference Demo")
    print("=" * 50)

    domains = ["httpbin.org"]

    try:
        print("\n📦 Step 1: Data Collection")
        collector = DataCollector(cache_dir="demo_separated")
        collection_data = collector.collect(domains)

        print("✅ Collection complete!")
        print(f"   Collection ID: {collection_data['collection_id']}")
        print(f"   Successful: {collection_data['summary']['successful']}")
        print(f"   Failed: {collection_data['summary']['failed']}")

        print("\n🧠 Step 2: Classification")
        classifier = DomainClassifier()

        print("Running text classification on collected data...")
        results = classifier.classify_from_collection(collection_data, method="text")

        print("✅ Inference complete!")
        for result in results:
            print(
                f"   {result['domain']}: {result.get('category', 'error')} "
                f"({result.get('confidence', 0):.3f})"
            )

        print("\n♻️  Data Reuse: The same collected data can now be used with:")
        print("   - Different ML model versions")
        print("   - LLM-based classification")
        print("   - Ensemble approaches")
        print("   - External analysis tools")

    except Exception as e:
        print(f"❌ Separated workflow demo failed: {e}")


if __name__ == "__main__":
    demo_json_api()
    demo_separated_workflow()
