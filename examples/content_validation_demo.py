#!/usr/bin/env python3
"""
Demonstration of content validation and security features in piedomains.

This script shows how the new content validation system protects against
potentially dangerous URLs like executables and PDFs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from piedomains.config import get_config
from piedomains.content_validation import ContentValidator
from piedomains.fetchers import LiveFetcher


def test_url_validation():
    """Test URL validation with various types of content."""
    print("🔒 Content Validation Demo")
    print("=" * 50)

    validator = ContentValidator()

    # Test URLs with different safety levels
    test_urls = [
        # Safe URLs
        ("https://wikipedia.org", "Safe webpage"),
        ("https://github.com", "Safe webpage"),
        # Potentially dangerous URLs
        ("https://example.com/malware.exe", "Executable file"),
        ("https://example.com/document.pdf", "PDF document"),
        ("https://example.com/archive.zip", "Archive file"),
        ("https://example.com/download/file.jar", "Java executable"),
        # Suspicious patterns
        ("https://example.com/attachment/file.bin", "Binary attachment"),
        ("https://example.com/file.txt?download=true", "Forced download"),
    ]

    for url, description in test_urls:
        print(f"\n📋 Testing: {description}")
        print(f"URL: {url}")
        print("-" * 40)

        result = validator.validate_url(url)

        print(f"Safe: {'✅' if result.is_safe else '❌'}")
        print(f"Content-Type: {result.content_type or 'Unknown'}")

        if result.warnings:
            print(f"Warnings: {'; '.join(result.warnings)}")

        if not result.is_safe:
            print(f"❌ BLOCKED: {result.error_message}")

        if result.sandbox_recommended:
            sandbox_cmd = validator.get_sandbox_command(url)
            print(f"🔒 Sandbox recommended: {sandbox_cmd}")


def test_fetcher_integration():
    """Test content validation integrated with fetchers."""
    print(f"\n{'=' * 50}")
    print("🌐 Fetcher Integration Test")
    print("=" * 50)

    fetcher = LiveFetcher()

    # Test safe URL
    print("\n✅ Testing safe URL (should work)")
    success, content, error = fetcher.fetch_html("https://httpbin.org/html")
    print(f"Success: {success}")
    if error:
        print(f"Message: {error}")
    print(f"Content length: {len(content) if success else 0} chars")

    # Test dangerous URL (simulated)
    print("\n❌ Testing dangerous URL (should be blocked)")
    try:
        success, content, error = fetcher.fetch_html(
            "https://httpbin.org/response-headers?Content-Type=application/octet-stream"
        )
        print(f"Success: {success}")
        if not success:
            print(f"BLOCKED: {error}")
        else:
            print(f"Unexpectedly allowed - Content length: {len(content)} chars")
    except Exception as e:
        print(f"Exception: {e}")

    # Test with force_fetch override
    print("\n⚠️ Testing force_fetch override (dangerous but allowed)")
    try:
        success, content, error = fetcher.fetch_html(
            "https://httpbin.org/response-headers?Content-Type=application/octet-stream",
            force_fetch=True,
        )
        print(f"Success: {success}")
        if success:
            print(f"Force fetch succeeded - Content length: {len(content)} chars")
        print(f"Message: {error}")
    except Exception as e:
        print(f"Exception: {e}")


def test_configuration():
    """Test configuration options for content validation."""
    print(f"\n{'=' * 50}")
    print("⚙️ Configuration Test")
    print("=" * 50)

    config = get_config()

    print("Current security settings:")
    print(f"• Content validation enabled: {config.enable_content_validation}")
    print(f"• Safety mode: {config.content_safety_mode}")
    print(f"• Max content length: {config.max_content_length / (1024 * 1024):.1f} MB")
    print(f"• Sandbox required: {config.sandbox_mode_required}")
    print(f"• Allowed content types: {', '.join(config.allowed_content_types)}")
    print(f"• Blocked extensions: {', '.join(config.blocked_extensions[:5])}...")

    print("\n💡 You can override these settings with:")
    print("• Environment variables (PIEDOMAINS_CONTENT_SAFETY_MODE=permissive)")
    print("• Configuration updates (config.set('content_safety_mode', 'moderate'))")
    print(
        "• Method parameters (force_fetch=True, allow_content_types=['application/pdf'])"
    )


def main():
    """Run all content validation demos."""
    try:
        test_url_validation()
        test_fetcher_integration()
        test_configuration()

        print(f"\n{'=' * 50}")
        print("✅ Content Validation Demo Complete")
        print("=" * 50)
        print("\n🛡️ Key takeaways:")
        print("• piedomains now validates content safety by default")
        print("• Dangerous file types (.exe, .pdf, etc.) are blocked")
        print("• Sandbox execution is recommended for risky content")
        print("• Users can override restrictions with clear warnings")
        print("• Multiple configuration options provide flexibility")

        print("\n📚 For more security examples, see:")
        print("• examples/sandbox/ - Complete sandboxing solutions")
        print("• examples/sandbox/secure_classify.py - Safe classification script")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("This might indicate missing dependencies or configuration issues.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
