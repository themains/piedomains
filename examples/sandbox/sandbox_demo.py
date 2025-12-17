#!/usr/bin/env python3
"""
Sandbox Security Demo for piedomains

This script demonstrates how to safely run piedomains in isolated environments
to protect against potentially malicious domains.

Security Approaches Demonstrated:
1. Docker container isolation (recommended)
2. macOS sandbox-exec (built-in Mac sandboxing)
3. Network isolation techniques
4. Safe testing with known domains

Author: piedomains team
"""

import subprocess
from pathlib import Path


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def run_docker_sandbox():
    """Demonstrate running piedomains in Docker container."""
    print_section("DOCKER CONTAINER SANDBOX (Recommended)")

    print("Docker provides excellent isolation from the host system.")
    print("Even if malicious content is fetched, it's contained within the container.")
    print()

    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        docker_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        docker_available = False

    if docker_available:
        print("✅ Docker is available on your system")
        print()
        print("To run piedomains in Docker sandbox:")
        print("1. Build the container:")
        print("   docker build -t piedomains-sandbox .")
        print()
        print("2. Run classification in isolated container:")
        print('   docker run --rm piedomains-sandbox python -c "')
        print("   from piedomains import DomainClassifier")
        print("   classifier = DomainClassifier()")
        print("   result = classifier.classify(['example.com'])")
        print("   print(result[['domain', 'pred_label']])\"")
        print()
        print("3. For interactive session:")
        print("   docker run -it --rm piedomains-sandbox bash")
        print()
        print("Benefits:")
        print("- Complete filesystem isolation")
        print("- Network can be restricted with --network=none")
        print("- Temporary containers (--rm) leave no traces")
        print("- Can limit CPU/memory with --cpus and --memory")
    else:
        print("❌ Docker not found. To install Docker:")
        print("   Visit: https://docs.docker.com/desktop/install/mac-install/")

    return docker_available


def run_macos_sandbox():
    """Demonstrate macOS built-in sandboxing."""
    print_section("MACOS SANDBOX-EXEC (Built-in Mac Security)")

    print("macOS includes built-in sandboxing via sandbox-exec.")
    print("This restricts file system access and network operations.")
    print()

    # Create a simple sandbox profile
    sandbox_profile = """
(version 1)
(deny default)
(allow process-exec (literal "/usr/bin/python3"))
(allow file-read* (subpath "/usr/lib"))
(allow file-read* (subpath "/System/Library"))
(allow file-read* (literal "/dev/urandom"))
(allow file-read* (subpath "$(TMPDIR)"))
(allow file-write* (subpath "$(TMPDIR)"))
(allow network-outbound (remote ip))
(allow sysctl-read)
"""

    print("Example sandbox profile (restrictive):")
    print(sandbox_profile)
    print()

    print("To run with sandbox-exec:")
    print("1. Save the profile to a file (e.g., piedomains.sb)")
    print("2. Run:")
    print('   sandbox-exec -f piedomains.sb python3 -c "')
    print("   import piedomains; print('Sandboxed execution')\"")
    print()
    print("Note: This approach requires careful profile tuning.")
    print("Docker is generally easier and more reliable.")


def demonstrate_network_isolation():
    """Show network isolation techniques."""
    print_section("NETWORK ISOLATION TECHNIQUES")

    print("Additional network security measures:")
    print()
    print("1. Use VPN or isolated network:")
    print("   - Route traffic through VPN")
    print("   - Use isolated VM network")
    print()
    print("2. DNS filtering:")
    print("   - Use safe DNS servers (1.1.1.1, 8.8.8.8)")
    print("   - Block known malicious domains")
    print()
    print("3. Firewall rules:")
    print("   - Restrict outbound connections")
    print("   - Monitor network activity")
    print()
    print("4. Docker network isolation:")
    print("   docker run --network=none piedomains-sandbox")
    print("   (Requires pre-cached content or local testing)")


def safe_testing_demo():
    """Demonstrate safe testing practices."""
    print_section("SAFE TESTING WITH KNOWN DOMAINS")

    print("For testing piedomains safely, use these known-safe domains:")
    print()

    safe_domains = [
        "wikipedia.org",  # Education
        "github.com",  # Technology
        "stackoverflow.com",  # Forum
        "cnn.com",  # News
        "amazon.com",  # Shopping
        "google.com",  # Search Engine
        "apple.com",  # Technology
        "microsoft.com",  # Technology
    ]

    print("Safe test domains:")
    for domain in safe_domains:
        print(f"  - {domain}")
    print()

    print("Example safe classification:")
    print("```python")
    print("from piedomains import DomainClassifier")
    print("classifier = DomainClassifier()")
    print()
    print("# Test with safe, well-known domains")
    print("safe_domains = [")
    for domain in safe_domains[:4]:
        print(f"    '{domain}',")
    print("]")
    print()
    print("result = classifier.classify_by_text(safe_domains)")
    print("print(result[['domain', 'pred_label', 'pred_prob']])")
    print("```")


def create_docker_security_script():
    """Create a ready-to-use Docker security script."""
    print_section("CREATING SECURITY SCRIPT")

    script_content = """#!/bin/bash
# piedomains-secure: Run piedomains in secure Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔒 Starting secure piedomains environment..."

# Build container if needed
if [[ "$(docker images -q piedomains-sandbox 2> /dev/null)" == "" ]]; then
    echo "📦 Building piedomains sandbox container..."
    cd "$REPO_DIR"
    docker build -t piedomains-sandbox .
fi

# Run in secure container
echo "🚀 Running piedomains in isolated container..."
docker run -it --rm \\
    --name piedomains-secure \\
    --memory=2g \\
    --cpus=2 \\
    --read-only \\
    --tmpfs /tmp \\
    --tmpfs /var/tmp \\
    piedomains-sandbox bash

echo "✅ Secure session ended"
"""

    script_path = Path("examples/piedomains-secure.sh")
    print(f"Creating security script: {script_path}")
    print()
    print("Script content:")
    print(script_content)
    print()
    print("To use:")
    print("1. chmod +x examples/piedomains-secure.sh")
    print("2. ./examples/piedomains-secure.sh")

    return script_content


def main():
    """Main demonstration function."""
    print("🔒 piedomains Security Sandbox Demo")
    print("===================================")
    print()
    print("This demo shows how to safely run piedomains when analyzing")
    print("unknown or potentially suspicious domains.")
    print()
    print("⚠️  WARNING: Fetching content from unknown domains can be risky!")
    print("   - Malicious scripts could exploit browser vulnerabilities")
    print("   - Suspicious content might trigger security warnings")
    print("   - Screenshots could contain inappropriate content")
    print()
    print("✅ SOLUTION: Use isolated environments for unknown domains")

    # Demonstrate different approaches
    docker_available = run_docker_sandbox()
    run_macos_sandbox()
    demonstrate_network_isolation()
    safe_testing_demo()

    if docker_available:
        create_docker_security_script()

    print_section("SUMMARY & RECOMMENDATIONS")
    print()
    print("🥇 RECOMMENDED: Docker container isolation")
    print("   - Easy to set up and use")
    print("   - Complete system isolation")
    print("   - No permanent changes to host")
    print()
    print("🥈 ALTERNATIVE: macOS sandbox-exec")
    print("   - Built into macOS")
    print("   - Requires profile configuration")
    print("   - Less isolation than Docker")
    print()
    print("🥉 MINIMUM: Safe domain testing")
    print("   - Use known-good domains for testing")
    print("   - Avoid suspicious or unknown domains")
    print("   - Monitor system during execution")
    print()
    print("📚 Next steps:")
    print("1. Set up Docker for secure analysis")
    print("2. Test with safe domains first")
    print("3. Create isolated networks for suspicious domains")
    print("4. Consider VM isolation for high-risk analysis")


if __name__ == "__main__":
    main()
