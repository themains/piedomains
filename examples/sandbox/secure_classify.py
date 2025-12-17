#!/usr/bin/env python3
"""
Secure batch domain classification script

This script demonstrates safe domain classification using Docker isolation
to protect against potentially malicious content.

Usage:
    python3 secure_classify.py domain1.com domain2.com domain3.com
    python3 secure_classify.py --file domains.txt
    python3 secure_classify.py --interactive

Examples:
    python3 secure_classify.py wikipedia.org github.com
    python3 secure_classify.py --file suspicious_domains.txt --text-only
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check_docker():
    """Check if Docker is available."""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def classify_in_docker(domains, text_only=True, output_dir="output"):
    """Run classification in Docker container."""

    # Prepare classification method
    method = "classify_by_text" if text_only else "classify"

    # Create temporary script
    script_content = f"""
import sys
import os
sys.path.insert(0, '/app')

from piedomains import DomainClassifier
import pandas as pd

domains = {domains}
print(f"🔍 Analyzing {{len(domains)}} domains using {method}...")

classifier = DomainClassifier()

try:
    result = classifier.{method}(domains)

    # Save results
    os.makedirs('/app/output', exist_ok=True)
    result.to_csv('/app/output/results.csv', index=False)

    # Print summary
    print("\\n✅ Classification Results:")
    print("=" * 50)

    # Determine which columns to use based on method
    if '{method}' == 'classify_by_text':
        label_col, prob_col = 'text_label', 'text_prob'
    elif '{method}' == 'classify_by_images':
        label_col, prob_col = 'image_label', 'image_prob'
    else:  # combined classify method
        label_col, prob_col = 'pred_label', 'pred_prob'

    print(result[['domain', label_col, prob_col]].to_string(index=False))
    print(f"\\n📄 Detailed results saved to output/results.csv")

    # Category summary
    print("\\n📊 Category Summary:")
    print("-" * 30)
    category_counts = result[label_col].value_counts()
    for category, count in category_counts.items():
        print(f"{{category:15s}}: {{count:3d}} domains")

except Exception as e:
    print(f"❌ Error during classification: {{e}}")
    error_df = pd.DataFrame({{'domain': domains, 'error': str(e)}})
    error_df.to_csv('/app/output/errors.csv', index=False)
    sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Build container if needed
        try:
            subprocess.run(
                ["docker", "inspect", "piedomains-sandbox"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            print("📦 Building piedomains sandbox container...")
            repo_dir = Path(__file__).parent.parent.parent
            subprocess.run(
                ["docker", "build", "-t", "piedomains-sandbox", str(repo_dir)],
                check=True,
            )

        print("🔒 Running classification in secure Docker container...")

        # Run in Docker with security restrictions
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--memory=2g",  # Limit memory
                "--cpus=2",  # Limit CPU
                "--read-only",  # Read-only filesystem
                "--tmpfs",
                "/tmp",  # Temporary filesystem
                "--tmpfs",
                "/var/tmp",
                # Use pre-installed browsers (read-only is fine for browsers)
                "--tmpfs",
                "/app/cache:rw,uid=995,gid=995",  # piedomains cache with correct playwright user permissions
                "-v",
                f"{temp_script}:/app/classify.py",
                "-v",
                f"{os.path.abspath(output_dir)}:/app/output",
                "piedomains-sandbox",
                "python",
                "/app/classify.py",
            ],
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Docker execution failed: {e}")
        return False
    finally:
        os.unlink(temp_script)


def classify_with_macos_sandbox(domains, text_only=True):
    """Run classification with macOS sandbox-exec."""

    script_dir = Path(__file__).parent
    sandbox_profile = script_dir / "macos_sandbox_profile.sb"

    if not sandbox_profile.exists():
        print(f"❌ Sandbox profile not found: {sandbox_profile}")
        return False

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Create temporary Python script
    method = "classify_by_text" if text_only else "classify"
    script_content = f"""
from piedomains import DomainClassifier
import pandas as pd

domains = {domains}
print(f"🔍 Analyzing {{len(domains)}} domains with macOS sandbox...")

classifier = DomainClassifier()
result = classifier.{method}(domains)

result.to_csv('output/results.csv', index=False)
print("✅ Results saved to output/results.csv")
print(result[['domain', 'pred_label', 'pred_prob']])
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        print("🔒 Running with macOS sandbox-exec...")
        subprocess.run(
            ["sandbox-exec", "-f", str(sandbox_profile), "python3", temp_script],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Sandbox execution failed: {e}")
        return False
    finally:
        os.unlink(temp_script)


def read_domains_from_file(filepath):
    """Read domains from a text file."""
    domains = []
    with open(filepath) as f:
        for line in f:
            domain = line.strip()
            if domain and not domain.startswith("#"):
                domains.append(domain)
    return domains


def interactive_mode():
    """Interactive domain classification."""
    print("🔒 Interactive Secure Domain Classification")
    print("=" * 45)
    print("Enter domains one per line. Press Ctrl+C or enter 'quit' to finish.")
    print()

    domains = []
    try:
        while True:
            domain = input(f"Domain {len(domains) + 1}: ").strip()
            if domain.lower() in ["quit", "exit", ""]:
                break
            if domain:
                domains.append(domain)
    except KeyboardInterrupt:
        print("\n")

    if not domains:
        print("No domains entered.")
        return

    print(f"\n📝 Collected {len(domains)} domains:")
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")

    response = input("\nProceed with classification? (y/N): ")
    if response.lower().startswith("y"):
        return domains
    else:
        print("Classification cancelled.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Secure domain classification using sandboxed environments",
        epilog="""
Examples:
  %(prog)s wikipedia.org github.com
  %(prog)s --file domains.txt --text-only
  %(prog)s --interactive --method docker
  %(prog)s suspicious-domain.com --method macos-sandbox
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("domains", nargs="*", help="Domain names to classify")
    parser.add_argument(
        "--file", "-f", type=str, help="Read domains from file (one per line)"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode for entering domains",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["docker", "macos-sandbox", "auto"],
        default="auto",
        help="Sandboxing method to use",
    )
    parser.add_argument(
        "--text-only",
        "-t",
        action="store_true",
        help="Use text-only analysis (safer, no screenshots)",
    )
    parser.add_argument(
        "--output", "-o", default="output", help="Output directory for results"
    )

    args = parser.parse_args()

    # Collect domains
    domains = []

    if args.interactive:
        domains = interactive_mode()
        if not domains:
            return
    elif args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        domains = read_domains_from_file(args.file)
    else:
        domains = args.domains

    if not domains:
        print("❌ No domains provided. Use --help for usage information.")
        sys.exit(1)

    print(f"🔍 Preparing to classify {len(domains)} domains")

    # Safety check for suspicious domains
    suspicious_tlds = [".tk", ".ml", ".ga", ".cf", ".onion"]
    suspicious_domains = [
        d for d in domains if any(d.endswith(tld) for tld in suspicious_tlds)
    ]

    if suspicious_domains:
        print("⚠️  Warning: Detected potentially suspicious domains:")
        for domain in suspicious_domains:
            print(f"   - {domain}")
        response = input("Continue anyway? (y/N): ")
        if not response.lower().startswith("y"):
            print("Classification cancelled for safety.")
            sys.exit(0)

    # Choose sandboxing method
    if args.method == "auto":
        if check_docker():
            method = "docker"
        else:
            method = "macos-sandbox"
    else:
        method = args.method

    print(f"🛡️  Using {method} sandboxing")

    # Perform classification
    success = False

    if method == "docker":
        if not check_docker():
            print(
                "❌ Docker not available. Install Docker or use --method macos-sandbox"
            )
            sys.exit(1)
        success = classify_in_docker(domains, args.text_only, args.output)
    elif method == "macos-sandbox":
        success = classify_with_macos_sandbox(domains, args.text_only)

    if success:
        print("\n✅ Secure classification completed!")
        print(f"📁 Results available in: {args.output}/")
    else:
        print("\n❌ Classification failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
