#!/usr/bin/env python3
"""
Test sandbox setup for piedomains

This script tests the various sandboxing options available on your Mac
and provides setup instructions if needed.
"""

import subprocess
from pathlib import Path


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def check_docker():
    """Check if Docker is available and working."""
    print_section("TESTING DOCKER/RANCHER DESKTOP")

    # Check for Docker binary
    docker_paths = [
        "/usr/local/bin/docker",
        "/opt/homebrew/bin/docker",
        "docker",  # In PATH
    ]

    docker_binary = None
    for path in docker_paths:
        try:
            subprocess.run([path, "--version"], check=True, capture_output=True)
            docker_binary = path
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if not docker_binary:
        print("❌ Docker not found or not running")
        print("\n💡 Setup instructions:")

        # Check for installed container systems
        rancher_exists = Path("/Applications/Rancher Desktop.app").exists()
        docker_exists = Path("/Applications/Docker.app").exists()

        if rancher_exists:
            print("✅ Rancher Desktop found")
            print("1. Start Rancher Desktop from Applications")
            print("2. Wait for it to fully start (check system tray)")
            print("3. Run this test again")
        elif docker_exists:
            print("✅ Docker Desktop found")
            print("1. Start Docker Desktop from Applications")
            print("2. Wait for it to fully start")
            print("3. Run this test again")
        else:
            print("📦 No container runtime found. Install one of:")
            print("- Rancher Desktop: https://rancherdesktop.io/")
            print(
                "- Docker Desktop: https://docs.docker.com/desktop/install/mac-install/"
            )

        return False, None

    # Test Docker functionality
    try:
        subprocess.run(
            [docker_binary, "ps"], check=True, capture_output=True, text=True
        )
        print(f"✅ Docker is working: {docker_binary}")

        # Test if we can run a simple container
        subprocess.run(
            [docker_binary, "run", "--rm", "hello-world"],
            check=True,
            capture_output=True,
        )
        print("✅ Container execution test passed")

        return True, docker_binary

    except subprocess.CalledProcessError as e:
        print("❌ Docker found but not working properly")
        print(f"Error: {e}")
        print("\n💡 Try:")
        print("1. Restart Docker/Rancher Desktop")
        print("2. Check Docker daemon is running")
        return False, docker_binary


def check_sandbox_exec():
    """Check macOS sandbox-exec availability."""
    print_section("TESTING MACOS SANDBOX-EXEC")

    try:
        # Test sandbox-exec with a simple command
        subprocess.run(
            ["sandbox-exec", "-p", "(version 1) (allow default)", "echo", "test"],
            check=True,
            capture_output=True,
        )
        print("✅ macOS sandbox-exec is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ sandbox-exec not available or not working")
        print("This is unusual on macOS - it should be built-in")
        return False


def test_piedomains_import():
    """Test if piedomains can be imported."""
    print_section("TESTING PIEDOMAINS INSTALLATION")

    try:
        # Try to create classifier (this will test dependencies)
        from piedomains import DomainClassifier

        DomainClassifier()
        print("✅ piedomains imported and DomainClassifier can be created")

        return True
    except ImportError as e:
        print(f"❌ piedomains import failed: {e}")
        print("\n💡 Install with: pip install -e .")
        return False
    except Exception as e:
        print(f"⚠️  piedomains imported but classifier failed: {e}")
        print("This might be OK - could be missing models")
        return False


def test_safe_classification():
    """Test classification with a known safe domain."""
    print_section("TESTING SAFE CLASSIFICATION")

    try:
        from piedomains import DomainClassifier

        classifier = DomainClassifier()

        print("🔍 Testing with example.com (text-only for safety)...")
        result = classifier.classify_by_text(["example.com"])
        print("✅ Safe classification test passed!")
        print(
            f"   Result: {result.iloc[0]['pred_label']} ({result.iloc[0]['pred_prob']:.3f})"
        )

        return True

    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False


def test_docker_build():
    """Test building the piedomains Docker container."""
    print_section("TESTING DOCKER CONTAINER BUILD")

    docker_available, docker_binary = check_docker()
    if not docker_available:
        print("⏭️  Skipping Docker build test - Docker not available")
        return False

    # Check if we're in the right directory
    repo_root = Path(__file__).parent.parent.parent
    dockerfile = repo_root / "Dockerfile"

    if not dockerfile.exists():
        print(f"❌ Dockerfile not found at {dockerfile}")
        return False

    print(f"🔍 Building container from {repo_root}")

    try:
        # Build the container
        subprocess.run(
            [docker_binary, "build", "-t", "piedomains-sandbox-test", str(repo_root)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )  # 5 min timeout

        print("✅ Docker container built successfully!")

        # Test running the container
        subprocess.run(
            [
                docker_binary,
                "run",
                "--rm",
                "piedomains-sandbox-test",
                "python",
                "-c",
                "from piedomains import DomainClassifier; print('Container test OK')",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

        print("✅ Container execution test passed!")

        # Clean up test image
        subprocess.run(
            [docker_binary, "rmi", "piedomains-sandbox-test"], capture_output=True
        )

        return True

    except subprocess.TimeoutExpired:
        print("❌ Docker build/test timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker build/test failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    """Run all sandbox setup tests."""
    print("🔒 piedomains Sandbox Setup Test")
    print("=" * 40)

    results = {}

    # Test each component
    results["docker"] = check_docker()[0]
    results["sandbox_exec"] = check_sandbox_exec()
    results["piedomains"] = test_piedomains_import()

    # If basic tests pass, try more advanced tests
    if results["piedomains"]:
        results["classification"] = test_safe_classification()

    if results["docker"]:
        results["docker_build"] = test_docker_build()

    # Summary
    print_section("SUMMARY")

    passed = sum(results.values())
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print()

    for test, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test}")

    print()

    if results.get("docker") and results.get("piedomains"):
        print("🎉 You're ready for secure Docker-based classification!")
        print()
        print("Try:")
        print("  python3 examples/sandbox/secure_classify.py example.com")
        print("  python3 examples/sandbox/secure_classify.py --interactive")

    elif results.get("sandbox_exec") and results.get("piedomains"):
        print("✅ You can use macOS sandbox-exec for security!")
        print()
        print("Try:")
        print(
            "  python3 examples/sandbox/secure_classify.py example.com --method macos-sandbox"
        )

    elif results.get("piedomains"):
        print("⚠️  piedomains works, but no sandboxing available")
        print("   Consider installing Docker/Rancher Desktop for security")
        print()
        print("For now, only use with trusted domains:")
        print('  python3 -c "from piedomains import DomainClassifier; "')
        print(
            "  python3 -c \"result = DomainClassifier().classify(['wikipedia.org']); print(result)\""
        )

    else:
        print("❌ Setup incomplete - check the errors above")
        print()
        print("Next steps:")
        print("1. Install piedomains: pip install -e .")
        print("2. Install Docker/Rancher Desktop")
        print("3. Run this test again")


if __name__ == "__main__":
    main()
