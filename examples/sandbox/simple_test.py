#!/usr/bin/env python3
"""Simple test of sandbox functionality without piedomains dependencies."""

import subprocess


def test_macos_sandbox():
    """Test macOS sandbox-exec with a simple command."""
    print("Testing macOS sandbox-exec...")

    try:
        # Test sandbox-exec with a simple command
        result = subprocess.run(
            [
                "sandbox-exec",
                "-p",
                "(version 1) (allow default)",
                "python3",
                "-c",
                "print('Sandbox test successful!')",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        print("✅ macOS sandbox-exec works!")
        print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Sandbox test failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ sandbox-exec not found")
        return False


def test_docker():
    """Test Docker availability."""
    print("Testing Docker...")

    docker_paths = ["/usr/local/bin/docker", "docker"]

    for docker_path in docker_paths:
        try:
            result = subprocess.run(
                [docker_path, "--version"], check=True, capture_output=True, text=True
            )
            print(f"✅ Docker found: {docker_path}")
            print(f"Version: {result.stdout.strip()}")
            return True, docker_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    print("❌ Docker not found or not running")
    return False, None


def main():
    """Run basic sandbox tests."""
    print("🔒 Basic Sandbox Functionality Test")
    print("=" * 40)

    sandbox_ok = test_macos_sandbox()
    docker_ok, docker_path = test_docker()

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)

    if sandbox_ok:
        print("✅ macOS sandbox-exec is ready")
        print(
            "   You can use: sandbox-exec -f macos_sandbox_profile.sb python3 your_script.py"
        )

    if docker_ok:
        print("✅ Docker is ready")
        print("   You can build containers for isolation")

    if sandbox_ok or docker_ok:
        print("\n🎉 You have sandboxing capabilities!")
        print("   The security examples should work on your system")
    else:
        print("\n❌ No sandboxing available")
        print("   Consider starting Rancher Desktop or installing Docker")


if __name__ == "__main__":
    main()
