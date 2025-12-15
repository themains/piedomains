#!/bin/bash
# piedomains-secure: Run piedomains in secure Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "🔒 Starting secure piedomains environment..."

# Build container if needed
if [[ "$(docker images -q piedomains-sandbox 2> /dev/null)" == "" ]]; then
    echo "📦 Building piedomains sandbox container..."
    cd "$REPO_DIR"
    docker build -t piedomains-sandbox .
fi

# Create output directory
mkdir -p "$SCRIPT_DIR/output"

# Run in secure container
echo "🚀 Running piedomains in isolated container..."
docker run -it --rm \
    --name piedomains-secure \
    --memory=2g \
    --cpus=2 \
    --read-only \
    --tmpfs /tmp \
    --tmpfs /var/tmp \
    -v "$SCRIPT_DIR/output:/app/output" \
    piedomains-sandbox bash

echo "✅ Secure session ended"
echo "📁 Results saved in: $SCRIPT_DIR/output"
