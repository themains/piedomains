#!/bin/bash
# Docker entrypoint script for piedomains container
# Handles Playwright browser installation at runtime

set -e

BROWSER_CACHE_DIR="/home/playwright/.cache/ms-playwright"
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD="${PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:-0}"

echo "🐳 Starting piedomains container..."

# Function to check if browsers are installed
check_browsers() {
    if ls "$BROWSER_CACHE_DIR"/chromium-* 1> /dev/null 2>&1; then
        return 0  # Browsers found
    elif ls "$BROWSER_CACHE_DIR"/chromium_headless_shell-* 1> /dev/null 2>&1; then
        return 0  # Headless browser found
    else
        return 1  # Browsers not found
    fi
}

# Function to install browsers
install_browsers() {
    echo "📦 Installing Playwright browsers (this may take a few minutes)..."

    # Create cache directory with proper permissions
    mkdir -p "$BROWSER_CACHE_DIR"

    # Try installing browsers with retries
    local retries=3
    local count=0

    while [ $count -lt $retries ]; do
        if python -m playwright install chromium; then
            echo "✅ Browsers installed successfully"
            return 0
        else
            count=$((count + 1))
            if [ $count -lt $retries ]; then
                echo "⚠️  Browser installation failed, retrying ($count/$retries)..."
                sleep 2
            fi
        fi
    done

    echo "❌ Failed to install browsers after $retries attempts"
    echo "💡 You can skip browser installation by setting PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1"
    echo "💡 Or mount a pre-downloaded browser cache volume"
    return 1
}

# Main logic
if [ "$PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD" = "1" ]; then
    echo "⏭️  Skipping browser installation (PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1)"
elif check_browsers; then
    echo "✅ Browsers already installed"
else
    if ! install_browsers; then
        echo "⚠️  Continuing without browsers - some functionality may be limited"
    fi
fi

# Verify piedomains can be imported
echo "🔍 Verifying piedomains installation..."
if python -c "import piedomains; print('✅ piedomains imported successfully')"; then
    echo "🚀 piedomains container ready!"
else
    echo "❌ piedomains import failed"
    exit 1
fi

# Execute the provided command or default to interactive bash
if [ "$#" -eq 0 ]; then
    echo "🐚 Starting interactive shell..."
    exec bash
else
    echo "🏃 Executing: $*"
    exec "$@"
fi
