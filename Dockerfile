# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    # Playwright system dependencies
    libnss3 \
    libnspr4 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxss1 \
    libasound2 \
    libatspi2.0-0 \
    libgtk-3-0 \
    # Additional dependencies for headless operation
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy essential files for package installation
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN uv pip install --system -e .

# Install Playwright system dependencies
RUN python -m playwright install-deps chromium

# Create non-root user for security
RUN groupadd -r playwright && useradd -r -g playwright -G audio,video playwright \
    && mkdir -p /home/playwright/Downloads \
    && chown -R playwright:playwright /home/playwright \
    && chown -R playwright:playwright /app

# Copy entrypoint script first (as root)
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy source code (before switching user)
COPY . .

# Set ownership of the app directory (as root)
RUN chown -R playwright:playwright /app

# Switch to non-root user
USER playwright

# Set Playwright environment variables
ENV PLAYWRIGHT_BROWSERS_PATH=/home/playwright/.cache/ms-playwright

# Install browsers as playwright user
RUN python -m playwright install chromium

# Create cache directories
RUN mkdir -p /home/playwright/.cache/ms-playwright \
    && mkdir -p /app/cache/html \
    && mkdir -p /app/cache/images

# Expose port (if running web service)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Health check to verify installation
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import piedomains; print('✓ piedomains ready')" || exit 1

# Default command - interactive shell
CMD []
