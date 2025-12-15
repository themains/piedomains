# 🔒 Security & Sandbox Examples

This directory contains comprehensive examples for running piedomains safely in isolated environments to protect against potentially malicious domains.

## ⚠️ Why Sandbox?

When piedomains analyzes domains, it:
- Fetches live web content via HTTP
- Takes screenshots using Chrome browser
- Processes HTML/JavaScript that could contain malicious code
- Downloads images and other assets

**For unknown or suspicious domains, always use sandboxed environments.**

## 🚀 Quick Start

### 1. Docker Sandbox (Recommended)

```bash
# Run the secure script
python3 secure_classify.py wikipedia.org github.com --text-only

# Interactive mode
python3 secure_classify.py --interactive

# From file
echo -e "suspicious-domain.com\nunknown-site.org" > domains.txt
python3 secure_classify.py --file domains.txt
```

### 2. Manual Docker Setup

```bash
# Build container
cd ../../  # Go to repo root
docker build -t piedomains-sandbox .

# Run secure shell
./examples/sandbox/piedomains-secure.sh
```

### 3. macOS Sandbox (Native)

```bash
# Run with built-in macOS sandboxing
python3 secure_classify.py example.com --method macos-sandbox
```

## 📁 Files in This Directory

### Scripts
- **`secure_classify.py`** - Main secure classification script with multiple isolation methods
- **`piedomains-secure.sh`** - Docker container launcher script
- **`sandbox_demo.py`** - Comprehensive demo of all sandboxing approaches

### Configuration
- **`macos_sandbox_profile.sb`** - macOS sandbox-exec security profile
- **`security-sandbox.md`** - Detailed security documentation and setup guides

### Usage Examples

```bash
# Safe domains (for testing)
python3 secure_classify.py wikipedia.org github.com cnn.com

# Suspicious domains (use sandbox!)
python3 secure_classify.py suspicious-domain.tk --text-only

# Batch processing from file
python3 secure_classify.py --file domain_list.txt --output results/

# Interactive classification
python3 secure_classify.py --interactive
```

## 🛡️ Security Methods

### 1. Docker Container (Best)
- ✅ Complete filesystem isolation
- ✅ Memory/CPU limits
- ✅ Network restrictions
- ✅ Read-only container
- ✅ Easy cleanup

### 2. macOS sandbox-exec (Native)
- ✅ Built into macOS
- ✅ File system restrictions
- ✅ Network controls
- ⚠️ Requires profile configuration

### 3. VM Isolation (Maximum Security)
- ✅ Complete OS isolation
- ✅ Network isolation
- ✅ Snapshot/restore capability
- ⚠️ Resource intensive

## 🧪 Safe Testing Domains

Start with these verified safe domains:

```python
safe_domains = [
    "wikipedia.org",      # Education
    "github.com",         # Technology
    "stackoverflow.com",  # Forum
    "bbc.com",           # News
    "amazon.com",        # Shopping
    "python.org",        # Technology
    "google.com",        # Search
]
```

## 🔧 Advanced Usage

### Text-Only Analysis (Safer)
```bash
# Analyze content without taking screenshots
python3 secure_classify.py unknown-domain.com --text-only
```

### Custom Output Directory
```bash
# Save results to specific location
python3 secure_classify.py domains.com --output /secure/results/
```

### Maximum Security Container
```bash
# Run with extra restrictions
docker run --rm -it \
    --memory=1g \
    --cpus=1 \
    --read-only \
    --tmpfs /tmp \
    --network=none \
    piedomains-sandbox bash
```

## 🚨 Warning Signs

Stop and use maximum security if you encounter:

- Domains with suspicious TLDs (`.tk`, `.ml`, `.ga`, `.onion`)
- Shortened URLs from unknown services
- Domains from suspicious email/messages
- Any domain you wouldn't visit in your main browser

## 📚 Documentation

- **`security-sandbox.md`** - Complete security guide with VM setup, firewall rules, and incident response
- **`sandbox_demo.py`** - Run to see all available sandboxing methods
- **Main README** - Basic security section in the project root

## ⚡ Quick Commands

```bash
# See all sandboxing options
python3 sandbox_demo.py

# Classify safely with Docker
python3 secure_classify.py suspicious-domain.com

# Interactive secure mode
python3 secure_classify.py --interactive

# Check if Docker is available
docker --version

# Make scripts executable
chmod +x piedomains-secure.sh
```

## 🆘 Emergency Actions

If something goes wrong:
1. **Disconnect network**: `sudo ifconfig en0 down`
2. **Kill processes**: `sudo pkill -f chrome`
3. **Check connections**: `netstat -an | grep ESTABLISHED`
4. **Clean temp files**: `rm -rf /tmp/chrome* /tmp/scoped_dir*`

For detailed incident response, see `security-sandbox.md`.
