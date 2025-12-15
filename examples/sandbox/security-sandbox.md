# Security & Sandboxing Guide for piedomains

## Overview

When using piedomains to analyze unknown or potentially suspicious domains, it's crucial to protect your system from security risks. This guide provides comprehensive instructions for running piedomains safely in isolated environments on macOS.

## ⚠️ Security Risks

### Why Sandboxing Matters

When piedomains analyzes domains, it:
- **Fetches live web content** via HTTP requests
- **Takes screenshots** using a real browser (Chrome/Chromium)
- **Downloads and processes** HTML, CSS, JavaScript, and images
- **Executes browser code** that could contain malicious scripts

**Potential risks include:**
- Malicious JavaScript execution
- Browser exploit attempts
- Suspicious content download
- Privacy/tracking concerns
- System resource consumption
- Network-based attacks

## 🛡️ Sandboxing Solutions for macOS

### 1. Docker Container Isolation (Recommended)

Docker provides the best isolation with minimal setup complexity.

#### Setup

Requires Docker, Docker Desktop, Rancher Desktop, or similar container runtime.

```bash
# Navigate to piedomains directory
cd /path/to/piedomains

# Build secure container
docker build -t piedomains-sandbox .
```

**Container Runtime Options:**
- **Rancher Desktop**: Excellent Docker alternative with Kubernetes
- **Docker Desktop**: Official Docker distribution
- **Podman**: Daemonless container engine
- **Lima**: Lightweight VM-based containers

#### Basic Usage

```bash
# Run single classification in container
docker run --rm piedomains-sandbox python -c "
from piedomains import DomainClassifier
classifier = DomainClassifier()
result = classifier.classify(['suspicious-domain.com'])
print(result[['domain', 'pred_label', 'pred_prob']])
"

# Interactive session
docker run -it --rm piedomains-sandbox bash
```

#### Enhanced Security Options

```bash
# Maximum security container
docker run --rm -it \
    --name piedomains-secure \
    --memory=2g \              # Limit memory
    --cpus=2 \                 # Limit CPU cores
    --read-only \              # Read-only filesystem
    --tmpfs /tmp \             # Temporary filesystem
    --tmpfs /var/tmp \
    --network=none \           # No network (requires pre-cached content)
    --user=nobody \            # Non-root user
    --cap-drop=ALL \           # Drop all capabilities
    piedomains-sandbox bash
```

#### Persistent Results

```bash
# Mount output directory to save results
docker run --rm -it \
    -v "$(pwd)/sandbox-output:/app/output" \
    piedomains-sandbox python -c "
import pandas as pd
from piedomains import DomainClassifier

classifier = DomainClassifier()
result = classifier.classify(['example.com'])
result.to_csv('/app/output/results.csv', index=False)
print('Results saved to sandbox-output/results.csv')
"
```

### 2. macOS Sandbox-exec (Native)

macOS includes built-in sandboxing via `sandbox-exec`.

#### Create Sandbox Profile

Save as `piedomains.sb`:

```scheme
(version 1)

; Deny everything by default
(deny default)

; Allow Python execution
(allow process-exec
    (literal "/usr/bin/python3")
    (literal "/usr/local/bin/python3")
    (subpath "/opt/homebrew/bin"))

; Allow reading system libraries
(allow file-read*
    (subpath "/usr/lib")
    (subpath "/System/Library")
    (subpath "/opt/homebrew")
    (literal "/dev/urandom")
    (literal "/dev/null"))

; Allow temporary files
(allow file-read* file-write*
    (subpath "$(TMPDIR)")
    (subpath "/tmp"))

; Allow current directory (for piedomains)
(allow file-read*
    (subpath "$(PWD)"))

; Allow writing to output directory only
(allow file-write*
    (subpath "$(PWD)/sandbox-output"))

; Allow network for domain fetching
(allow network-outbound
    (remote ip))

; Allow system info
(allow sysctl-read)

; Allow DNS resolution
(allow network-outbound
    (remote ip "8.8.8.8")  ; Google DNS
    (remote ip "1.1.1.1")) ; Cloudflare DNS
```

#### Usage

```bash
# Create output directory
mkdir -p sandbox-output

# Run with sandbox
sandbox-exec -f piedomains.sb python3 -c "
from piedomains import DomainClassifier
import pandas as pd

classifier = DomainClassifier()
result = classifier.classify(['example.com'])
result.to_csv('sandbox-output/results.csv')
print('Classification completed safely')
"
```

### 3. Virtual Machine Isolation

For maximum security, use a dedicated VM.

#### VM Setup (UTM on Apple Silicon)

1. **Install UTM**: Download from https://mac.getutm.app/
2. **Create Ubuntu VM**:
   - Download Ubuntu 22.04 ARM64
   - Allocate 4GB RAM, 20GB storage
   - Enable network isolation

3. **Install Dependencies**:
```bash
# In VM
sudo apt update
sudo apt install -y python3-pip git docker.io
git clone https://github.com/themains/piedomains
cd piedomains
pip3 install -e .
```

4. **Analyze Domains**:
```bash
python3 -c "
from piedomains import DomainClassifier
classifier = DomainClassifier()
result = classifier.classify(['suspicious-domain.com'])
print(result)
"
```

#### VM Security Benefits
- Complete OS isolation
- Network can be fully isolated
- Snapshots for quick reset
- No impact on host system

### 4. Network Isolation Techniques

#### Firewall Rules

Create temporary firewall rules to limit network access:

```bash
# Block specific IPs (example)
sudo pfctl -f /dev/stdin << EOF
block out to 192.168.1.0/24
block out to 10.0.0.0/8
pass out to any port 80,443
EOF

# Run piedomains
python3 your_script.py

# Reset firewall
sudo pfctl -f /etc/pf.conf
```

#### VPN/Proxy Isolation

Route traffic through isolated networks:

```bash
# Use specific network interface
export HTTP_PROXY=http://isolated-proxy:8080
export HTTPS_PROXY=http://isolated-proxy:8080

python3 -c "
from piedomains import DomainClassifier
classifier = DomainClassifier()
result = classifier.classify(['domain.com'])
"
```

## 🧪 Safe Testing Practices

### Known-Safe Domains

Start with these verified safe domains for testing:

```python
safe_domains = [
    # Education
    "wikipedia.org",
    "khanacademy.org",
    "coursera.org",

    # Technology
    "github.com",
    "stackoverflow.com",
    "python.org",

    # News (reputable)
    "bbc.com",
    "reuters.com",
    "cnn.com",

    # E-commerce (major)
    "amazon.com",
    "ebay.com",
    "apple.com",

    # Search engines
    "google.com",
    "bing.com",
    "duckduckgo.com"
]
```

### Text-Only Classification (Safer)

For unknown domains, start with text-only analysis (no screenshots):

```python
from piedomains import DomainClassifier

classifier = DomainClassifier()

# Safer: text-only analysis
result = classifier.classify_by_text(['unknown-domain.com'])

# Only take screenshots if text analysis shows safe category
if result.iloc[0]['pred_label'] in ['education', 'news', 'government']:
    result = classifier.classify(['unknown-domain.com'])  # Full analysis
```

### Progressive Security Approach

1. **Text-only analysis first**
2. **Check reputation** (VirusTotal, URLVoid)
3. **Manual review** of domain/URL
4. **Screenshot analysis** in sandbox only

## 🔧 Automation Scripts

### Secure Batch Processing

Create `secure-classify.py`:

```python
#!/usr/bin/env python3
"""
Secure batch domain classification script
"""

import pandas as pd
import subprocess
import tempfile
import os
from pathlib import Path

def classify_in_docker(domains, output_file):
    """Run classification in Docker container."""

    # Create temporary script
    script_content = f"""
from piedomains import DomainClassifier
import pandas as pd

domains = {domains}
classifier = DomainClassifier()

try:
    result = classifier.classify_by_text(domains)  # Safer text-only
    result.to_csv('/app/output/{output_file}', index=False)
    print(f"✅ Classified {{len(domains)}} domains successfully")
except Exception as e:
    print(f"❌ Error: {{e}}")
    pd.DataFrame({{'domain': domains, 'error': str(e)}}).to_csv('/app/output/errors.csv')
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        # Run in Docker
        subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{temp_script}:/app/classify.py",
            "-v", f"{os.getcwd()}/output:/app/output",
            "piedomains-sandbox",
            "python", "/app/classify.py"
        ], check=True)
    finally:
        os.unlink(temp_script)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 secure-classify.py domain1 domain2 ...")
        sys.exit(1)

    domains = sys.argv[1:]
    os.makedirs("output", exist_ok=True)

    print(f"🔒 Securely classifying {len(domains)} domains...")
    classify_in_docker(domains, "secure-results.csv")
    print("📄 Results saved to output/secure-results.csv")
```

### Security Monitoring

Monitor system during analysis:

```bash
#!/bin/bash
# security-monitor.sh

echo "🔍 Starting security monitoring..."

# Monitor network connections
netstat -an | grep ESTABLISHED > baseline-connections.txt

# Monitor running processes
ps aux > baseline-processes.txt

echo "📊 Baseline established. Running piedomains..."

# Run your piedomains analysis here
python3 your_classification_script.py

echo "🔍 Checking for changes..."

# Check for new connections
netstat -an | grep ESTABLISHED > current-connections.txt
diff baseline-connections.txt current-connections.txt

# Check for new processes
ps aux > current-processes.txt
diff baseline-processes.txt current-processes.txt

echo "✅ Security monitoring complete"
```

## 🚨 Incident Response

If you suspect malicious activity:

### Immediate Actions

1. **Disconnect network**: `sudo ifconfig en0 down`
2. **Kill browser processes**: `sudo pkill -f chrome`
3. **Check running processes**: `ps aux | grep -E "(chrome|python)"`
4. **Review network connections**: `netstat -an | grep ESTABLISHED`

### Investigation

```bash
# Check recent file modifications
find /tmp -mtime -1 -type f

# Check downloaded files
find ~/Downloads -mtime -1 -type f

# Check browser history/cache
ls -la ~/Library/Application\ Support/Google/Chrome/Default/

# Check system logs
tail -f /var/log/system.log
```

### System Cleanup

```bash
# Clear browser data
rm -rf ~/Library/Application\ Support/Google/Chrome/Default/Cache
rm -rf /tmp/chrome*
rm -rf /tmp/scoped_dir*

# Clear piedomains cache
rm -rf ~/.cache/piedomains
rm -rf /tmp/piedomains*

# Reset network
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
```

## ✅ Best Practices Summary

### Before Analysis
- [ ] Review domain reputation
- [ ] Set up isolated environment
- [ ] Limit network access if possible
- [ ] Use text-only analysis for unknowns

### During Analysis
- [ ] Monitor system resources
- [ ] Watch network connections
- [ ] Log all activities
- [ ] Limit batch sizes

### After Analysis
- [ ] Review results for anomalies
- [ ] Clean temporary files
- [ ] Check system for changes
- [ ] Backup results securely

### Emergency Procedures
- [ ] Network disconnection plan
- [ ] Process termination commands
- [ ] System restoration steps
- [ ] Incident reporting process

## 📚 Additional Resources

- **Docker Security**: https://docs.docker.com/engine/security/
- **macOS Sandbox**: `man sandbox-exec`
- **UTM Virtualization**: https://mac.getutm.app/support/
- **Network Security**: https://support.apple.com/guide/mac-help/control-access-to-your-mac-mh11783/

For questions or security concerns, please open an issue on the piedomains GitHub repository.
