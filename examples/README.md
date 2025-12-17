# Piedomains Examples

This directory contains examples demonstrating the piedomains library's capabilities.

## 🚀 Quick Start - New JSON API

The piedomains library now features a clean JSON-only API that separates data collection from inference:

```python
from piedomains import DomainClassifier

# Simple classification - returns JSON instead of DataFrames
classifier = DomainClassifier()
results = classifier.classify(["cnn.com", "github.com"])

for result in results:
    print(f"{result['domain']}: {result['category']} ({result['confidence']:.3f})")
    print(f"  Model: {result['model_used']}")
    print(f"  Data: {result['text_path']}, {result['image_path']}")
```

## 🔧 Separated Workflow

For advanced use cases, separate data collection from inference:

```python
from piedomains import DataCollector, DomainClassifier

# Step 1: Collect data (can be reused)
collector = DataCollector()
data = collector.collect(["example.com"])

# Step 2: Run inference (try different models on same data)
classifier = DomainClassifier()
text_results = classifier.classify_from_collection(data, method="text")
image_results = classifier.classify_from_collection(data, method="images")
```

## 📁 Available Examples

### Core Functionality
- `json_only_demo.py` - **NEW**: JSON-only API demonstration
- `separated_workflow_demo.py` - **NEW**: Data collection & inference separation
- `new_api_demo.py` - Traditional API (now returns JSON)
- `jupyter_demo.py` - Jupyter notebook examples

### Archive & Historical Analysis
- `final_archive_demo.py` - Archive.org integration
- Historical snapshots with `archive_date="20200101"`

### LLM-Powered Classification
- `llm_demo.py` - LLM-based classification with multiple providers

## 🔒 Security & Sandbox Examples

**⚠️ Important**: For unknown/suspicious domains, use the sandbox examples to protect your system:

```bash
# Safe, isolated domain classification
cd examples/sandbox
python3 secure_classify.py suspicious-domain.com --text-only

# Interactive secure mode
python3 secure_classify.py --interactive

# See all sandboxing options
python3 sandbox_demo.py
```

See **[`sandbox/`](sandbox/)** directory for complete security examples including Docker isolation, macOS sandboxing, and VM setup guides.

### LLM Demo Setup

For LLM examples, set your API key:
```bash
export OPENAI_API_KEY="sk-..."       # OpenAI
export ANTHROPIC_API_KEY="sk-ant-..." # Anthropic
export GOOGLE_API_KEY="..."          # Google
```

## Installation

Note: These scripts require the piedomains package to be installed:
```bash
pip install -e ..
```
