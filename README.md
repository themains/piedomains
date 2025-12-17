# piedomains: Classify website content using ML Models or LLMs

[![CI](https://github.com/themains/piedomains/actions/workflows/ci.yml/badge.svg)](https://github.com/themains/piedomains/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/piedomains.svg)](https://pypi.org/project/piedomains)
[![Downloads](https://pepy.tech/badge/piedomains)](https://pepy.tech/project/piedomains)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://themains.github.io/piedomains/)

## 🚀 What's New in v0.5.0

- **Playwright Migration**: Complete transition from Selenium to modern Playwright for faster, more reliable web content extraction
- **12.8x Performance Boost**: Optimized parallel processing (13.2s → 1.0s per domain)
- **Enhanced Docker Security**: Production-ready containerization with security sandboxing and resource limits
- **Unified Content Pipeline**: Text and image extraction now use the same Playwright engine for consistency

## Installation

```bash
pip install piedomains
```

Requires Python 3.11+

## Basic Usage

```python
from piedomains import DomainClassifier

classifier = DomainClassifier()
result = classifier.classify(["cnn.com", "amazon.com", "wikipedia.org"])
print(result[['domain', 'pred_label', 'pred_prob']])

# Output:
#        domain    pred_label  pred_prob
# 0     cnn.com          news   0.876543
# 1  amazon.com      shopping   0.923456
# 2 wikipedia.org   education   0.891234
```

## Classification Methods

```python
# Combined text + image analysis (most accurate)
result = classifier.classify(["github.com"])

# Text-only classification (faster)
result = classifier.classify_by_text(["news.google.com"])

# Image-only classification
result = classifier.classify_by_images(["instagram.com"])

# Batch processing
results = classifier.classify_batch(domains, method="text", batch_size=50)
```

## Historical Analysis

```python
# Analyze archived versions from archive.org
old_result = classifier.classify(["facebook.com"], archive_date="20100101")
```

## LLM Classification

```python
# Configure LLM provider
classifier.configure_llm(
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",
    categories=["news", "shopping", "social", "tech"]
)

# LLM-powered classification
result = classifier.classify_by_llm(["example.com"])

# With custom instructions
result = classifier.classify_by_llm(
    ["site.com"],
    custom_instructions="Classify by educational value"
)
```

Set API keys via environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## Categories

41 categories: news, finance, shopping, education, government, adult content, gambling, social networks, search engines, and others based on Shallalist taxonomy.

## Security & Docker

**v0.5.0** includes production-ready Docker containerization for secure domain analysis:

```bash
# Build secure sandbox container
docker build -t piedomains-sandbox .

# Run with security constraints (2GB RAM, 2 CPU, read-only filesystem)
docker run --rm --memory=2g --cpus=2 --read-only \
  --tmpfs /tmp --tmpfs /var/tmp \
  piedomains-sandbox python -c "
from piedomains import DomainClassifier
classifier = DomainClassifier()
result = classifier.classify(['example.com'])
print(result[['domain', 'pred_label']])
"
```

**Batch Processing in Container:**
```bash
# Use the included secure classification script
cd examples/sandbox
echo -e "wikipedia.org\ngithub.com\ncnn.com" > domains.txt
python3 secure_classify.py --file domains.txt
```

For testing, use known-safe domains: `["wikipedia.org", "github.com", "cnn.com"]`

## Documentation

- [API Reference](https://themains.github.io/piedomains/)
- [Examples](examples/)
- [Security Guide](examples/sandbox/)

## Development

```bash
git clone https://github.com/themains/piedomains
cd piedomains
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT License

## Citation

```bibtex
@software{piedomains,
  title={piedomains: AI-powered domain content classification},
  author={Chintalapati, Rajashekar and Sood, Gaurav},
  year={2024},
  url={https://github.com/themains/piedomains}
}
```
