# piedomains

[![CI](https://github.com/themains/piedomains/actions/workflows/ci.yml/badge.svg)](https://github.com/themains/piedomains/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/piedomains.svg)](https://pypi.org/project/piedomains)
[![Downloads](https://pepy.tech/badge/piedomains)](https://pepy.tech/project/piedomains)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://themains.github.io/piedomains/)

Classify website content categories using machine learning models or LLMs (GPT-4, Claude, Gemini).

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

## Security

When analyzing unknown domains, use Docker or isolated environments:

```bash
docker build -t piedomains-sandbox .
docker run --rm -it piedomains-sandbox python -c "
from piedomains import DomainClassifier
classifier = DomainClassifier()
result = classifier.classify(['example.com'])
print(result[['domain', 'pred_label']])
"
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
