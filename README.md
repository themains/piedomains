# piedomains: AI-powered domain content classification

[![CI](https://github.com/themains/piedomains/actions/workflows/ci.yml/badge.svg)](https://github.com/themains/piedomains/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/piedomains.svg)](https://pypi.python.org/pypi/piedomains)
[![Documentation](https://github.com/themains/piedomains/actions/workflows/docs.yml/badge.svg)](https://github.com/themains/piedomains/actions/workflows/docs.yml)

**piedomains** predicts website content categories using traditional ML models or modern LLMs (GPT-4, Claude, Gemini). Analyze domain names, text content, and homepage screenshots to classify websites as news, shopping, adult content, education, etc. with high accuracy and flexible custom categories.

## 🚀 Quickstart

Install and classify domains in 3 lines:

```python
pip install piedomains

from piedomains import DomainClassifier
classifier = DomainClassifier()

# Classify current content
result = classifier.classify(["cnn.com", "amazon.com", "wikipedia.org"])
print(result[['domain', 'pred_label', 'pred_prob']])

# Expected output:
#        domain    pred_label  pred_prob
# 0     cnn.com          news   0.876543
# 1  amazon.com      shopping   0.923456
# 2 wikipedia.org   education   0.891234
```

## 📊 Key Features

- **High Accuracy**: Combines text analysis + visual screenshots for 90%+ accuracy
- **LLM-Powered**: Use GPT-4o, Claude 3.5, Gemini with custom categories and instructions
- **Historical Analysis**: Classify websites from any point in time using archive.org
- **Fast & Scalable**: Batch processing with caching for 1000s of domains
- **Easy Integration**: Modern Python API with pandas output
- **Flexible Categories**: 41 default categories or define your own with AI models

## ⚡ Usage Examples

### Basic Classification

```python
from piedomains import DomainClassifier

classifier = DomainClassifier()

# Combined analysis (most accurate)
result = classifier.classify(["github.com", "reddit.com"])

# Text-only (faster)
result = classifier.classify_by_text(["news.google.com"])

# Images-only (good for visual content)  
result = classifier.classify_by_images(["instagram.com"])
```

### Historical Analysis

```python
# Analyze how Facebook looked in 2010 vs today
old_facebook = classifier.classify(["facebook.com"], archive_date="20100101")
new_facebook = classifier.classify(["facebook.com"])

print(f"2010: {old_facebook.iloc[0]['pred_label']}")
print(f"2024: {new_facebook.iloc[0]['pred_label']}")
```

### Batch Processing

```python
# Process large lists efficiently
domains = ["site1.com", "site2.com", ...] # 1000s of domains
results = classifier.classify_batch(
    domains, 
    method="text",           # text|images|combined
    batch_size=50,           # Process 50 at a time
    show_progress=True       # Progress bar
)
```

### 🤖 LLM-Powered Classification

Use modern AI models (GPT-4, Claude, Gemini) for flexible, accurate classification:

```python
from piedomains import DomainClassifier

classifier = DomainClassifier()

# Configure your preferred AI provider
classifier.configure_llm(
    provider="openai",           # openai, anthropic, google
    model="gpt-4o",              # multimodal model
    api_key="sk-...",            # or set via environment variable
    categories=["news", "shopping", "social", "tech", "education"]
)

# Text-only LLM classification
result = classifier.classify_by_llm(["cnn.com", "github.com"])

# Multimodal classification (text + screenshots)
result = classifier.classify_by_llm_multimodal(["instagram.com"])

# Custom classification instructions
result = classifier.classify_by_llm(
    ["khanacademy.org", "reddit.com"],
    custom_instructions="Classify by educational value: educational, entertainment, mixed"
)

# Track usage and costs
stats = classifier.get_llm_usage_stats()
print(f"API calls: {stats['total_requests']}, Cost: ${stats['estimated_cost_usd']:.4f}")
```

**LLM Benefits:**
- **Custom Categories**: Define your own classification schemes
- **Multimodal Analysis**: Combines text + visual understanding
- **Latest AI**: GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro
- **Cost Tracking**: Built-in usage monitoring and limits
- **Flexible Prompts**: Customize instructions for specific use cases

**Supported Providers:**
- **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus/Haiku
- **Google**: Gemini 1.5 Pro, Gemini Pro Vision
- **Others**: Any litellm-supported model

```bash
# Set API keys via environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## 🏷️ Supported Categories

News, Finance, Shopping, Education, Government, Adult Content, Gambling, Social Networks, Search Engines, and 32 more categories based on the Shallalist taxonomy.

## 📈 Performance

- **Speed**: ~10-50 domains/minute (depends on method and network)
- **Accuracy**: 85-95% depending on content type and method
- **Memory**: <500MB for batch processing
- **Caching**: Automatic content caching for faster re-runs

## 🔧 Installation

**Requirements**: Python 3.11+

```bash
# Basic installation
pip install piedomains

# For development
git clone https://github.com/themains/piedomains
cd piedomains
pip install -e .
```

## 💡 API Usage

```python
from piedomains import DomainClassifier
classifier = DomainClassifier()
result = classifier.classify_by_text(["example.com"])
```

## 📖 Documentation

- **API Reference**: https://piedomains.readthedocs.io
- **Examples**: `/examples` directory
- **Notebooks**: `/notebooks` (training & analysis)

## 🤝 Contributing

```bash
# Setup development environment
git clone https://github.com/themains/piedomains
cd piedomains
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check piedomains/
```

## 📄 License

MIT License - see LICENSE file.

## 📚 Citation

If you use piedomains in research, please cite:

```bibtex
@software{piedomains,
  title={piedomains: AI-powered domain content classification},
  author={Chintalapati, Rajashekar and Sood, Gaurav},
  year={2024},
  url={https://github.com/themains/piedomains}
}
```