# Examples

This directory contains example scripts demonstrating piedomains functionality:

## Traditional ML Classification
- `new_api_demo.py`: Modern DomainClassifier API demonstration
- `archive_demo.py`: Basic archive.org classification demo
- `archive_functionality_demo.py`: Archive functionality testing
- `final_archive_demo.py`: Final archive integration test
- `jupyter_demo.py`: Jupyter notebook demonstration

## LLM-Powered Classification
- `llm_demo.py`: LLM-based classification with OpenAI, Anthropic, Google models

## Running Examples

```bash
cd examples
python new_api_demo.py
python llm_demo.py  # Requires API key
```

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
