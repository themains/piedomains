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