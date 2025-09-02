# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
- Run all tests: `pytest piedomains/tests/ -v`
- Run tests without ML models: `pytest piedomains/tests/ -v -m "not ml"`
- Run specific test: `pytest piedomains/tests/test_001_pred_domain_text.py`
- Run with coverage: `pytest piedomains/tests/ --cov=piedomains`

### Linting and Code Quality
- Run pylint: `pylint piedomains/` (uses configuration from `pylintrc`)
- Run flake8 (CI configuration): 
  - Syntax errors: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - General linting: `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics`

### Installation and Development
- Install package: `pip install -e .` (from repository root)
- Install with dev dependencies: `pip install -e ".[dev]"`
- Console script: `classify_domains` (entry point defined in pyproject.toml)

### Package Management
- Build package: `python -m build`
- Upload to PyPI: `python -m twine upload dist/*`
- Validate README: `python -c "import docutils.core; docutils.core.publish_doctree(open('README.rst').read())"`

### Documentation
- Build docs: `cd docs && make html`
- Documentation is built with Sphinx and deployed to ReadTheDocs

## Architecture

### v0.3.0+ Modern Architecture

**New API Design (`api.py`)**: Modern, user-friendly interface
- `DomainClassifier`: Main class with intuitive methods
  - `.classify()`: Combined text + image analysis (most accurate)
  - `.classify_by_text()`: Text-only analysis (faster)
  - `.classify_by_images()`: Image-only analysis (visual content)
  - `.classify_batch()`: Batch processing with progress tracking
- `classify_domains()`: Convenience function for quick usage
- Archive.org integration for historical analysis

**Modular Classifiers (`classifiers/`)**:
- `TextClassifier`: Specialized text content analysis
- `ImageClassifier`: Screenshot-based visual analysis  
- `CombinedClassifier`: Ensemble approach combining both modalities

**Content Processors (`processors/`)**:
- `TextProcessor`: HTML parsing, text extraction and cleaning
- `ContentProcessor`: Content fetching and caching logic

**Legacy API (`domain.py`)**: Backward-compatible functions
- `pred_shalla_cat_*()` functions preserved for existing users
- Will show deprecation warnings in future versions

**Core Engine (`piedomain.py`)**: Low-level prediction engine with ML pipeline
- TensorFlow model inference with proper memory management
- Batch processing with configurable sizes
- Resource cleanup with context managers

### Machine Learning Pipeline

1. **Content Fetching**: 
   - Live content: HTTP requests with retry logic and connection pooling
   - Historical content: Archive.org integration with `ArchiveFetcher`
   - Caching: Automatic file-based caching for reuse
2. **Text Processing**: 
   - HTML parsing with BeautifulSoup
   - Text extraction and cleaning (removing non-English words, stopwords, punctuation)
   - NLTK-based text preprocessing with fallbacks
3. **Image Processing**: 
   - Screenshot capture via Selenium WebDriver with proper resource management
   - Image resizing to 254x254 with PIL
   - Tensor preprocessing for CNN model
4. **Model Inference**: 
   - TensorFlow 2.11+ models with explicit memory cleanup
   - Batch processing with configurable sizes for scalability
   - Text model calibration using isotonic regression
5. **Ensemble**: Final predictions combine text and image probabilities with equal weighting

### Data Flow Architecture
- **Input**: List of domain names or URLs, optional archive dates
- **Fetching**: Modular fetcher system (`LiveFetcher`/`ArchiveFetcher`)
- **Processing**: Separate text and image processing pipelines
- **Inference**: TensorFlow models with batch optimization and memory management
- **Output**: Pandas DataFrame with predictions, probabilities, and comprehensive metadata
- **Cleanup**: Automatic resource cleanup (WebDriver, temp files, tensors)

### Categories
The model predicts among 41 Shallalist categories defined in `constants.py` including: adv, alcohol, automobile, dating, downloads, drugs, education, finance, forum, gamble, government, news, politics, porn, recreation, shopping, socialnet, etc.

### Model Storage
- **Download**: Models automatically downloaded from Harvard Dataverse on first use
- **Cache Structure**: `model/shallalist/` directory structure
- **Text Model**: `saved_model/piedomains/` (TensorFlow SavedModel format)
- **Image Model**: `saved_model/pydomains_images/` (TensorFlow SavedModel format)  
- **Calibrators**: `calibrate/text/*.sav` files (scikit-learn isotonic regression)
- **Version Management**: `latest=True` parameter forces model updates

### Key Dependencies & Architecture
- **TensorFlow 2.11-2.15**: Neural network inference with memory management
- **Selenium 4.8**: WebDriver automation with context manager cleanup
- **NLTK**: Text processing with lazy initialization and fallbacks
- **scikit-learn 1.5**: Model calibration and post-processing
- **BeautifulSoup4**: HTML parsing and content extraction
- **Pillow 10.3**: Image processing and tensor conversion
- **webdriver-manager**: Automatic ChromeDriver management
- **pandas 1.4**: DataFrame output and data manipulation

## Usage Patterns

### Modern API (Recommended)
```python
from piedomains import DomainClassifier

classifier = DomainClassifier()
result = classifier.classify(["cnn.com", "amazon.com"])
```

### Legacy API (Backward Compatible)
```python
from piedomains import pred_shalla_cat
result = pred_shalla_cat(["cnn.com", "amazon.com"])
```

### Archive Analysis
```python
# Historical content from 2020
result = classifier.classify(["facebook.com"], archive_date="20200101")
```

## Performance & Scaling
- **Batch Size**: Default 32, configurable via environment variables
- **Memory Management**: Explicit TensorFlow tensor cleanup in batch operations
- **Resource Cleanup**: Automatic WebDriver and temp file cleanup via context managers
- **Caching**: File-based caching for HTML and images reduces repeated fetching
- **Network**: HTTP connection pooling with session reuse for improved performance
- **Reliability**: Retry logic with exponential backoff and proper error handling

## Critical Quality Assurance

### Security Features
- **Input Sanitization**: Comprehensive validation for URLs/domains and archive dates
- **Path Traversal Protection**: Safe tar extraction in `utils.safe_extract()`
- **Resource Limits**: Configurable timeouts and batch sizes prevent resource exhaustion
- **Error Isolation**: Robust error handling prevents crashes from malformed inputs

### Performance Monitoring
- **Memory Usage**: TensorFlow tensor cleanup and resource management
- **Network Efficiency**: Connection pooling reduces overhead for batch operations
- **Progress Tracking**: Built-in progress monitoring for long-running operations
- **Cache Optimization**: Intelligent caching reduces redundant network requests

### Testing Strategy
- **Unit Tests**: 14 test modules covering all components
- **Integration Tests**: End-to-end testing with mock and real scenarios
- **Performance Tests**: Memory usage and batch processing validation
- **Security Tests**: Input validation and edge case handling
- **ML Tests**: Marked with `@pytest.mark.ml` for optional model testing