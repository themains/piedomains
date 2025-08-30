# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
- Run all tests: `cd piedomains/tests && pytest`
- Run tests with tox: `tox` (tests with Python 3.10, 3.11)
- Run specific test: `pytest piedomains/tests/test_001_pred_domain_text.py`

### Linting and Code Quality
- Run pylint: `pylint piedomains/` (uses configuration from `pylintrc`)
- Run flake8 (CI configuration): 
  - Syntax errors: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - General linting: `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics`

### Installation and Development
- Install package: `pip install -e .` (from repository root)
- Install with requirements: `pip install -r requirements.txt`
- Console script: `classify_domains` (entry point defined in setup.py)

### Documentation
- Build docs: `cd docs && make html`
- Documentation is built with Sphinx and deployed to ReadTheDocs

## Architecture

### Core Components

**Piedomain Class (`piedomain.py`)**: Central prediction engine that implements three prediction methods:
- `pred_shalla_cat_with_text()`: Text-based domain classification using HTML content
- `pred_shalla_cat_with_images()`: Image-based classification using homepage screenshots  
- `pred_shalla_cat()`: Combined approach using both text and images

**Base Class (`base.py`)**: Handles model downloading and loading from Harvard Dataverse. Models are cached locally after first download.

**Domain Module (`domain.py`)**: Main API entry point that exposes the three prediction functions from Piedomain class.

### Machine Learning Pipeline

1. **Text Processing**: HTML content is scraped, cleaned (removing non-English words, stopwords, punctuation), and fed to a TensorFlow text model
2. **Image Processing**: Screenshots are taken via Selenium WebDriver, resized to 254x254, and processed by a TensorFlow CNN model  
3. **Model Calibration**: Text predictions are post-processed using isotonic regression calibrators (stored in `model/calibrate/text/`)
4. **Ensemble**: Final predictions combine text and image probabilities with equal weighting

### Data Flow
- Input: List of domain names
- HTML extraction: Requests + BeautifulSoup for text content
- Screenshot capture: Selenium Chrome WebDriver in headless mode
- Feature processing: NLTK for text cleanup, PIL for image preprocessing
- Prediction: TensorFlow models for both modalities
- Output: Pandas DataFrame with predictions, probabilities, and metadata

### Categories
The model predicts among 41 Shallalist categories defined in `constants.py` including: adv, alcohol, automobile, dating, downloads, drugs, education, finance, forum, gamble, government, news, politics, porn, recreation, shopping, socialnet, etc.

### Model Storage
- Models are downloaded from Harvard Dataverse on first use
- Cached in `model/shallalist/` directory structure
- Text model: `saved_model/piedomains/`  
- Image model: `saved_model/pydomains_images/`
- Calibrators: `calibrate/text/*.sav` files

### Key Dependencies
- TensorFlow 2.11+ for neural network inference
- Selenium 4.8 for web scraping and screenshots (requires ChromeDriver)
- NLTK for text processing and English word filtering
- scikit-learn for model calibration
- BeautifulSoup4 for HTML parsing
- Pillow for image processing