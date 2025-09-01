# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-09-01

### Added
- **Archive.org Historical Classification**: New functionality for analyzing historical website content
  - `pred_shalla_cat_archive()`: Combined text+image classification using archive.org snapshots
  - `pred_shalla_cat_with_text_archive()`: Text-only classification from historical content
  - `pred_shalla_cat_with_images_archive()`: Image-only classification from historical screenshots
  - Support for point-in-time analysis using 'YYYYMMDD' date format
  - Automatic discovery of closest available snapshots to target dates
  - Modular fetcher architecture with `BaseFetcher`, `LiveFetcher`, and `ArchiveFetcher` classes
- **Enhanced URL Support**: Improved handling of full URLs vs domain names
  - Better URL parsing and domain extraction
  - Support for URLs with paths, ports, and protocols
  - Archive URL construction and validation
- **Testing Infrastructure**: Comprehensive test suite for archive functionality
  - Tests for 10 major domains across different time periods (2005-2020)
  - Archive content fetching and processing validation
  - Integration tests for historical content classification

### Changed
- Extended main API in `domain.py` to export new archive functions
- Updated `__init__.py` to include archive functions in public API
- Enhanced documentation with archive.org usage examples

### Technical Details
- Archive snapshots fetched via direct web.archive.org URLs
- Automatic HTML content cleaning to remove archive.org wrapper elements
- Selenium WebDriver support for archived page screenshots
- Compatible with existing caching and batch processing features

## [0.1.0] - 2024-08-30

### Added
- **Configuration Management**: New `config.py` module with environment variable support
  - Configurable timeouts, retry settings, batch sizes, and WebDriver options
  - Environment variables for customization (e.g., `PIEDOMAINS_HTTP_TIMEOUT`)
- **Context Managers**: New `context_managers.py` module for resource management
  - WebDriver context manager for automatic cleanup
  - Temporary directory and file management
  - Error recovery context with logging
  - ResourceManager class for comprehensive cleanup
- **Domain Validation**: Robust domain name validation with regex patterns
  - Protocol handling (http/https)
  - Domain normalization
  - Invalid domain filtering and reporting
- **Batch Processing**: Memory-efficient processing for large domain lists
  - Configurable batch sizes
  - Memory management with tensor cleanup
  - Progress tracking and logging
- **Retry Logic**: Exponential backoff for network requests
  - Configurable max retries and delay
  - HTTP and WebDriver error recovery
- **Enhanced Logging**: Structured logging throughout the application
  - INFO, DEBUG, WARNING, and ERROR levels
  - Operation progress tracking
  - Error details and context
- **Comprehensive Test Suite**: 6 new test modules added
  - Domain validation tests
  - Text processing tests  
  - Error handling tests
  - Utility function tests
  - Configuration system tests
  - Context manager tests

### Changed
- **Version Bump**: 0.0.19 → 0.1.0 (minor version due to significant improvements)
- **Development Status**: Alpha → Beta (improved stability and features)
- **Type Hints**: Standardized all `string` type hints to `str`
- **Error Handling**: Enhanced with specific exception types and better recovery
- **Documentation**: Comprehensive docstrings following Python conventions
- **Dependencies**: Added webdriver-manager for automatic ChromeDriver management
- **Console Script**: Fixed entry point path (`piedomain:main` → `piedomains.domain:main`)

### Fixed
- **Security**: Fixed unsafe tar extraction vulnerability in `utils.py`
- **Logic Error**: Resolved undefined `args.type` reference in main function
- **Hard-coded Paths**: Removed hard-coded ChromeDriver path dependency
- **Memory Leaks**: Added proper tensor and resource cleanup
- **Resource Management**: WebDriver instances now properly closed in all cases

### Security
- **Path Traversal Protection**: Fixed tar extraction to prevent malicious archives
- **Input Validation**: Added domain name validation to prevent injection attacks

### Performance
- **Memory Optimization**: Batch processing with memory management
- **Resource Cleanup**: Automatic cleanup of temporary files and WebDriver instances
- **Tensor Management**: Proper cleanup of TensorFlow tensors to prevent memory leaks

## [0.0.19] - Previous Release
- Legacy version with basic functionality