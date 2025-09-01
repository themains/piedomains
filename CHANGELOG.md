# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2025-09-01

### Fixed
- **Critical Dependency Issue**: Fixed numpy/pandas binary incompatibility error on installation
  - Updated pandas from `==1.4.2` to `>=1.5.0,<3.0.0` for better compatibility
  - Relaxed dependency constraints to use compatible ranges instead of exact pins
  - Prevents `ValueError: numpy.dtype size changed` error on fresh installations

### Enhanced
- **HTTP Performance**: Added connection pooling with `PooledHTTPClient` for batch operations
- **Critical Integration Tests**: Added comprehensive test suite for security and edge cases
- **Documentation**: Updated architecture documentation in CLAUDE.md

### Dependencies Updated
- pandas: `==1.4.2` → `>=1.5.0,<3.0.0`
- scikit-learn: `==1.5.0` → `>=1.3.0,<2.0.0`
- Other dependencies: Changed from exact pins to compatible ranges for better ecosystem compatibility

## [0.3.1] - 2025-09-01

### Documentation
- **README Overhaul**: Complete rewrite with modern, quickstart-focused approach
  - 3-line quickstart example for immediate use
  - Clear migration guide from old API to new DomainClassifier
  - Prominent examples for all classification methods (text, images, combined)
  - Archive.org historical analysis prominently featured
  - Batch processing examples and performance guidelines
  - Moved legacy API documentation to LEGACY_API.rst for reference
- **User Experience**: Much clearer onboarding and usage examples

## [0.3.0] - 2025-09-01

### 🚀 Major API Overhaul - Modern, Intuitive Interface

### Added
- **New Modern API**: Complete redesign for better user experience
  - `DomainClassifier` class with intuitive methods: `.classify()`, `.classify_by_text()`, `.classify_by_images()`
  - `classify_domains()` convenience function for quick usage
  - Integrated archive.org support (no separate functions needed)
  - Batch processing with progress tracking via `.classify_batch()`
  - Consistent parameter naming: `domains` instead of `input`
  - Better error handling and logging throughout
- **Modular Architecture**: Complete code reorganization
  - `piedomains/classifiers/`: Focused classification modules (TextClassifier, ImageClassifier, CombinedClassifier)
  - `piedomains/processors/`: Content processing utilities (TextProcessor, ContentProcessor)
  - Eliminated monolithic 974-line piedomain.py into maintainable modules
  - Clear separation of concerns and better testability
- **Enhanced Testing Suite**: 85+ comprehensive tests
  - `test_011_new_api_integration.py`: New API functionality testing
  - `test_012_archive_functionality.py`: Archive.org integration testing
  - `test_013_performance_benchmarks.py`: Performance and scalability testing
  - Mock-based testing for reliable CI/CD
  - Performance benchmarking and memory usage monitoring
- **Improved Documentation**: 
  - New quickstart-focused README with 3-line setup
  - Comprehensive API examples and migration guide
  - `examples/new_api_demo.py`: Interactive demonstration script

### Changed
- **API Interface**: Modern, class-based design replacing function-based approach
  - DateTime support for archive dates (accepts both strings and datetime objects)
  - Progress tracking for batch operations
  - Automatic cache directory management
  - Integrated fetcher architecture (LiveFetcher/ArchiveFetcher)
- **Code Quality**: Significantly improved maintainability
  - Type hints throughout new codebase
  - Comprehensive error handling
  - Resource management and cleanup
  - Memory-efficient batch processing

### Backward Compatibility
- **Legacy API Preserved**: All existing functions still work
  - `pred_shalla_cat()`, `pred_shalla_cat_with_text()`, etc. unchanged
  - No breaking changes for existing users
  - Deprecation warnings will be added in future versions
- **Import Compatibility**: Both old and new APIs available
  ```python
  # Old API still works
  from piedomains import pred_shalla_cat
  
  # New API available
  from piedomains import DomainClassifier
  ```

### Performance
- **Batch Processing**: Efficient handling of large domain lists
- **Caching**: Improved cache management and directory structure
- **Memory Management**: Better resource cleanup and optimization
- **Scalability**: Tested with 1000+ domain batches

### Developer Experience
- **Better Error Messages**: More descriptive error handling
- **Type Safety**: Full type hints for better IDE support
- **Logging**: Structured logging throughout application
- **Testing**: Comprehensive test coverage for all new functionality

## [0.2.1] - 2025-09-01

### Changed
- **Repository Organization**: Improved code structure and organization
  - Moved test/demo scripts from root directory to `examples/` folder
  - Cleaned up build artifacts (`build/`, `dist/`, `*.egg-info/`)
  - Added `examples/README.md` with usage instructions
  - Updated `.gitignore` to prevent future clutter with better patterns

### Documentation
- Enhanced documentation structure for better maintainability
- ReadTheDocs configuration optimized for reliable builds

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