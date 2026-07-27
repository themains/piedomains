# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-07-26

> **Versioning note.** Under the py-canon standard the git tag *is* the version
> (`uv-dynamic-versioning`); there are no version strings in source. Tags in this repo
> stopped at `v0.3.2`, while 0.4.0–0.5.0 were published to PyPI by manual
> `workflow_dispatch` off the old static `project.version`, leaving no tags and no GitHub
> Releases behind. That history cannot be recovered, and tagging guessed commits would
> fabricate it, so it is left alone.
>
> **The 0.6.0 entry below was never published.** PyPI went 0.5.0 → 0.7.0; everything
> documented under 0.6.0 ships inside this release. It is kept as a separate entry
> because it is a distinct, breaking API change and folding it in would misrepresent
> when the work happened.

### Added

- **Bot walls are detected and recovered, not silently classified.** DataDome, Cloudflare,
  Akamai, Imperva and PerimeterX interstitials were previously classified as if they were
  the site — a ~1470-byte CAPTCHA stub whose only visible text is the domain name. They are
  now identified (`piedomains.blocking`) and the page is refetched from archive.org, which
  already holds it. Recovered rows carry `source: "archive"` and the realized
  `snapshot_timestamp`; the run report gains `by_source`.

  Detection is tiered on purpose. `reddit`, `walmart`, `tinyurl`, `quora` and
  `bankofamerica` all serve real pages while embedding reCAPTCHA, PerimeterX or a
  Turnstile widget, so an ambiguous marker only counts when the page also *looks* like an
  interstitial. Treating those as blocks would have discarded good classifications.

- **A capture older than `archive_max_age_days` (default 365) is refused** rather than
  passed off as the current page. A domain whose only captures are years old reports
  `cannot_classify` instead of being labelled from a page that no longer exists.

- **Refusal instead of a confident guess on empty pages.** Below `min_tokens` (default 30)
  the model returns its prior — `recreation` 0.31, `shopping` 0.21, `porn` 0.19 on empty
  input — which is where results like `facebook.com → porn` came from. Such rows now report
  `thin_content` and no category.

### Added
- **Run reports.** `classify()`, `classify_by_text()` and `classify_by_images()` now return
  `{"results": [...], "report": {...}}`. The report gives `total`/`classified`/`failed`,
  `by_reason`, `by_stage`, `elapsed_ms`, and `missing` — the explicit list of domains that
  produced no classification.
- **Outcome taxonomy** (`piedomains.outcomes`): every result row carries `status`,
  `stage` (`validate`/`fetch`/`process`/`infer`), a stable `error_code` and `retryable`,
  so failures across a large URL list can be grouped without string-matching.
- **Structured logging**: `PIEDOMAINS_LOG_FORMAT=json` emits JSON lines; `bind_context()`
  threads a `run_id` (plus `domain`/`stage`/`error_code`) through every record so logs join
  against the report. Human-readable text remains the default.
- `classify_domains` CLI gained `--report PATH`, prints a failure summary to stderr, and
  exits non-zero when any domain failed.

- **Archive snapshots now report what was actually fetched**: results and collection
  metadata carry `snapshot_timestamp` (the realized capture), not just the requested date.

### Changed
- **BREAKING**: the top-level `classify*` functions return a dict envelope rather than a
  bare list. Use `run["results"]` for the rows.
- **archive.org now goes through the [`wayback`](https://github.com/edgi-govdata-archiving/wayback)
  library** (CDX + Memento) instead of ~850 lines of hand-rolled availability-API calls,
  sleeps and toolbar stripping. Text is fetched raw via `id_` playback — no browser
  needed — and screenshots render via `if_`, which hides the Wayback toolbar while keeping
  archived CSS and images.
- **Only status-200 captures are used.** Previously an archived 301 or 404 was fetched and
  classified as though it were real content.
- The cache key now includes the archive date, so a live fetch and snapshots from
  different years no longer overwrite each other.
- Archive config replaced: `archive_cdx_rate_limit`, `archive_page_delay`,
  `archive_retry_on_429` and `archive_429_wait_time` gave way to `archive_window_days`,
  `archive_search_rate`, `archive_memento_rate`, `archive_retries`, `archive_backoff`,
  `archive_render_settle_ms` and `archive_screenshot_timeout`.

- `piedomains.__version__` is now derived from installed distribution metadata via
  `importlib.metadata`, per the fleet standard — no version string in source.
- **Publishing keeps the legacy `python-publish.yml` workflow on purpose.** This project's
  PyPI trusted publisher predates py-canon adoption and is keyed to that filename with
  environment `pypi`; OIDC claims reference the workflow *file*, so moving publishing into
  `release.yml` would break trusted publishing until the pypi.org config changes. The
  publish job is therefore stripped from `release.yml`. It also triggers on the tag rather
  than `release: published`, because releases created by the reusable workflow use
  `GITHUB_TOKEN`, and GitHub does not fire workflow triggers for `GITHUB_TOKEN` events.

### Removed
- `piedomains.archive_org_downloader` — dead in production (nothing in `src/` imported it)
  and a partial duplicate of `ArchiveFetcher`.

### Fixed
- **`networkidle` was losing whole sites.** Page loads waited for network quiet, which never
  arrives on a chatty page: `theverge`, `stackoverflow` and `weather.com` timed out entirely
  (3 of 10 popular sites tested) and `outlook.com` yielded **1** usable token against 414.
  Loads now wait for the DOM, settle briefly, then race a *capped* network-quiet window, so
  `nytimes.com` keeps the extra text it genuinely gains without the 20-second cliff.
- **Failed fetches were cached and silently reused.** `spotify.com` sat in the cache with 8
  usable tokens against 292 on a live refetch, so evaluation partly measured stale failures.
  A page that renders under `min_tokens` words now fails the fetch, and nothing is written.
- **A navigation timeout reached callers as `unknown`**, hiding the most common fetch failure
  and preventing the archive fallback from being tried at all.
- **Batch collection dropped `error_code` and `snapshot_timestamp`.** Only the single-domain
  path carried them, so every real run (anything over ten domains) reported a detected bot
  wall as `unknown` and gave no way to tell which capture an archive batch used.
- `archive.org` being rate-limited no longer hardens into a terminal `cannot_classify`.
  Throttling says nothing about the domain, so that verdict stays retryable.
- The archive toolbar stripper matched **nothing**: `find_all(["script","link","div"],
  attrs={"src":…, "href":…})` requires *both* attributes to match, so a `<script src=…>`
  never matched. Moot now that `id_` returns the raw capture.
- A failed screenshot no longer reports an `image_path` pointing at a file that does not
  exist, which made downstream image classification fail on a missing file.
- Archived screenshots no longer stall on "waiting for fonts to load" — fonts, media,
  websockets and manifests are blocked during the archive render.
- Adopted the py-canon packaging standard: `src/` layout, ruff-only linting,
  pyright type checking, PEP 735 dependency groups, and reusable CI/docs/release
  workflows.
- Version is now derived from the git tag via `uv-dynamic-versioning` rather than
  a static `project.version`.

### Fixed
- `classify_domains` console script pointed at a nonexistent module
  (`piedomains.domain:main`) and could never run. Implemented the CLI.
- `pytest` no longer forces coverage reports on every local run.

## [0.6.0] - 2025-12-17

### 💥 BREAKING CHANGES
- **API Modernization**: Complete removal of DataFrame outputs in favor of pure JSON responses
- **Deprecated Method Removal**: Removed `collect_data()` → Use `collect_content()`
- **Deprecated Parameter Removal**: Removed `latest_models` → Use `latest`
- **Deprecated Alias Removal**: Removed `classify_from_data()` → Use `classify_from_collection()`
- **No Backward Compatibility**: Clean break from v0.5.x for cleaner, maintainable codebase

### 🎯 API Improvements
- **Consistent Parameter Naming**: Unified `latest` parameter across all classification methods
- **JSON-Only Responses**: All methods now return `List[Dict]` with consistent schema
- **Separated Workflow**: Clear distinction between data collection and inference phases
- **Method Naming**: More intuitive method names following verb-noun patterns

### 📋 Comprehensive Documentation
- **JSON Schema Documentation**: Complete schema definitions for all API responses
- **Field Documentation**: Detailed field descriptions with data types and examples
- **Supported Categories**: Full list of 41 Shallalist categories with examples
- **Updated Examples**: All examples updated to demonstrate new JSON-only API

### 🧪 Testing & Quality
- **Updated Test Suite**: All tests migrated to new API methods and JSON expectations
- **Linting Compliance**: Full `ruff` compliance with automatic formatting
- **Example Updates**: All demonstration scripts updated for new API
- **Documentation Sync**: README, examples, and docstrings fully synchronized

### 🏗️ Code Quality
- **Removed Dead Code**: Eliminated all deprecated compatibility shims and warnings
- **Cleaner Imports**: Removed unused imports and circular dependency risks
- **Consistent Error Messages**: Standardized error messages and exception handling
- **Type Consistency**: Better type hints and consistent return types

### 🚀 Migration Guide
For users upgrading from v0.5.x:

```python
# OLD (v0.5.x) - No longer supported
result = classifier.classify(domains)
df = pd.DataFrame(result)  # DataFrame access
data = classifier.collect_data(domains)  # Deprecated method
classifier.classify_from_data(data, latest_models=True)  # Deprecated parameter

# NEW (v0.6.0) - Required changes
results = classifier.classify(domains)  # Returns List[Dict] directly
collection = classifier.collect_content(domains)  # New method name
classifier.classify_from_collection(collection, latest=True)  # New parameter name
```

## [0.5.0] - 2025-12-17

### 🚀 Major Features
- **Playwright Migration**: Complete migration from Selenium to Playwright for modern web content extraction
- **Unified Content Pipeline**: Text extraction and screenshots now use the same Playwright pipeline for better consistency
- **Docker Security Integration**: Full Docker containerization with security sandbox for safe domain analysis
- **Performance Improvements**: 12.8x performance boost through parallelization (13.2s → 1.0s per domain)

### ⚡ Performance & Architecture
- **Modern Web Content Handling**: Playwright-based fetching with resource blocking for videos and heavy content
- **Parallel Processing**: Unified content extraction with async/sync compatibility patterns
- **Resource Blocking**: Automatic blocking of video/media content for faster processing
- **Browser Context Management**: Efficient browser reuse with proper cleanup

### 🛡️ Security & Sandbox
- **Secure Classification Scripts**: New `secure_classify.py` with Docker isolation and read-only filesystem
- **Container Sandbox**: Pre-built Docker images with security constraints (2GB RAM, 2 CPU, read-only)
- **Non-root Execution**: All container operations run as non-root playwright user (uid=995)
- **Resource Isolation**: Tmpfs mounts for temporary data with proper permission management

### 🐳 Docker & DevOps
- **Production-Ready Containers**: Optimized Dockerfile with pre-installed Playwright browsers
- **Rancher Desktop Support**: Full compatibility with Rancher Desktop for local development
- **Entrypoint Automation**: Smart browser installation detection and runtime optimization
- **Multi-stage Builds**: Efficient Docker builds with proper layer caching

### 🔧 API & Developer Experience
- **Backwards Compatibility**: Maintained full API compatibility despite internal Playwright migration
- **Enhanced Error Handling**: Improved error messages and debugging information
- **Comprehensive Logging**: Detailed logging throughout content extraction pipeline
- **Security Validation**: Input sanitization and path traversal protection

### 📦 Project Structure
- **Reorganized Examples**: Moved Docker files and Streamlit demo to examples/ directory
- **Cleaned Dependencies**: Updated pyproject.toml with Playwright dependencies
- **Documentation**: Updated README and examples for new Playwright-based workflow

### 🔄 Breaking Changes
- **Selenium Removal**: Complete removal of Selenium dependencies (clean break, no backward compatibility)
- **Deprecated Methods**: Legacy `get_driver()`, `save_image()`, and `extract_images()` methods marked as deprecated

### 🐛 Bug Fixes
- **URL Normalization**: Fixed URL handling for domains without http/https protocol
- **JavaScript Errors**: Resolved regex syntax errors in browser-based text extraction
- **Container Permissions**: Fixed tmpfs mount permissions for secure sandbox execution
- **Browser Detection**: Improved browser installation detection in Docker environments

### 📊 Benchmarks
- **Standard Container**: 9.41 seconds total processing time
- **Sandbox Container**: 7.47 seconds (20.6% faster due to optimized configuration)
- **Batch Processing**: 5 seconds average per domain in batch mode
- **Container Startup**: Minimal overhead (~1-2 seconds)

## [0.4.2] - 2025-12-15

### Fixed
- **Dependency Management**: Removed `_has_llm` anti-pattern and implemented proper Python dependency management via pyproject.toml
- **BeautifulSoup Warning**: Fixed deprecation warning by replacing `text=True` with `string=True` in text processor
- **Pytest Warnings**: Added missing `performance` marker to pytest configuration to eliminate unknown mark warnings
- **LLM Classifier**: Fixed duplicate `max_tokens` parameter error in connection test

### Changed
- **Documentation Links**: Updated all references from ReadTheDocs to GitHub Pages (https://themains.github.io/piedomains/)
- **PyPI Links**: Updated PyPI badge to use current domain (pypi.org instead of pypi.python.org)
- **README**: Streamlined documentation by removing editorial content and marketing language, focusing on minimal practical instructions

### Improved
- **Code Quality**: All tests now run without warnings (eliminated 3 targeted warnings)
- **Package Building**: Resolved build conflicts and ensured clean package compilation
- **Link Verification**: All documentation and package links verified as working

## [0.4.0] - 2025-12-15

### 🚨 Breaking Changes
- **REMOVED**: Complete removal of legacy API functions (`pred_shalla_cat()`, `pred_shalla_cat_with_text()`, `pred_shalla_cat_with_images()`, `pred_shalla_cat_archive()`)
- **REMOVED**: Dropped Python 3.10 support - now requires Python 3.11+
- **MOVED**: Tests and notebooks relocated from `piedomains/` subdirectories to project root

### ✨ Added
- **Modern Python 3.11+ Features**: Full adoption of PEP 604 union syntax (`str | None` instead of `Union[str, None]`)
- **Enhanced Logging**: Replaced all `print()` statements with proper `logger` calls for better debugging
- **Improved Code Quality**: Comprehensive ruff linting with auto-fixes applied

### 🔧 Changed
- **Type Hints**: Modernized all type annotations to use Python 3.11+ union syntax (`|`)
- **Import Structure**: Added `from __future__ import annotations` for cleaner type hints
- **Project Structure**:
  - Moved `piedomains/tests/` → `tests/`
  - Moved `piedomains/notebooks/` → `notebooks/`
- **Configuration**: Enhanced error handling with proper logging in config validation

### 📚 Documentation
- **Updated README**: Removed legacy API examples and migration guides
- **Cleaned CLAUDE.md**: Updated test paths and removed backward compatibility references
- **Modernized Examples**: Updated all example scripts to use new API only

### 🧹 Removed
- **LEGACY_API.rst**: Completely removed legacy documentation
- **Archive Support Legacy Functions**: Removed old archive functionality implementations
- **Backward Compatibility**: No more deprecation warnings or legacy function wrappers

### 🔧 Development
- **Ruff Integration**: Full code formatting and linting with modern standards
- **Test Suite**: All 99 tests passing with updated mocking for new logging approach
- **Python 3.11+ Only**: Updated all tooling configs (black, ruff, mypy) for Python 3.11+

This release represents a major cleanup and modernization of the codebase, removing all legacy functionality and fully embracing Python 3.11+ features. Users must migrate to the modern `DomainClassifier` API.

## [0.3.4] - 2025-09-02

### Removed
- Eliminated `safe_import_pandas` helper and related dependency checks.

### Changed
- Reverted pandas and NumPy requirements to the 1.x series to clarify supported environments.

## [0.3.3] - 2025-09-01

### Added
- **Continuous Deployment**: Introduced GitHub Actions workflow for automated PyPI publishing.

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

### API Modernization
- **Modern Interface**: New class-based design for better usability
- **Import Compatibility**: Clean, modern import structure
  ```python
  # Modern API
  from piedomains import DomainClassifier

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
