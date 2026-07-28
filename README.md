# piedomains: Classify website content using ML Models or LLMs

[![CI](https://github.com/themains/piedomains/actions/workflows/ci.yml/badge.svg)](https://github.com/themains/piedomains/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/piedomains.svg)](https://pypi.org/project/piedomains)
[![Downloads](https://pepy.tech/badge/piedomains)](https://pepy.tech/project/piedomains)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://themains.github.io/piedomains/)

## What's New in v0.9.0

- **Retrained text model.** A fine-tuned [mmBERT](https://github.com/JHU-CLSP/mmBERT)
  encoder, no longer fed through an English-dictionary filter that was discarding 39.8%
  of every page. Macro-F1 on the evaluation set goes 0.602 → **0.707**.
- **Confidence is a real probability**: temperature-scaled softmax, rather than 39
  per-class isotonic regressions applied elementwise and never renormalized.
- **TensorFlow is gone**, so Python 3.14 works.
- **Bot walls are recovered from archive.org** rather than classified as if the challenge
  page were the site.
- **Failures are named.** Every row carries a stable `error_code`; the run report
  aggregates by reason, stage and source.

**Not multilingual.** Despite a multilingual encoder, the training corpus is English, so
non-English pages classify poorly (0.333 accuracy on a small sample against 0.667 for
English). Fixing this needs multilingual training data, not a configuration change.

**Breaking:** image classification is unavailable while the screenshot model is retrained,
so `classify()` is text-only and `classify_by_images()` raises. This changes no labels —
the old "combined" path returned the text label every time and only averaged the
confidences, so the image model could not affect an answer. See the
[changelog](https://github.com/themains/piedomains/blob/main/CHANGELOG.md).

## Installation

```bash
pip install piedomains
```

Requires Python 3.11+ (3.14 supported).

## Basic Usage

```python
from piedomains import DomainClassifier, DataCollector

classifier = DomainClassifier()
run = classifier.classify(["cnn.com", "amazon.com", "wikipedia.org"])

for result in run["results"]:
    print(f"{result['domain']}: {result['category']} ({result['confidence']:.3f})")

# Output:
# cnn.com: news (0.876)
# amazon.com: shopping (0.923)
# wikipedia.org: education (0.891)
```

## Knowing What Failed

Every call returns both the per-domain rows and a run report, so a long URL list
never fails silently. Each row carries a machine-readable `status`, the `stage` it
reached, and a stable `error_code`:

```python
run = classifier.classify(open("domains.txt").read().split())

print(run["report"])
# {'run_id': '8fe4cc80eeb5', 'total': 500, 'classified': 461, 'failed': 39,
#  'by_reason': {'dns_error': 12, 'timeout': 9, 'cannot_classify': 11, 'thin_content': 7},
#  'by_stage': {'fetch': 32, 'infer': 7},
#  'by_source': {'live': 448, 'archive': 13},
#  'missing': ['foo.com', 'bar.org', ...],
#  'started_at': ..., 'finished_at': ..., 'elapsed_ms': 184203}

# Retry only what is worth retrying
retry = [r["domain"] for r in run["results"] if r.get("retryable")]
```

`error_code` is a closed, stable set — safe to group on: `invalid_domain`,
`dns_error`, `connection_error`, `timeout`, `http_error`, `robots_blocked`,
`content_type_rejected`, `content_too_large`, `no_archive_snapshot`,
`archive_rate_limited`, `empty_text`, `missing_input_path`, `missing_screenshot`,
`model_load_error`, `model_error`, `llm_error`, `bot_blocked`, `thin_content`,
`cannot_classify`, `unknown`.

`cannot_classify` is the umbrella terminal state: branch on it when you do not want
to enumerate every cause.

## Bot Walls

Roughly one domain in seven serves an anti-bot interstitial rather than a page.
Changing the user-agent does not help — DataDome and Cloudflare fingerprint headless
Chromium itself — so `piedomains` detects the interstitial and refetches the page from
archive.org, which already has it. No evasion, and no challenge page classified as
though it were the site.

```python
run = classifier.classify(["etsy.com", "reuters.com", "indeed.com"])
for r in run["results"]:
    print(r["domain"], r["category"], r["source"], r["snapshot_timestamp"])
# etsy.com     shopping  archive  20260727020309
# reuters.com  news      archive  20260718201522
# indeed.com   jobsearch archive  20260724170521
```

A capture older than `archive_max_age_days` (default 365) is refused rather than
passed off as the live page; those domains report `cannot_classify`. Set
`archive_fallback=False` to turn this off and have blocked domains report
`bot_blocked` instead.

From the command line:

```bash
classify_domains --file domains.txt --output json --report run-report.json
# stdout: {"results": [...], "report": {...}}
# stderr: 461/500 classified, 39 failed (run 8fe4cc80eeb5)
#           dns_error: 12
#           timeout: 9
# exit status is non-zero if any domain failed
```

### Structured logs

Human-readable text is the default. For pipelines, opt into JSON lines — every
record carries the `run_id`, plus `domain`/`stage`/`error_code` where relevant, so
logs join against the report:

```bash
PIEDOMAINS_LOG_FORMAT=json classify_domains --file domains.txt
# {"ts":"...","level":"ERROR","run_id":"8fe4cc80eeb5","domain":"foo.com",
#  "stage":"fetch","error_code":"timeout","msg":"navigation timed out"}
```

## Classification Methods

```python
# Text-only. `classify` and `classify_by_text` are the same thing in this
# version -- see the note on image classification above.
run = classifier.classify(["github.com"])
run = classifier.classify_by_text(["news.google.com"])

# Batch processing with separated workflow
collector = DataCollector()
collection = collector.collect_batch(domains, batch_size=50)
results = classifier.classify_from_collection(collection, method="text")
```

## Historical Analysis

```python
# Analyze archived versions from archive.org
old_run = classifier.classify(["facebook.com"], archive_date="20100101")

# Batch processing with archive.org (respects rate limits)
domains = ["google.com", "wikipedia.org", "cnn.com"]
collector = DataCollector(archive_date="20050101")
collection = collector.collect_batch(domains, batch_size=10)  # Archive.org uses conservative defaults
historical_results = classifier.classify_from_collection(collection, method="text")
```

### How archive analysis works

Snapshot discovery and retrieval go through the
[`wayback`](https://github.com/edgi-govdata-archiving/wayback) library (CDX + Memento):

- **Only status-200 captures are used.** An archived redirect or 404 is never classified
  as if it were content; the domain reports `no_archive_snapshot` instead.
- **The capture actually used is reported** as `snapshot_timestamp` — not the date you
  asked for. Requesting `20100101` for `cnn.com` yields `20100101041727`.
- **Text is fetched raw** via Wayback's `id_` playback mode: no injected Wayback
  JavaScript, no rewritten URLs, and no browser required — so the text path is fast.
- **Screenshots render via `if_`**, which hides the Wayback toolbar but keeps archived
  CSS and images, so the page looks as it did. (`id_` would render an unstyled skeleton.)
- Rate limiting, retries and exponential backoff are handled by the `wayback` session.

The cache key includes the archive date, so a live fetch and snapshots from different
years coexist rather than overwriting one another:

```
cache/html/cnn.com.html            # live
cache/html/cnn.com@20050101.html   # 2005 snapshot
cache/html/cnn.com@20150101.html   # 2015 snapshot
```

Configure via `piedomains.config`: `archive_max_parallel`, `archive_window_days`,
`archive_search_rate`, `archive_memento_rate`, `archive_retries`, `archive_backoff`.

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

39 categories: news, finance, shopping, education, government, adult content, gambling, social networks, search engines, and others based on Shallalist taxonomy.

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
run = classifier.classify(['example.com'])
print(run['results'][0]['category'])
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
- [Examples](https://github.com/themains/piedomains/tree/main/examples)
- [Security Guide](https://github.com/themains/piedomains/tree/main/examples/sandbox)

## Development

```bash
git clone https://github.com/themains/piedomains
cd piedomains
uv sync --all-groups
uv run pytest tests/ -v
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
