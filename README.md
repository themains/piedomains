# piedomains

Guess what a website is about from its homepage.

[![CI](https://github.com/themains/piedomains/actions/workflows/ci.yml/badge.svg)](https://github.com/themains/piedomains/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/piedomains.svg)](https://pypi.org/project/piedomains)
[![Downloads](https://pepy.tech/badge/piedomains)](https://pepy.tech/project/piedomains)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://themains.github.io/piedomains/)
[![Text model](https://img.shields.io/badge/%F0%9F%A4%97-text%20model-yellow)](https://huggingface.co/gojiberries/piedomains-text)
[![Image model](https://img.shields.io/badge/%F0%9F%A4%97-image%20model-yellow)](https://huggingface.co/gojiberries/piedomains-image)

Give it a list of domains. It fetches each homepage, reads the text, and returns
one of 44 categories with a calibrated probability.

```python
from piedomains import DomainClassifier

classifier = DomainClassifier()
run = classifier.classify(["cnn.com", "wikipedia.org"])

for r in run["results"]:
    if r["status"] == "ok":
        print(f"{r['domain']:16s} {r['category']:10s} {r['confidence']:.3f}")
    else:
        print(f"{r['domain']:16s} failed: {r['error_code']}")

# cnn.com          news       0.712
# wikipedia.org    library    0.554
```

A domain that could not be fetched comes back with `category` set to `None` and
a reason in `error_code`, so check `status` before reading a label.

## Install

```bash
pip install piedomains
```

Python 3.11 or newer. The wheel is about 245 KB; model weights download from
Hugging Face on first use and cache locally.

Live pages render in headless Chromium, so install the browser once:

```bash
playwright install chromium
```

Archived pages are fetched over plain HTTP and need no browser.

## How accurate it is

Read this before you trust a label. These are the numbers the two shipped
checkpoints report for themselves, on documents held out of their own training:

| model | accuracy | macro-F1 | held-out n | calibration error |
|---|---|---|---|---|
| text (the default) | 0.818 | 0.758 | 4,382 | 0.017 |
| screenshots (opt-in) | 0.501 | 0.370 | 4,451 | 0.014 |

Both benchmarks are held-out splits of the training corpus, so they share its
distribution and its labeling conventions. Expect worse on domains that look
nothing like it. The last check against labels this project did not write, an
agreement test on 155 popular domains cross-referenced with Curlie, scored 0.543
on the previous checkpoint. Read the table as an in-distribution ceiling rather
than what you will see on an arbitrary crawl.

Accuracy is also very uneven across the 44 classes. The text model scores F1
0.99 on `parked` and 0.97 on `science` and `religion`, against 0.15 on
`urlshortener`, 0.33 on `library`, 0.37 on `socialnet`, and 0.45 on `shopping`.
Two of those weak classes are the answers in the example above, which is a fair
warning about how much to read into a single label.

Confidence is worth reading. Temperature scaling brings calibration error to
0.017 for text, which means a batch of predictions at 0.7 is right about 70% of
the time. Plenty of domains come back below 0.5, and those are close to guesses.

The training corpus is overwhelmingly English, and non-English pages score
measurably worse. Usable, not equal.

Every number above comes from the pinned checkpoints and can be read back:

```python
import json
from huggingface_hub import hf_hub_download
from piedomains.text import DEFAULT_TEXT_MODEL, DEFAULT_TEXT_REVISION

path = hf_hub_download(
    DEFAULT_TEXT_MODEL, "test_metrics.json", revision=DEFAULT_TEXT_REVISION
)
print(json.load(open(path))["accuracy"])
```

## What you get back

Every call returns per-domain rows and a run report:

```python
run = classifier.classify(["cnn.com"])
row = run["results"][0]
```

```json
{
  "domain": "cnn.com",
  "category": "news",
  "confidence": 0.712,
  "categories": [
    {"category": "news", "probability": 0.712},
    {"category": "radiotv", "probability": 0.218}
  ],
  "raw_predictions": {"news": 0.712, "radiotv": 0.218, "movies": 0.026, "...": "all 44"},
  "model_used": "text/shallalist_ml",
  "source": "live",
  "snapshot_timestamp": null,
  "status": "ok",
  "stage": "infer",
  "error_code": null,
  "retryable": false
}
```

`category` is the argmax and `confidence` is its probability.

The label set is not mutually exclusive. `shopping` says what a site does,
`automobile` says what it is about, and a car dealership is honestly both, so
`categories` reports every label above a probability floor. About 65% of domains
get exactly one; only ambiguous ones get two. Reporting a second label raises the
chance of covering the right answer from 79.7% to 86.6%, measured on 4,673
held-out documents. Whether that second label is itself correct is not something
the evaluation data can answer, because its gold labels are single-label. Treat
it as a candidate, not a finding.

## Knowing what failed

A long domain list never fails silently. Each row carries a `status`, the `stage`
it reached, and a stable `error_code`, and the report aggregates them:

```python
run = classifier.classify(open("domains.txt").read().split())

print(run["report"])
# {'run_id': '8fe4cc80eeb5', 'total': 500, 'classified': 461, 'failed': 39,
#  'by_reason': {'dns_error': 12, 'timeout': 9, 'cannot_classify': 11, 'thin_content': 7},
#  'by_stage': {'fetch': 32, 'infer': 7},
#  'by_source': {'live': 448, 'archive': 13},
#  'missing': ['foo.com', 'bar.org'],
#  'elapsed_ms': 184203}

retry = [r["domain"] for r in run["results"] if r.get("retryable")]
```

`error_code` is a closed set of 21 values, safe to group on: `invalid_domain`,
`dns_error`, `private_address`, `connection_error`, `timeout`, `http_error`,
`robots_blocked`, `content_type_rejected`, `content_too_large`,
`no_archive_snapshot`, `archive_rate_limited`, `empty_text`,
`missing_input_path`, `missing_screenshot`, `model_load_error`, `model_error`,
`llm_error`, `bot_blocked`, `thin_content`, `cannot_classify`, `unknown`. Branch
on `cannot_classify` when you do not want to enumerate every cause.

## Command line

```bash
classify_domains --file domains.txt --output json --report run-report.json
# stdout: {"results": [...], "report": {...}}
# stderr: 461/500 classified, 39 failed (run 8fe4cc80eeb5)
#           dns_error: 12
#           timeout: 9
```

Exit status is non-zero if any domain failed. `--method` takes `text` (the
default), `images`, or `combined`. `--archive-date YYYYMMDD` classifies an
archived snapshot instead of the live page.

For pipelines, opt into JSON logs. Every record carries the `run_id`, plus
`domain`, `stage`, and `error_code` where relevant, so logs join against the
report:

```bash
PIEDOMAINS_LOG_FORMAT=json classify_domains --file domains.txt
# {"ts":"...","level":"ERROR","run_id":"8fe4cc80eeb5","domain":"foo.com",
#  "stage":"fetch","error_code":"timeout","msg":"navigation timed out"}
```

## Bot walls

Roughly one domain in seven serves an anti-bot interstitial rather than a page.
Changing the user-agent does not help, because DataDome and Cloudflare
fingerprint headless Chromium itself. So piedomains detects the interstitial and
refetches the page from archive.org, which already has it. No evasion, and no
challenge page classified as though it were the site.

```python
run = classifier.classify(["etsy.com", "reuters.com", "indeed.com"])
for r in run["results"]:
    print(r["domain"], r["category"], r["source"], r["snapshot_timestamp"])
# etsy.com     shopping  archive  20260727020309
# reuters.com  news      archive  20260718201522
# indeed.com   jobsearch archive  20260724170521
```

A capture older than `archive_max_age_days` (default 365) is refused rather than
passed off as the live page, and those domains report `cannot_classify`. Set
`archive_fallback=False` to turn this off and have blocked domains report
`bot_blocked`.

## Crawling politely

The fetcher reads robots.txt through `protego`, Scrapy's parser, and obeys it
before making any other request. It throttles per host and bounds concurrency.
Its user-agent names the package and carries a contact URL.

Robots failures are directional. A 5xx or an unparseable robots body fails
closed, while an unreachable host fails open, so you get the real `dns_error`
rather than a claim that the host refused you.

## Historical analysis

```python
old_run = classifier.classify(["facebook.com"], archive_date="20100101")

from piedomains import DataCollector

collector = DataCollector(archive_date="20050101")
collection = collector.collect_batch(["google.com", "cnn.com"], batch_size=10)
results = classifier.classify_from_collection(collection, method="text")
```

Snapshot discovery and retrieval go through the
[`wayback`](https://github.com/edgi-govdata-archiving/wayback) library. Only
status-200 captures are used, so an archived redirect or 404 is never classified
as if it were content; the domain reports `no_archive_snapshot` instead. The
capture actually used comes back as `snapshot_timestamp`, which is not
necessarily the date you asked for: requesting `20100101` for `cnn.com` yields
`20100101041727`.

Text is fetched raw through Wayback's `id_` playback mode, so there is no
injected Wayback JavaScript, no rewritten URLs, and no browser. Screenshots
render through `if_`, which hides the Wayback toolbar but keeps archived CSS and
images so the page looks as it did.

The cache key includes the archive date, so a live fetch and snapshots from
different years coexist rather than overwriting one another:

```
cache/html/cnn.com.html            # live
cache/html/cnn.com@20050101.html   # 2005 snapshot
cache/html/cnn.com@20150101.html   # 2015 snapshot
```

Rate limits, retries, and backoff are configurable through `piedomains.config`:
`archive_max_parallel`, `archive_window_days`, `archive_search_rate`,
`archive_memento_rate`, `archive_retries`, `archive_backoff`.

## Other ways to classify

Text is the default and the most accurate option available here.

```python
run = classifier.classify_by_text(["news.google.com"])
```

Screenshots are opt-in and weak. The image model scores 0.501 accuracy against
the text model's 0.818 on comparable held-out splits, and lower still on pages
captured from the web as it looks today. Four ways of
combining the two were measured and all four came out worse than text alone, so
the ensemble was built and not shipped. Reach for screenshots when there is no
text to read, which is the case they exist for.

```python
run = classifier.classify(["github.com"], use_screenshots=True)
run = classifier.classify_by_images(["github.com"])
```

An LLM can classify into your own label set instead of the built-in 44:

```python
classifier.configure_llm(
    provider="openai",
    model="gpt-4o",
    categories=["news", "shopping", "social", "tech"],
)
run = classifier.classify_by_llm(["example.com"])
run = classifier.classify_by_llm(
    ["site.com"], custom_instructions="Classify by educational value"
)
```

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY` in the
environment.

To fetch once and classify several ways, separate collection from inference:

```python
collector = DataCollector()
collection = collector.collect_batch(domains, batch_size=50)
results = classifier.classify_from_collection(collection, method="text")
```

## Categories

44 labels: `adult`, `alcohol`, `automobile`, `cooking`, `dating`, `downloads`,
`drugs`, `education`, `finance`, `fortunetelling`, `forum`, `gamble`, `games`,
`gardening`, `government`, `homestyle`, `hospitals`, `humor`, `imagehosting`,
`isp`, `jobsearch`, `library`, `military`, `movies`, `music`, `news`, `parked`,
`pets`, `politics`, `radiotv`, `realestate`, `religion`, `restaurants`,
`science`, `searchengines`, `shopping`, `socialnet`, `sports`, `travel`,
`unavailable`, `urlshortener`, `weapons`, `webmail`, `wellness`.

`parked` and `unavailable` describe domains with no site behind them.

The set derives from Shallalist, with one rule deciding every case: is the
category visible in the page text? Classes describing how a site is built and
paid for rather than what it says (`adv`, `tracker`, `spyware`, `redirector`)
are gone, because a homepage selling handmade goods reads identically whether or
not its operator runs trackers. Those four sat behind a third of all evaluation
errors. See
[`piedomains.training.taxonomy`](https://github.com/themains/piedomains/blob/main/src/piedomains/training/taxonomy.py)
for the reasoning on each one.

## Running in a container

```bash
docker build -t piedomains-sandbox .

docker run --rm --memory=2g --cpus=2 --read-only \
  --tmpfs /tmp --tmpfs /var/tmp \
  piedomains-sandbox python -c "
from piedomains import DomainClassifier
run = DomainClassifier().classify(['example.com'])
print(run['results'][0]['category'])
"
```

`examples/sandbox/secure_classify.py` runs a batch under the same constraints.

## Retraining and evaluation

The scripts that built and scored the checkpoints ship with the package:

```bash
classify_domains --training-scripts   # prints where they are installed
```

## Links

[API reference](https://themains.github.io/piedomains/) |
[Examples](https://github.com/themains/piedomains/tree/main/examples) |
[Changelog](https://github.com/themains/piedomains/blob/main/CHANGELOG.md) |
[Sandbox guide](https://github.com/themains/piedomains/tree/main/examples/sandbox)

## Development

```bash
git clone https://github.com/themains/piedomains
cd piedomains
uv sync --all-groups
uv run pytest tests/ -v
```

## License

MIT

## Citation

```bibtex
@software{piedomains,
  title={piedomains: classify website content from homepage text},
  author={Chintalapati, Rajashekar and Sood, Gaurav},
  year={2026},
  url={https://github.com/themains/piedomains}
}
```
