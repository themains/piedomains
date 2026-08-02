# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Commands

This repo follows the **py-canon** fleet standard and uses `uv` for everything.

```bash
uv sync --all-groups          # set up
uv run pytest tests/ -v       # all tests
uv run pytest tests/ -m "not ml and not archive"   # offline only
uv run pytest tests/ -m archive                    # hits live archive.org

uv run ruff check .           # lint  (the only linter; no black/isort/flake8)
uv run ruff format .          # format
uv run pyright                # types (the only type checker; no mypy)
uv run pydoclint --config pyproject.toml src/      # docstring/signature match
uvx zizmor --min-severity high .github/workflows/  # workflow security
uvx preen check --strict      # fleet conformance
```

`pydoclint` and the docs `doctest` builder both run in CI and are easy to forget
locally — run them before claiming a change is green.

Docs:

```bash
uv run sphinx-build -W -b html docs _site    # warnings are errors in CI
uv run sphinx-build -b doctest docs _doctest
```

## Architecture

Flat modules under `src/piedomains/`. There is no `classifiers/` or
`processors/` package.

| Module | Role |
|---|---|
| `api.py` | `DomainClassifier` facade; `_run` collects, classifies, annotates, reports |
| `data_collector.py` | `DataCollector` — fetch + cache HTML/screenshots |
| `images.py` | `resize_for_model` — the **one** screenshot transform every path shares |
| `labels.py` | `top_labels` (multi-label) and `project` (map a checkpoint onto the current space) |
| `training/splits.py` | `split_of(domain)` — train/val/test as a pure function of the domain |
| `fetchers.py` | `PlaywrightFetcher` (live), `ArchiveFetcher` (archive.org via `wayback`) |
| `commoncrawl.py` | Common Crawl as a third source: CDX index + `warcio` byte-range WARC |
| `politeness.py` | robots.txt via `protego`, per-host throttle, honest UA |
| `netsafety.py` | stdlib `ipaddress` address guard; see its docstring for what it cannot cover |
| `text.py` / `image.py` | Transformers inference paths (mmBERT / SigLIP 2) |
| `text_processor.py` | The live HTML→text cleaner |
| `outcomes.py` | `Stage`/`ErrorCode` taxonomy and run-report builder |
| `llm_classifier.py`, `llm/` | litellm-based classification |
| `piedomain.py` | **Legacy.** Only the static URL/domain validators are live |
| `cli.py` | `classify_domains` console script |

### Return shape

`classify()`, `classify_by_text()` and `classify_by_images()` return
`{"results": [...], "report": {...}}` — **not** a bare list. Every row carries
`status`, `stage`, `error_code` and `retryable`; the report gives counts by
reason and stage plus `missing`, the domains that produced nothing. Results are
reconciled against the requested list, so a domain the pipeline drops still
appears.

### Archive.org

Uses the `wayback` library (CDX + Memento), not hand-rolled availability-API
calls. Text is fetched with `Mode.original` (`id_`) — raw capture, no browser
needed. Screenshots render the `if_` URL, which hides the Wayback toolbar but
keeps archived CSS/images. Only `statuscode:200` captures are used, and the
realized capture is reported as `snapshot_timestamp`. The cache key includes the
archive date.

## Categories

**44** categories, defined in `constants.py`. `piedomains.constants.classes` is
the source of truth. Two of them are not topics: `parked` and `unavailable` mean
the domain resolves but there is no site.

The set is **deliberately not mutually exclusive** — four questions share one
vocabulary (status, topic, risk, what-a-site-*is*), and error rate tracks the
axis: status ~1%, topic ~15%, risk ~21%, form ~31%. `classify()` therefore
returns a `categories` list alongside the argmax. That lifts the chance of
reporting the right label from 81.8% to 87.3% at 1.30 labels per domain — **read
that as recall**, since the gold is single-label and nothing establishes whether
a second label is correct.

## Model state — read before trusting any accuracy number

Both checkpoints are Hub-hosted and unpinned, so `soodoku/piedomains-{text,image}`
is what ships.

- **Text: accuracy 0.818, macro-F1 0.758** over 44 classes, temperature 1.248,
  ECE 0.049 → 0.017. Replaced the TensorFlow model that measured 0.267/0.191.
- **Image: accuracy 0.501, macro-F1 0.370** over **39** classes — no `parked`,
  `unavailable`, `library`, `military` or `homestyle`, because the screenshot
  corpus was captured in 2022 and predates those labels. Replaces a model whose
  honest figure was 0.429. Screenshots are simply weaker than text; prefer text
  when you have it.
- Backbone choice was measured, not assumed: SigLIP 2 beat ViT-base-in21k on
  this corpus **0.531/0.397 vs 0.335/0.140**.

**Fusion does not help, and this is now settled.** On 1,604 paired domains the
fitted text weight is 0.986 — the image model carries 1.4% of the decision and
the best variant by macro-F1 changes *zero* predictions (McNemar p=1.000).
Stacking buys 0.6pp of accuracy while losing 1.1pp of macro-F1, p=0.25. So
`classify(use_screenshots=True)` falls back to text and screenshot
classification stays opt-in; `fusion.json` is deliberately **not** published.

Re-check it locally with `data/fusion-corpus` (8,825 held-out screenshots, 86 MB,
exported by the image kernel) — it no longer needs a Kaggle session.

**Known defect — `drugs` absorbs retail.** Only 28% of the 672 `drugs` training
documents mention any drug or pharmacy term (vs 4% of `shopping`); the rest are
expired pharmacy domains recycled into SEO spam, plus parked and dead pages the
extraction missed. Live: `walmart.com` → drugs 0.56, `zappos.com` → drugs 0.40,
and `drugs` is top-2 for four of five retailers tested. This is a labelling
problem, not a data-volume one.

**Weak classes are incoherent, not starved.** Training volume does not predict
per-class F1 in this corpus:

| class | train n | F1 |
|---|---|---|
| `military` | 93 | **0.800** |
| `weapons` | 161 | 0.837 |
| `cooking` | 196 | 0.880 |
| `library` | 96 | 0.333 |
| `urlshortener` | 158 | 0.154 |
| `socialnet` | 217 | 0.372 |
| `shopping` | **1,337** | **0.453** |

`military` at 93 documents beats `shopping` at 1,337. What separates them is
whether the class names one recognisable thing: `shopping` says what a site
*does* while competing against topics, and `urlshortener` is a redirect stub with
almost no page to read. So the remedy for a weak class is a cleaner definition,
not more documents — and a small coherent class is a perfectly good outcome.

Do not use `--min-docs` as evidence that a class is unviable; it is a CLI default,
not a measurement.

Do not add accuracy claims to docs without a measured number.

## Versioning

The **git tag is the version** (`uv-dynamic-versioning`); no version strings in
source, and `__version__` comes from `importlib.metadata`. Tags stop at v0.3.2
for 0.4.0–0.5.0 because those were published by manual `workflow_dispatch`; the
baseline was re-established at `v0.6.0`.

Publishing stays in `python-publish.yml`, **not** `release.yml`: this project's
PyPI trusted publisher predates py-canon adoption and is keyed to that filename.
OIDC claims reference the workflow file, so moving it breaks trusted publishing.

## Conventions

- No backward-compatibility shims unless explicitly asked for.
- Model artifacts under `src/piedomains/model/shallalist/` are gitignored (491MB,
  downloaded on first use); the isotonic `calibrate/` files are tracked.
- Tests are unittest-style; `tests/**` per-file-ignores in `pyproject.toml`
  cover the resulting ruff idioms.
