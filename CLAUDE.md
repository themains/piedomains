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
| `fetchers.py` | `PlaywrightFetcher` (live), `ArchiveFetcher` (archive.org via `wayback`) |
| `text.py` / `image.py` | TensorFlow inference paths |
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

**39** categories, defined in `constants.py` — not 41, despite what older docs
said. `piedomains.constants.classes` is the source of truth.

## Model state — read before trusting any accuracy number

Measured by `training/evaluate.py` against `tests/eval/labels.csv`:

- **Text model: accuracy 0.267, macro-F1 0.191** — against a training-time
  figure of 71.3% (`notebooks/04_train_model.ipynb`). Baseline recorded in
  `tests/eval/baseline_text_tf.json`.
- **Calibration is inactive.** All 39 pickled isotonic calibrators unpickle
  cleanly under scikit-learn 1.9 but predict `NaN`, so every one is dropped and
  confidences are raw model outputs. Logged as a WARNING.
- **The image model is not trustworthy.** It labels Khan Academy and Yahoo as
  `porn` under both `/255` and raw-0-255 scaling. Do not "fix" the
  preprocessing line without re-measuring — the audit's premise that
  `resnet50.preprocess_input` is baked into the graph did not reproduce.

Do not add accuracy claims to docs without a number from `training/evaluate.py`.

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
