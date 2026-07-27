# Training and evaluation

Scripts for measuring the current models and for retraining them. Nothing here
is imported by the package; `torch` and `transformers` live in the `train`
dependency group so installing `piedomains` does not pull them.

```bash
uv sync --group train
```

## Why retrain

Measured, not assumed — run `evaluate.py` yourself and see:

| | reported at training time | measured on `tests/eval/labels.csv` |
|---|---|---|
| text model | 71.3% accuracy | **accuracy 0.267, macro-F1 0.191** |
| image model | 52.9% accuracy | labels Khan Academy and Yahoo as `porn` |

Two further findings from that run:

- **Calibration is inactive.** All 39 pickled isotonic calibrators unpickle
  under scikit-learn 1.9 and then predict `NaN`, so every one is discarded and
  reported confidences are raw model outputs. Temperature scaling replaces them.
- **Shallalist is gone.** The taxonomy was discontinued in 2022, so labels
  cannot be refreshed from source. UT-Capitole (UT1) is the maintained
  successor and is what `download_corpus.py --set ut1` fetches.

## Pipeline

### 1. Evaluate what exists

```bash
uv run python evaluate.py --method text --out baseline.json
uv run python evaluate.py --method images
uv run python evaluate.py --method combined
```

Reports accuracy, macro-F1, per-class precision/recall, top confusions, and ECE.
Content is cached under `--cache-dir`, so repeat runs need no network.

**No accuracy claim should be added to any doc without a number from this.**

### 2. Fetch corpora

```bash
uv run python download_corpus.py --list                      # sizes first
uv run python download_corpus.py --set labels  --out data/   # ~10MB, start here
uv run python download_corpus.py --set shallalist_all --out data/   # ~18GB
uv run python download_corpus.py --set screenshots    --out data/   # ~48GB
uv run python download_corpus.py --set ut1            --out data/   # fresh labels
```

Downloads resume with HTTP Range and verify every part's published MD5, which
matters at 2.15GB per part. Split tarballs are reassembled automatically.

Two Dataverse quirks the script handles, both of which cause confusing checksum
failures otherwise: tabular files (`.tab`, `.csv`) are served in a *converted*
form unless you ask for `format=original`, and the reported `filesize` describes
that converted copy rather than the original.

Total catalogue is ~75GB. Check free disk before starting.

### 3. Prepare text

```bash
tar xzf data/shallalist_all.tar.gz -C data/shallalist
uv run python prepare_text.py --corpus data/shallalist --out data/prepared
```

Reproduces the original run's filtering, which is why the model has 39 classes
and not the 73 in `shallalist_cats.txt`: `chat`, `hacking` and `webtv` are
dropped by name, then any category with fewer than 100 documents, then any
document under 6 tokens.

Deduplication runs on page text **before** the domain prefix is added. Doing it
after makes every copy of shared navigation chrome look unique, because each
carries a different prefix — that is a real trap, and the reason the step is
ordered this way.

Writes `train/val/test.jsonl` (80/10/10) plus `labels.json`.

### 4. Train

Not yet written. The intended target is a `mmBERT-base` fine-tune with a pooled
CLS head, lr 1e-5–5e-5, weight decay ~0.01 — chosen because it is multilingual,
and the current pipeline is English-only (it strips non-English words and uses
NLTK English stopwords, so non-English pages degrade to noise).

Run it on Colab or a rented GPU; checkpoint to Drive or object storage so a
disconnect does not cost the run.

### 5. Calibrate

Not yet written. Temperature scaling on the validation split: a single scalar
that always yields a normalized distribution, with no pickled-sklearn
compatibility surface. This is what makes the reported confidence mean
something again.

## Files

| File | Status |
|---|---|
| `metrics.py` | done — macro-F1, per-class P/R, confusions, ECE |
| `evaluate.py` | done — scores a labelled set, writes JSON |
| `download_corpus.py` | done — resumable, checksum-verified |
| `prepare_text.py` | done — filtering reproduced from notebooks/04 |
| `train_text.py` | **not written** |
| `calibrate.py` | **not written** |

The `notebooks/` directory holds the original Colab pipeline that produced the
shipped models. It is kept for provenance; the paths in it are Colab-specific
(`/content/drive/MyDrive/...`) and it is excluded from linting.
