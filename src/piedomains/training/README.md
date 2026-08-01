# Training and evaluation

Scripts for measuring the current models and for retraining them. Nothing here
is imported by the package; `torch` and `transformers` live in the `train`
dependency group so installing `piedomains` does not pull them.

```bash
uv sync --group train
```

## They ship with the package

Every accuracy figure in the README comes out of these scripts, and a number
nobody can re-run is a number taken on faith. So they install alongside the
package rather than living only in this repository — `pip install piedomains` is
enough to reproduce any published claim.

```bash
classify_domains --training-scripts     # prints where they landed
cd "$(classify_domains --training-scripts)"
python evaluate.py --method text --out /tmp/eval.json
```

From a checkout they resolve here instead, so the same commands work either way.
`tests/test_cli.py` fails if any script behind a published number stops shipping.

## Why retrain

Measured, not assumed — run `evaluate.py` yourself and see:

| | reported at training time | measured on `tests/eval/labels.csv` |
|---|---|---|
| text model | 71.3% accuracy | **accuracy 0.395, macro-F1 0.262** |
| image model | 52.9% accuracy | labels Khan Academy and Yahoo as `porn` |

The text numbers are after the acquisition work, and that work is most of the
movement so far — 0.267/0.191 before it, on the same labels and the same model.
Fixing how pages are fetched was worth about +0.13 accuracy; the remaining gap to
0.713 is the model, which is what the rest of this directory is for.

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
uv run python download_corpus.py --set labels --out data/labels   # ~10MB
uv run python prepare_text.py \
    --corpus data/corpus/shallalist_all.tar.gz \
    --out data/prepared --max-per-class 2000
```

**Do not extract the tarball.** It is 18GB compressed and expands past 47GB,
which will fill a normal disk — it took this one to 100% and had to be killed
mid-run. `prepare_text.py` streams it instead; every document is read once
either way, so streaming costs nothing. A pre-unpacked `<category>/*.html` tree
is still accepted if you have one.

**Labels do not come from the tarball.** It is flat —
`shallalist_all/<domain>.html`, no category anywhere in the path — and
`screenshot-index.tab` carries only `index,full_domain`. The original notebooks
fetched `<category>/domains` lists from the `cbuijs/shallalist` GitHub mirror,
and **that repository is gone**; Shallalist was discontinued in 2022.
`Azothyran/ShallalistMirror` is a surviving copy with all 74 category
directories, and is what the script reads. Fetched lists are cached under
`--label-cache`, so the network is hit once.

Reproduces the original run's filtering, which is why the model has 39 classes
and not the 73 in `shallalist_cats.txt`: `chat`, `hacking` and `webtv` are
dropped by name, then any category with fewer than 100 documents, then any
document under 6 tokens. Shallalist nests (`sports`,
`finance/banking`); those collapse to their parent, which is exactly how 74 raw
categories become 39 — the seven parents with no directory of their own
(`automobile`, `education`, `finance`, `hobby`, `recreation`, `science`, `sex`)
are the tell. `--keep-subcategories` opts out.

`--max-per-class` caps documents per class. The corpus is long-tailed and only
~750k of the 1.5M labelled domains appear in it at all, so the realistic yield
is **34,627 documents across the 39 classes** — `education` 1,506 at the head,
`military` 85 at the tail.

Deduplication runs on page text **before** the domain prefix is added. Doing it
after makes every copy of shared navigation chrome look unique, because each
carries a different prefix — that is a real trap, and the reason the step is
ordered this way.

Writes `train/val/test.jsonl` (80/10/10) plus `labels.json`.

### 4. Train

```bash
uv run python train_text.py --data data/prepared --out models/text-v2
uv run python train_text.py --data data/prepared --out models/text-v2 --resume
```

Fine-tunes [mmBERT](https://github.com/JHU-CLSP/mmBERT) with a pooled CLS head.
Multilingual on purpose — but that only became real once `filter_non_english` was
turned off.

**Check `filter_non_english` before training anything.** It keeps only words in
NLTK's `words` corpus, a Webster's-era dictionary with no brand names and no
inflected forms. It was on by default through v0.8.0, which meant a multilingual
encoder was trained and served on text with every non-English word already
deleted. On *English* pages it discards 39.8% of tokens: `bbc.com` loses
`america`, `american`, `accuses` and `acclaimed`; other pages lose `spotify`,
`facebook`, `download` and `email` — the tokens that identify a site.

Its effect on the corpus is not marginal:

| | filter on | filter off |
|---|---|---|
| training documents | 27,702 | 34,464 |
| median words per document | 73 | 148 |
| mean words per document | 121 | 228 |

With it on, half of all documents were *shorter* than the 128-token budget the
model reads. Train/serve parity means both sides read the same config, so
changing it is a retrain rather than a config tweak.

**`mmBERT-small` is the default, on a measurement.** On an M4 with 16GB unified
memory:

| model | params | s/step | min/epoch |
|---|---|---|---|
| `mmBERT-base` @ len 256, bs 16 | 308M | 11.42 | 329 |
| `mmBERT-base` @ len 128, bs 32 | 308M | 10.74 | 155 |
| **`mmBERT-small` @ len 128, bs 32** | **141M** | **1.45** | **21** |

A 7x gap for 2.2x the parameters is memory pressure, not compute — base plus
Adam states does not fit comfortably. On a real GPU base is the better model:
pass `--model jhu-clsp/mmBERT-base --max-length 256`.

Checkpoints every epoch and `--resume` picks up the newest, so a dropped session
costs one epoch rather than the run. Early stopping is on val macro-F1, not
accuracy — the class distribution is long-tailed and accuracy flatters a model
that only learns the head.

### 5. Calibrate

```bash
uv run python calibrate.py --model models/text-v2 --data data/prepared
```

Temperature scaling fitted on the **validation** split with LBFGS: one scalar,
`softmax(logits / T)`. Fitting on train would return T ≈ 1 and calibrate
nothing, since the model has already memorized it.

This replaces 39 per-class `IsotonicRegression` pickles that had three separate
problems: they were applied elementwise and never renormalized, so `confidence`
was not a probability and argmax was not the argmax of any distribution; they
are unpinned joblib pickles, so they became a version-compatibility surface; and
via that surface they had stopped working entirely — every one unpickles under
scikit-learn 1.9 and then predicts `NaN`, so all 39 were silently discarded at
load. Writes `calibration.json` and reports ECE before and after.

## Files

| File | Status |
|---|---|
| `metrics.py` | done — macro-F1, per-class P/R, confusions, ECE |
| `evaluate.py` | done — scores a labelled set, writes JSON |
| `download_corpus.py` | done — resumable, checksum-verified |
| `prepare_text.py` | done — filtering reproduced from notebooks/04 |
| `train_text.py` | done — mmBERT fine-tune, resumable, early stopping |
| `calibrate.py` | done — temperature scaling on val, text or image |
| `taxonomy.py` | done — Shallalist to the 47-class label set |
| `prepare_images.py` | done — streams the screenshot tarballs, resizes to 224px |
| `train_image.py` | done — ViT fine-tune, fully unfrozen |
| `fuse.py` | done — fits late fusion, gates on beating text alone |
| `kaggle/train_image_kaggle.py` | done — the whole image pipeline on Kaggle's free tier |

The `notebooks/` directory holds the original Colab pipeline that produced the
shipped models. It is kept for provenance; the paths in it are Colab-specific
(`/content/drive/MyDrive/...`) and it is excluded from linting.


## 6. The image model

Run it on [Kaggle](https://www.kaggle.com/code): 30 GPU-hours a week free, 12-hour
sessions, 200 GB of dataset storage. The job needs 3–5 hours, so it costs nothing. Paste
`kaggle/train_image_kaggle.py` into a notebook with Accelerator set to GPU and Internet
on.

**Never download the whole corpus.** It is 47.58 GB across 28 tarballs — more than
Kaggle's 20 GB persistent volume and, on the machine this was written on, more than the
free disk. The notebook fetches one tarball at a time, streams it through
`prepare_images.py` to write 224px copies, and deletes it before fetching the next. Peak
disk is one tarball plus the growing resized set.

Publish that resized set as a Kaggle Dataset when it finishes. Later sessions mount it
read-only and skip straight to training, which matters because the 12-hour session cap
will interrupt you at least once.

```bash
uv run python -m piedomains.training.prepare_images --corpus data/corpus --out data/images-224
uv run python -m piedomains.training.train_image --data data/images-224 --out models/image-v1
uv run python -m piedomains.training.calibrate --model models/image-v1 \
    --data data/images-224 --modality image
```

### Fusion, and the bar it has to clear

```bash
uv run python -m piedomains.training.fuse \
    --text models/text-v4 --image models/image-v1 \
    --text-data data/prepared-taxonomy --image-data data/images-224
```

Both models are calibrated first, because averaging an uncalibrated softmax against a
calibrated distribution is exactly what the original package did — and why its "ensemble"
could not change an answer. Weights are fitted on the 26,280 domains that have both a
page and a screenshot, not assumed to be 0.5.

`fuse.py` exits non-zero when fusion fails to beat text alone. **Text alone is accuracy
0.725 / macro-F1 0.705.** If fusion does not clear it, image classification ships opt-in
with that number in the README rather than because we built it.

### What a 224px screenshot can say

Not much text — a page is unreadable at that resolution, so the model sees layout, colour
and gross structure. If image-only macro-F1 is poor, resolution is the first lever
(`--size 384`, roughly 3× the compute), not more epochs.

Also worth knowing: the screenshots are 2022 captures and inference runs on today's
pages. Visual design ages faster than prose, so expect the image model to transfer worse
than its held-out score suggests. The evaluation set is the check, since its screenshots
were taken by our own pipeline this week.
