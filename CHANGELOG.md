# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0] - 2026-07-31

The model was reading an alphabetised set of words.

`TextProcessor.clean_and_normalize_text` deduplicated tokens twice, sorted them
alphabetically and stripped every non-ASCII character. The stored training text was
literally `"accueil adresse alfonso aller anciens animation ans archives..."`. Across 14
real pages that discarded **73% of all words**; `asahi.com` kept 2.7% of its own.

Three things were lost, and each mattered more than it sounds:

- **Term frequency.** 200 mentions of sport and one sportsbook advert in the footer
  weighed the same. That is the mechanism behind `deadspin.com` being classified `gamble`
  at 0.98 confidence.
- **Word order**, to an alphabetical sort — which nullifies the reason to use a contextual
  encoder at all.
- **Every non-Latin script**, making a multilingual model multilingual in name only.

None of it was unreasonable for the model it was built for: a `GlobalAveragePooling1D`
bag-of-embeddings is order-invariant by construction. It simply was not revisited when the
model became mmBERT.

### `parked` is a category now, and it is the best class in the model

Restoring term frequency exposed something the old cleaner had been hiding: **7.9% of the
training corpus is domain-parking placeholders**, and they concentrate hard.

| class | was parking pages |
|---|---|
| **drugs** | **42%** |
| webmail | 23% |
| downloads | 18% |
| adult | 17% |

Expired pharmacy domains get parked, so the model had learned that a "this domain is for
sale" template *means* drugs — the complete explanation for `zappos.com`, `newlook.com` and
`suicidepreventionlifeline.org` all returning `drugs` at low confidence on an independent
test set. Exact-text deduplication missed all of it, because the pages differ only by the
domain name embedded in them.

`parked` now scores **F1 0.992** on 378 held-out documents. Contamination is 0.0% in every
class it was taken from. `ringtones` falls below the 100-document floor as a result — it
was largely parked domains — so the label set is 47: the previous set minus `ringtones`,
plus `parked`.

### Measured

| benchmark | v0.11 | v0.12 |
|---|---|---|
| **Curlie x Tranco agreement** (155 domains, independent human labels) | 0.529 | **0.543** |
| hand-labelled eval, defensible alternates credited (49) | 0.735 | 0.714 |
| calibration ECE | 0.149 | **0.010** |
| blockable-category rate on Tranco-top-100k | 13% | **9%** |

The differences on the small sets are inside noise (SE ≈ 0.04 and ≈ 0.065). The Curlie
figure is the one to weigh: independent labels, larger sample, and it shares neither the
taxonomy nor the selection bias of the training corpus.

### Two things that looked obviously right and were not

**Stripping standalone punctuation.** Table pipes and layout dashes are 4.8% of tokens and
40% of the 99th-percentile page, so removing them seemed clearly correct. Trained both ways
on otherwise identical corpora it was *worse*: Curlie 0.543 → 0.523, held-out macro-F1
0.7267 → 0.7134, and `deadspin.com` went back to `gamble`. Structural punctuation evidently
says something about what kind of page it is. Available as `strip_punctuation=True`.

**trafilatura as the extractor.** It cuts `deadspin.com`'s gambling tokens from 260 (7.1%
of the page) to 7 (1.1%), which matters far more now that frequency is preserved — under
the old deduplicating cleaner both collapsed to one. It is the default, and an earlier note
in this repo claiming it was worse was measured wrong: it compared trafilatura's raw words
against the legacy cleaner's *cleaned* tokens.

### The ensemble was built, measured, and not shipped

Combining the two models was tried four ways, all scored on the same 1,725 paired domains
with both checkpoints verified disjoint from that split:

| combiner | out-of-fold CV | test macro-F1 | vs text |
|---|---|---|---|
| **text only** | — | **0.7067** | — |
| image only | — | 0.3306 | −0.376 |
| stacked, logistic | 0.6534 | 0.6749 | −0.032 |
| stacked, gradient-boosted | 0.6363 | 0.6550 | −0.052 |
| stacked, MLP | 0.6488 | 0.6632 | −0.044 |

Every combiner is **worse** than text alone, and the nonlinear ones are worse than the
linear one — 1,770 fitting examples across 47 classes is ~38 each, and the extra capacity
buys overfitting. Weighted per-class fusion independently reached the same conclusion by
driving the text weight to 1.0, which is the fitted way of saying "ignore the screenshot".

So `classify()` stays text-only and the screenshot model stays opt-in. A 224px screenshot
carries little that the page text does not, and at 0.331 macro-F1 it is too weak for a
combiner to recover anything from.

### Added

- `strip_punctuation` and `text_cleaning` config, both defaulting to the measured winner.
  `text_cleaning="legacy"` reproduces v0.11 exactly.
- `piedomains.training.stack` — selects a late-fusion combiner across logistic,
  gradient-boosted and MLP families by out-of-fold macro-F1 on stratified 5-fold CV.
- `piedomains.training.train_joint` — a two-tower model over both representations.
- Anti-bot interstitials are refused at corpus-preparation time rather than trained on.

### Fixed

- **Training metrics described weights nobody could load.** `evaluate_split` scored the
  in-memory model, which after early stopping is the *last* epoch, while the artifact on
  disk is the *best*. Both trainers now reload the saved checkpoint and log which epoch
  they are scoring.
- **A checkpoint could be scored on data it trained on.** Re-preparing the text corpus
  reshuffled the split assignments, so 73% of the new test domains were in the old training
  set. The guard compared *split files*, which agreed perfectly because both had been
  rebuilt with `--respect-splits` — it could not see that the checkpoint predated them.
  Checkpoints now record their own training domains and fusion interrogates that.
- `fuse.py` chose between fusion forms by comparing macro-F1 on the **test** split, making
  the reported number optimistically biased. It now chooses on the fit split.

## [0.11.0] - 2026-07-30

Screenshot classification works again, and it is opt-in because the measurement says so.

`classify_by_images()` has raised since 0.8.0, when the old ResNet50 was withdrawn: it
reported 52.9% with `base_model.trainable = False` — only a linear head was ever fitted —
and in production labelled Khan Academy and Yahoo as `porn`, because the serving path
divided pixels by 255 before a graph that already baked in `resnet50.preprocess_input`.
It is replaced by a fully fine-tuned, temperature-calibrated SigLIP2 model.

### The number that matters

**On screenshots taken today the image model scores 0.317 accuracy / 0.212 macro-F1.**
Not the 0.429 it gets on the corpus it was trained from. The training screenshots are 2022
captures and inference runs on pages rendered now; measured on 183 self-captured
screenshots of held-out domains, that four-year shift costs about 0.11 accuracy.

### Fusion was measured and not adopted

Calibrated late fusion, fitted on 1,704 held-out paired domains and scored on 1,742 more:

| model | accuracy | macro-F1 |
|---|---|---|
| text only | 0.794 | 0.699 |
| image only | 0.429 | 0.306 |
| fused (per-class) | 0.798 | 0.700 |

+0.001 macro-F1 is noise at that sample size, and the fitted text weight is **0.973** — the
optimiser puts almost nothing on the screenshot. So `classify()` now returns the text
answer by default and does not load a 350MB vision model to gain nothing.

**Read this before re-deriving a different number.** The first fusion run reported image
only at 0.768 and made fusion look like a clear win. It was wrong. `prepare_text.py` and
`prepare_images.py` each shuffled *their own* list with the same seed, and the lists differ
(46,754 documents against 44,712 screenshots), so a domain landed in unrelated splits on
each side — about 80% of the domains fusion scored on were in the image model's training
set. `prepare_images.py --respect-splits` fixes it; image only fell 0.768 → 0.429.

### Breaking

- **`classify()` is text-only by default.** Pass `use_screenshots=True` to fuse. The CLI's
  `--method` default moves from `combined` to `text` for the same reason.
- `classify()`'s docstring no longer claims to be "the most comprehensive classification
  method ... for maximum accuracy". It was not.

### Added

- **The training scripts ship with the package**, as `piedomains.training`. Every accuracy
  figure here is produced by one of them, and a number nobody can re-run is a number taken
  on faith. `classify_domains --training-scripts` prints where they installed; each runs as
  `python -m piedomains.training.<name>`.
- `piedomains.training.validate_curlie` — scores the model against Curlie's human labels on
  Tranco-ranked domains, an independent set that shares neither the taxonomy nor the
  selection bias of the training corpus.
- `piedomains.training.capture_screenshots` — captures through the same `DataCollector`
  used at inference, so training data matches serving.
- `screenshot_scale` config (default 1), threaded into Playwright's `device_scale_factor`.

### Fixed

- **Archived captures bypassed the thin-content floor.** Archive text is fetched as raw
  HTML rather than rendered, so it never reached the check the live path has had since
  0.7.0. `sapphirecasino.com` was recorded as a successful fetch from a 114-byte capture
  holding four characters of text — a blank document and a blank screenshot, stored as data.
- **Corpus downloads gave up on transient failures.** One 502 from Harvard Dataverse killed
  a two-hour training run. Downloads now retry 429/5xx with backoff and resume a broken
  transfer by range; a tarball that still fails is skipped, named and counted rather than
  aborting the run.
- Browser-context settings are built in one place, so screenshots taken by the batch path
  cannot drift from those taken by the single-domain path.

## [0.10.0] - 2026-07-28

Stops asking the model questions a page cannot answer.

The 39 classes conflated three unrelated questions — what a site is about, whether you
would block it, and how it is built and monetised — into one mutually-exclusive softmax.
`amazon.com` is a shopping site that also runs one of the largest ad networks on the web;
forced to choose, the model picked `adv` and we scored it wrong.

One rule decides every case: **is it visible in the page text?**

Measured on `tests/eval/labels.csv`, 51 scored, same cached content throughout:

| | accuracy | macro-F1 | English acc/F1 | non-English acc/F1 |
|---|---|---|---|---|
| 0.8.0 | 0.627 | 0.602 | 0.643 / 0.642 | 0.556 / 0.472 |
| 0.9.0 | 0.608 | 0.707 | 0.667 / 0.722 | 0.333 / 0.317 |
| **0.10.0** | **0.725** | 0.705 | **0.738 / 0.729** | **0.667 / 0.536** |

10 domains fixed against 0.9.0, 4 regressed. Every fix the change predicted landed:
`amazon`, `etsy` and `walmart` → `shopping` (were `adv`), `paypal` → `finance` (was
`spyware`), `bitly` → `urlshortener` (was `adv`). It also **reversed 0.9.0's non-English
regression** and beat 0.8.0 there — most likely because the removed classes had been
absorbing pages the model could not read, foreign ones included.

Regressed: `coursera.org`→`jobsearch`, `irs.gov`→`finance`, `nature.com`→`news`,
`nih.gov`→`drugs`. Institutions whose subject matter genuinely overlaps another category.

Calibration is the best of the three: **T = 1.812, ECE 0.123 → 0.010**.

### 💥 Breaking — the label set changed

47 classes, not 39. See `training/taxonomy.py` for the mapping and its reasoning.

- **Removed**: `adv`, `tracker`, `spyware`, `redirector`. These describe who runs a site
  and how it is monetised, which a page does not state — a homepage selling handmade
  goods reads identically whether or not the operator also runs trackers. They were 4.1%
  of training data and sat behind a third of evaluation errors. Cloudflare reaches the
  same conclusion from the other side: it runs these off domain age, reputation and DNS
  behaviour, never off content. `redirector` is now an outcome, since such a page has no
  content by construction.
- **Split**: `recreation` → `recreation/{sports,travel,humor,restaurants,wellness}` and
  `hobby` → `hobby/{pets,games-online,games-misc,gardening,cooking}`. `recreation` was
  98% travel-or-sports (278,346 of 283,587); no annotator could apply it consistently, so
  neither could a model, and any evaluation measuring it was partly noise.
- **Merged**: `porn` + `sex` + `models` → `adult`, which can be thresholded for recall as
  a single safety signal rather than a three-way choice.
- **Promoted**: `finance/realestate` → `realestate`, as IAB, Cloudflare and WebOrganizer
  all treat it. `sex/education` → `education` rather than inheriting `adult` —
  conflating sexual health with adult content is what makes filters block the resources
  people most need, and Shallalist's own description concedes the category "can be
  misdetected as porn".

No identity categories exist, deliberately. Cloudflare files LGBTQ beside
`Lingerie & Bikini` and `Swimsuits`; filtering systems that treat identity as sexual
content have a documented history of harm.

`prepare_text.py --raw-categories` reproduces the pre-0.10 label set.

## [0.9.0] - 2026-07-28

Stops discarding 40% of every page before the model reads it.

`filter_non_english` kept only words in NLTK's `words` corpus — a Webster's-era
dictionary with no brand names and no inflected forms. Measured across 20 evaluation
pages it discarded **39.8% of tokens on English pages**: `bbc.com` alone lost `america`,
`american`, `accuses` and `acclaimed`; other pages lost `spotify`, `facebook`,
`download`, `email`. Retraining without it doubles the median document from 73 to 148
words and lifts 6,762 more documents over the minimum-token floor.

Measured on `tests/eval/labels.csv` — same 54 domains, same cached content, only the
filter differs:

| | 0.8.0 | 0.9.0 |
|---|---|---|
| **English** (n=42) accuracy / macro-F1 | 0.643 / 0.642 | **0.667 / 0.722** |
| non-English (n=9) accuracy / macro-F1 | 0.556 / 0.472 | 0.333 / 0.317 |
| overall (n=51) | 0.627 / 0.602 | 0.608 / **0.707** |

**The non-English column is a regression, and the reason is worth stating plainly: this
package is not multilingual, and removing the filter revealed that rather than causing
it.** With the filter on, a French or Japanese page was stripped to the few English
tokens it contained — brand names, `news`, `shop` — and that residue happened to match
the English training distribution. The filter was an accidental domain-adaptation step.
Without it the model sees full French and Japanese, which the Shallalist corpus never
taught it. A multilingual encoder does not make a multilingual system if the training
corpus is English; that needs multilingual training data, not a config flag.

Caveat on that column: **n=9, and the difference is two domains.** It is suggestive, not
established. The English figures rest on 42 domains.

Fixed by this change: `bbc.com`→news, `coursera.org`→education, `nature.com`→science,
`imgur.com`→imagehosting, `leboncoin.fr`→shopping. Regressed: `amazon`/`walmart`→adv,
`paypal`→spyware, plus the three non-English cases above. Three of those six regressions
land in `adv`/`spyware` — classes a page never states, and which a following release
removes from the label space entirely.

### Changed

- `filter_non_english` now defaults to `False`. The flag remains, so 0.8.0's numbers stay
  reproducible, and a test pins the default — flipping it back silently would be
  train/serve skew, not merely a quality regression.
- `tests/eval/labels.csv` gains ten non-English domains. The set was entirely English, so
  the multilingual claim was untestable rather than merely untested.

### Fixed

- `train_text.py` saved a checkpoint every epoch while printing "new best" only on
  improvement, so the artifact left on disk was the *last* epoch rather than the best.
  0.8.0 shipped epoch 4 (val 0.6582) over epoch 3 (0.6634); this release ships the best
  epoch, as intended.

## [0.8.0] - 2026-07-27

Measured on `tests/eval/labels.csv` (44 hand-labelled popular domains) against
identical cached content, so this is model-vs-model with acquisition held constant:

| | TensorFlow (0.7.0) | PyTorch (0.8.0) |
|---|---|---|
| accuracy | 0.395 | **0.590** |
| macro-F1 | 0.262 | **0.607** |
| ECE | 0.210 | **0.190** |

13 domains change from wrong to right, 5 the other way. Fixed: `khanacademy.org` and
`mit.edu` → `education`, `google`/`bing` → `searchengines`, `irs.gov`/`usa.gov`/`nih.gov`
→ `government`, `spotify` → `music`, `imdb`/`netflix` → `movies`, `mayoclinic` →
`hospitals`, `tinyurl` → `urlshortener`. Regressed: `cnn`/`bbc` → `radiotv` (news
outlets with video), and `amazon`/`paypal`/`etsy` → `adv`/`spyware` — both are large,
noisy Shallalist categories.

On a held-out split of the training corpus the model scores accuracy **0.734**,
macro-F1 **0.648**. The old model reported 71.3% on that kind of split and delivered
0.395 on the set above; the gap is what an unchecked in-distribution number looks like.
**0.59 is the honest expectation.**

Calibration is the other half. The raw model is badly overconfident; temperature
scaling (T = 3.416, fitted on validation) takes expected calibration error from
**0.203 to 0.022** on the corpus test split.

### 💥 Breaking

- **TensorFlow is removed.** It ships no `cp314` wheels, which held `requires-python`
  below 3.14 — the standard py-canon matrix. `torch` does, so the upper bound is gone
  and CI runs 3.11 + 3.14.
- **Image classification is unavailable** while the screenshot model is retrained.
  `classify_by_images()` raises `NotImplementedError` naming the reason, and
  `classify()` is text-only.

  This changes no labels. The old "combined" path never merged the two probability
  vectors: it returned the *text* label every time and only averaged the confidences —
  a calibrated, unnormalized text score against an uncalibrated image softmax — so the
  image model could not change an answer, only blur the number attached to it. The
  model it removes reported 52.9% with `base_model.trainable = False` (only a linear
  head was ever fitted) and in practice labelled Khan Academy and Yahoo as `porn`.
  Screenshot *collection* is unaffected.
- Result rows no longer carry `text_category` / `image_category` / `image_confidence`,
  which described a contribution that was never made.

### Changed

- **The text model is a fine-tuned [mmBERT](https://github.com/JHU-CLSP/mmBERT)
  encoder**, replacing `Embedding(525473, 64) → GlobalAveragePooling1D → Dense` — 403 MB
  of embedding table that could not use word order, so `free shipping returns` and
  `free returns shipping` were the same input. Multilingual by design: the old pipeline
  strips non-dictionary words and applies NLTK *English* stopwords, so a non-English
  page degraded to noise before reaching the model.
- **Confidence is a probability again.** Temperature-scaled softmax replaces 39 per-class
  `IsotonicRegression` pickles that were applied elementwise and never renormalized — so
  `confidence` was not a probability and `argmax` was not the argmax of any distribution.
  They had also stopped working entirely: every one unpickles under scikit-learn 1.9 and
  then predicts `NaN`, so all 39 were silently discarded at load and reported confidences
  were raw model outputs.
- **Class order is read from the checkpoint**, not a module constant, so retraining
  cannot silently permute every label.
- Model weights load from the Hugging Face Hub via `transformers`, replacing a hardcoded
  Dataverse datafile id that had no checksum verification and made `latest=True` a no-op
  re-download of the same artifact. Override with `PIEDOMAINS_TEXT_MODEL`.

### Fixed

- `training/prepare_text.py` could not consume the corpus at all:
  - **It extracted the tarball.** `shallalist_all.tar.gz` is 18 GB compressed and expands
    past 47 GB, which fills a normal disk. It now streams.
  - **It looked for labels in the archive.** The archive is flat
    (`shallalist_all/<domain>.html`, no category in the path), and the
    `cbuijs/shallalist` mirror the original notebooks read now 404s — Shallalist was
    discontinued in 2022. `Azothyran/ShallalistMirror` still has all 74 category
    directories and is what is read and cached.
- `piedomain.py` drops from 598 to 302 lines; everything removed was TensorFlow plumbing
  with no caller outside the module.

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

Measured on `tests/eval/labels.csv` (44 domains, `training/evaluate.py --method text`),
against the same model and the same labels — this release changes only how pages are
acquired:

| | before | after |
|---|---|---|
| domains accounted for | 33 | **44** |
| accuracy | 0.267 | **0.395** |
| macro-F1 | 0.191 | **0.262** |
| ECE | 0.219 | **0.210** |
| failures | silent, or `unknown` | every one named |

On the 33 domains the old pipeline returned at all, accuracy goes 0.267 → 0.407 and
macro-F1 0.191 → 0.301. The remaining gap to the 71.3% reported at training time is the
model, not acquisition; see `training/README.md`.

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
