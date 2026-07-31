#!/usr/bin/env python3
r"""Combine the text and image models, and report whether it was worth it.

**The bug this exists to avoid.** The original package advertised an ensemble and did not
have one. It ran both models, returned the *text* label every time, and set confidence to
``(text_conf + image_conf) / 2`` — averaging an isotonic-calibrated, unnormalized text
score against a raw image softmax. The image model could not change an answer, only blur
the number attached to it.

Two things make this a real ensemble instead:

* **Both sides are calibrated first.** Temperature scaling puts the two probability
  vectors on the same scale, which is the precondition for combining them at all.
  Without it, whichever model is more overconfident wins by default.
* **The combination is fitted, not assumed.** Weights come from held-out data rather than
  from a hard-coded 0.5.

**Fusion is fitted on the overlap, not on everything.** Only 26,280 domains have both a
page and a screenshot, against 46,754 text documents and 248,003 screenshots. That is why
the encoders are trained separately and only the combiner sees paired data — joint
training would have discarded 89% of the screenshots.

Usage::

    uv run python training/fuse.py \
        --text models/text-v4 --image models/image-v1 \\
        --text-data data/prepared-taxonomy --image-data data/images-224

Reports text-only, image-only and fused side by side. **If fused does not beat text-only,
say so and ship image classification opt-in** — that is the whole point of measuring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .metrics import macro_f1, per_class_report
from .train_text import pick_device, read_jsonl


def load_calibrated(model_dir: Path) -> float:
    """Read a model's fitted temperature.

    Args:
        model_dir: Directory holding the checkpoint.

    Returns:
        float: The temperature, or ``1.0`` when absent.
    """
    path = model_dir / "calibration.json"
    if not path.exists():
        print(
            f"WARNING: {model_dir} has no calibration.json. Fusing uncalibrated "
            "outputs is the exact bug this script exists to avoid -- run "
            "calibrate.py first.",
            file=sys.stderr,
        )
        return 1.0
    return float(json.loads(path.read_text(encoding="utf-8")).get("temperature", 1.0))


def text_probabilities(
    model_dir: Path, rows: list[dict[str, Any]], labels: list[str], batch_size: int
) -> Any:
    """Score domains with the text model, temperature applied.

    Args:
        model_dir: Text checkpoint directory.
        rows: Records carrying ``text``.
        labels: Ordered class names.
        batch_size: Inference batch size.

    Returns:
        Any: A ``(n, classes)`` tensor of calibrated probabilities.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    from .train_text import TextDataset

    # ModernBERT compiles its encoder through TorchInductor, whose Triton backend requires
    # compute capability >= 7.0. On Kaggle's P100 (sm_60) that raises BackendCompilerFailed
    # and took fusion down with it. Eager is a few percent slower for one inference pass
    # and works everywhere.
    #
    # It has to be set on the config: `from_pretrained(..., reference_compile=False)` is
    # forwarded to __init__ and raises TypeError, which is only visible by trying it.
    # Set unconditionally, not behind `hasattr`: a freshly loaded config does not carry
    # the attribute at all (ModernBERT defaults it at __init__), so a hasattr guard never
    # fires and the P100 failure comes straight back. Configs accept extra attributes, so
    # this is inert for architectures that do not read it.
    torch._dynamo.config.suppress_errors = True
    config = AutoConfig.from_pretrained(model_dir)
    config.reference_compile = False

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, config=config
    ).to(device)
    model.eval()
    temperature = load_calibrated(model_dir)

    # Typed Any because torch's DataLoader stub demands a Dataset subclass, while the
    # runtime only needs __len__/__getitem__ -- which TextDataset has. See its docstring.
    dataset: Any = TextDataset(rows, tokenizer, labels, 128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            chunks.append(torch.softmax(logits.cpu() / temperature, dim=-1))
    return torch.cat(chunks)


def project(probabilities: Any, source: list[str], target: list[str]) -> Any:
    """Widen a distribution into a larger label space, zero for classes it cannot emit.

    The image model has 42 classes against the text model's 47: ``aggressive``,
    ``homestyle``, ``library``, ``military`` and ``ringtones`` had too few screenshots
    clearing the per-class floor. Refusing to fuse over that would throw away a usable
    model; asserting the two are identical would silently permute every class.

    Zero is the honest value — the screenshot model never votes for a class it was never
    trained on — but it does mean those columns lose mass under a scalar weight. That is
    what the per-class weight override in :func:`main` exists to correct.

    Args:
        probabilities: A ``(n, len(source))`` tensor.
        source: Class order of ``probabilities``.
        target: Class order to widen into; must contain every name in ``source``.

    Returns:
        Any: A ``(n, len(target))`` tensor.
    """
    import torch

    index = {name: i for i, name in enumerate(target)}
    widened = torch.zeros(probabilities.shape[0], len(target))
    for j, name in enumerate(source):
        widened[:, index[name]] = probabilities[:, j]
    return widened


def image_probabilities(
    model_dir: Path,
    rows: list[dict[str, Any]],
    image_dir: Path,
    labels: list[str],
    image_labels: list[str],
    batch_size: int,
) -> Any:
    """Score domains with the image model, temperature applied, in the text label space.

    Args:
        model_dir: Image checkpoint directory.
        rows: Records carrying ``domain``.
        image_dir: Directory of ``<domain>.jpg``.
        labels: The full (text) class order to return columns in. Also used for the
            dataset's own label lookup, since rows can carry categories the image model
            has no class for; those targets are discarded here, only logits are used.
        image_labels: The image checkpoint's own class order.
        batch_size: Inference batch size.

    Returns:
        Any: A ``(n, len(labels))`` tensor of calibrated probabilities.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from .train_image import ScreenshotDataset

    device = pick_device()
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
    model.eval()
    temperature = load_calibrated(model_dir)

    dataset: Any = ScreenshotDataset(rows, image_dir, labels, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            chunks.append(torch.softmax(logits.cpu() / temperature, dim=-1))
    return project(torch.cat(chunks), image_labels, labels)


def fit_weight(text_p: Any, image_p: Any, targets: Any, *, per_class: bool) -> Any:
    """Fit the fusion weight on held-out data.

    A single scalar mixes the two distributions globally; per-class weights let the image
    model carry more where it is actually informative — screenshots plausibly help more
    on `adult` and `shopping` than on `government`. Per-class costs one parameter per
    class, which 26k paired examples can support.

    Args:
        text_p: Calibrated text probabilities.
        image_p: Calibrated image probabilities.
        targets: Gold class indices.
        per_class: Fit one weight per class rather than one overall.

    Returns:
        Any: The fitted weight(s), in ``[0, 1]``, applied to the text side.
    """
    import torch

    classes = text_p.shape[1]
    raw = torch.zeros(classes if per_class else 1, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.1, max_iter=200)
    loss_fn = torch.nn.NLLLoss()

    def closure():
        optimizer.zero_grad()
        w = torch.sigmoid(raw)
        mixed = w * text_p + (1 - w) * image_p
        loss = loss_fn(torch.log(mixed.clamp_min(1e-9)), targets)
        loss.backward()
        return loss

    optimizer.step(closure)  # pyright: ignore[reportArgumentType]
    return torch.sigmoid(raw).detach()


def fit_stacker(
    fit_text: Any, fit_image: Any, fit_targets: Any, labels: list[str], folds: int = 5
) -> tuple[Any, float]:
    """Fit a meta-classifier on both models' probability vectors, chosen by CV.

    **Why this and not the weighted average above.** A per-class weight can only say
    "trust text 0.62, image 0.38 for `news`". It cannot say "when the image model calls
    this `adult` at high confidence *and* the text model says `shopping`, go with adult" --
    an interaction between the two vectors, which is exactly the kind of thing a screenshot
    is useful for. Stacking the concatenated distributions can represent that.

    Cross-validated because the fitting set is small (1,704 paired domains against 94
    features), so a single split would neither use the data well nor estimate honestly.
    The reported score is out-of-fold, and the returned model is refitted on everything.

    Args:
        fit_text: Calibrated text probabilities, ``(n, classes)``.
        fit_image: Calibrated image probabilities, ``(n, classes)``.
        fit_targets: Gold class indices.
        labels: Ordered class names.
        folds: Cross-validation folds.

    Returns:
        tuple[Any, float]: The refitted stacker and its out-of-fold macro-F1.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    # Multinomial is the default and the `multi_class` argument was removed in
    # scikit-learn 1.7, so passing it is now an error rather than a clarification.
    features = np.hstack([fit_text.numpy(), fit_image.numpy()])
    targets = fit_targets.numpy()

    # Stratified so rare classes appear in every fold; several have under 20 examples.
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    out_of_fold = np.zeros_like(targets)
    for train_idx, test_idx in splitter.split(features, targets):
        fold = LogisticRegression(max_iter=2000, C=1.0)
        fold.fit(features[train_idx], targets[train_idx])
        out_of_fold[test_idx] = fold.predict(features[test_idx])

    truth = [labels[int(t)] for t in targets]
    predicted = [labels[int(p)] for p in out_of_fold]
    cv_f1 = macro_f1(truth, predicted)

    stacker = LogisticRegression(max_iter=2000, C=1.0)
    stacker.fit(features, targets)
    return stacker, cv_f1


def score_stacker(
    stacker: Any, text_p: Any, image_p: Any, targets: Any, labels: list[str]
) -> dict[str, Any]:
    """Score the stacker on a held-out split.

    Args:
        stacker: The fitted meta-classifier.
        text_p: Calibrated text probabilities.
        image_p: Calibrated image probabilities.
        targets: Gold class indices.
        labels: Ordered class names.

    Returns:
        dict[str, Any]: Accuracy, macro-F1 and the per-class report.
    """
    import numpy as np

    predicted = stacker.predict(np.hstack([text_p.numpy(), image_p.numpy()]))
    truth_names = [labels[int(t)] for t in targets.numpy()]
    pred_names = [labels[int(p)] for p in predicted]
    accuracy = sum(t == p for t, p in zip(truth_names, pred_names, strict=True)) / max(
        1, len(truth_names)
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1(truth_names, pred_names),
        "per_class": per_class_report(truth_names, pred_names),
    }


def score(probabilities: Any, targets: Any, labels: list[str]) -> dict[str, Any]:
    """Turn a probability matrix into accuracy and macro-F1.

    Args:
        probabilities: A ``(n, classes)`` tensor.
        targets: Gold class indices.
        labels: Ordered class names.

    Returns:
        dict[str, Any]: Accuracy, macro-F1 and the per-class report.
    """
    predicted = probabilities.argmax(dim=-1)
    truth_names = [labels[int(t)] for t in targets.tolist()]
    pred_names = [labels[int(p)] for p in predicted.tolist()]
    accuracy = sum(t == p for t, p in zip(truth_names, pred_names, strict=True)) / max(
        1, len(truth_names)
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1(truth_names, pred_names),
        "per_class": per_class_report(truth_names, pred_names),
    }


def refuse_if_leaky(text_data: Path, image_data: Path) -> None:
    """Refuse to fuse when the image model trained on the domains fusion scores.

    **This is the bug that made the first fused number look like a clear win.**
    ``prepare_text.py`` and ``prepare_images.py`` each shuffled *their own* list with the
    same seed, and the lists differ -- 46,754 documents against 44,712 screenshots -- so a
    domain landed in unrelated splits on each side. Around 80% of the domains fusion fits
    and scores on were in the image model's training set, and image-only read 0.768 where
    the honest figure was 0.429.

    ``prepare_images.py --respect-splits`` fixes it, but a flag you can forget is not a
    fix for a failure that produces a plausible wrong number rather than an error. So this
    checks rather than trusts.

    Args:
        text_data: A ``prepare_text.py`` output directory.
        image_data: A ``prepare_images.py`` output directory.

    Raises:
        SystemExit: If any domain held out on the text side is in the image training
            split, naming the count and the remedy.
    """
    train_path = image_data / "train.jsonl"
    if not train_path.exists():
        print(
            f"WARNING: no {train_path}, so the image splits cannot be checked against "
            "the text splits. If the image model trained on the domains scored below, "
            "every number in this report is inflated.",
            file=sys.stderr,
        )
        return

    image_train = {
        json.loads(line)["domain"].lower()
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    held_out = set()
    for split in ("val", "test"):
        path = text_data / f"{split}.jsonl"
        if path.exists():
            held_out |= {
                json.loads(line)["domain"].lower()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

    leaked = held_out & image_train
    if leaked:
        raise SystemExit(
            f"{len(leaked):,} domains held out from the text model are in the image "
            f"model's training split (e.g. {', '.join(sorted(leaked)[:3])}).\n"
            "Fusion would be fitted and scored on data the image model memorised. "
            "Re-run prepare_images.py with --respect-splits <text-data>."
        )
    print(f"splits are aligned: no overlap across {len(held_out):,} held-out domains")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Fit and evaluate late fusion")
    parser.add_argument("--text", required=True, help="train_text.py output directory")
    parser.add_argument(
        "--image", required=True, help="train_image.py output directory"
    )
    parser.add_argument("--text-data", required=True, help="prepare_text.py output")
    parser.add_argument("--image-data", required=True, help="prepare_images.py output")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="", help="Write the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Fit fusion on paired domains and report all three models.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status. Non-zero when fusion fails to beat text alone,
        so a pipeline can gate on it.

    Raises:
        SystemExit: If the two label spaces cannot be reconciled, or no domain has both a
            page and a screenshot.
    """
    args = build_parser().parse_args(argv)

    import torch

    text_dir, image_dir = Path(args.text), Path(args.image)
    labels = json.loads((text_dir / "labels.json").read_text(encoding="utf-8"))
    image_labels = json.loads((image_dir / "labels.json").read_text(encoding="utf-8"))
    # A different *order* silently permutes every class, so it is still fatal. A smaller
    # image label space is not: it means some classes had too few screenshots to train on,
    # and those columns can be projected to zero instead of discarding the model.
    extra = sorted(set(image_labels) - set(labels))
    if extra:
        raise SystemExit(
            f"image model emits classes the text model does not know: {extra}"
        )
    ordered = [name for name in labels if name in set(image_labels)]
    if ordered != image_labels:
        raise SystemExit(
            "text and image label orders disagree; fusing would permute every class"
        )
    absent = [i for i, name in enumerate(labels) if name not in set(image_labels)]
    if absent:
        print(
            f"image model covers {len(image_labels)}/{len(labels)} classes; "
            f"no screenshots for {[labels[i] for i in absent]}"
        )

    text_data, image_data = Path(args.text_data), Path(args.image_data)
    available = {p.stem for p in (image_data / "images").glob("*.jpg")}
    refuse_if_leaky(text_data, image_data)

    def paired(split: str) -> list[dict[str, Any]]:
        rows = read_jsonl(text_data / f"{split}.jsonl")
        return [r for r in rows if r["domain"].lower() in available]

    fit_rows, test_rows = paired("val"), paired("test")
    print(f"paired domains: {len(fit_rows):,} to fit, {len(test_rows):,} to score")
    if not fit_rows or not test_rows:
        raise SystemExit("no domains have both a page and a screenshot")

    index = {name: i for i, name in enumerate(labels)}
    results: dict[str, Any] = {}

    for name, rows in (("fit", fit_rows), ("test", test_rows)):
        print(f"scoring the {name} split...")
        text_p = text_probabilities(text_dir, rows, labels, args.batch_size)
        image_p = image_probabilities(
            image_dir,
            rows,
            image_data / "images",
            labels,
            image_labels,
            args.batch_size,
        )
        targets = torch.tensor([index[r["category"]] for r in rows])
        results[name] = (text_p, image_p, targets)

    fit_text, fit_image, fit_targets = results["fit"]
    scalar = fit_weight(fit_text, fit_image, fit_targets, per_class=False)
    per_class = fit_weight(fit_text, fit_image, fit_targets, per_class=True)

    # Classes the image model cannot emit carry zero image mass, so blending them at any
    # weight below 1 shrinks them against every other class and invents misses the text
    # model would not have made. LBFGS will not discover this on its own: if no fitting
    # example carries the class, its gradient is zero and the weight stays at
    # sigmoid(0) = 0.5 -- halving it. Pin those to text-only.
    for i in absent:
        per_class[i] = 1.0

    # Pick the form on the *fit* split, before the test split is scored. Choosing by test
    # macro-F1 -- which this used to do -- makes the test set part of model selection, so
    # the number reported and the fusion.json published are both optimistically biased.
    # The choice is small (the two forms differ by ~0.001 here) but the principle is the
    # same one that made the split leak matter: held-out data must not steer a decision.
    fit_scalar = score(
        scalar * fit_text + (1 - scalar) * fit_image, fit_targets, labels
    )
    fit_per_class = score(
        per_class * fit_text + (1 - per_class) * fit_image, fit_targets, labels
    )
    better_per_class = fit_per_class["macro_f1"] >= fit_scalar["macro_f1"]
    print(
        f"fusion form chosen on the fit split: "
        f"{'per-class' if better_per_class else 'scalar'} "
        f"({fit_per_class['macro_f1']:.4f} vs {fit_scalar['macro_f1']:.4f} macro-F1)"
    )

    # A stacker over both probability vectors, cross-validated. It can represent
    # interactions a per-class weight cannot -- "image says adult confidently while text
    # says shopping" -- which is the shape of case a screenshot should win.
    stacker, stacker_cv_f1 = fit_stacker(fit_text, fit_image, fit_targets, labels)
    print(f"stacker: {stacker_cv_f1:.4f} macro-F1 out-of-fold on the fit split")

    test_text, test_image, test_targets = results["test"]
    report = {
        "text_only": score(test_text, test_targets, labels),
        "image_only": score(test_image, test_targets, labels),
        "fused_scalar": score(
            scalar * test_text + (1 - scalar) * test_image, test_targets, labels
        ),
        "fused_per_class": score(
            per_class * test_text + (1 - per_class) * test_image, test_targets, labels
        ),
        "fused_stacked": score_stacker(
            stacker, test_text, test_image, test_targets, labels
        ),
        "stacker_cv_macro_f1": stacker_cv_f1,
        "scalar_text_weight": float(scalar.item()),
        "paired_fit": len(fit_rows),
        "paired_test": len(test_rows),
    }

    print(f"\n{'model':20s} {'accuracy':>9s} {'macro-F1':>9s}")
    for key in (
        "text_only",
        "image_only",
        "fused_scalar",
        "fused_per_class",
        "fused_stacked",
    ):
        r = report[key]
        print(f"{key:20s} {r['accuracy']:9.3f} {r['macro_f1']:9.3f}")
    print(
        f"\nfitted text weight: {scalar.item():.3f} "
        f"({1 - scalar.item():.3f} to the image model)"
    )

    best_fused = max(
        report["fused_scalar"]["macro_f1"],
        report["fused_per_class"]["macro_f1"],
        report["fused_stacked"]["macro_f1"],
    )
    text_f1 = report["text_only"]["macro_f1"]
    helps = best_fused > text_f1
    report["fusion_helps"] = helps

    print()
    if helps:
        print(f"Fusion beats text alone: {text_f1:.3f} -> {best_fused:.3f} macro-F1.")
    else:
        print(
            f"Fusion does NOT beat text alone ({best_fused:.3f} vs {text_f1:.3f}).\n"
            "Ship image classification opt-in and put this number in the README."
        )

    # The weights are an artifact, not just a number in a log. Write them next to the
    # image model so inference can load them; without this file `combined` refuses to
    # fuse rather than falling back to a guessed 0.5.
    chosen = per_class if better_per_class else scalar
    (image_dir / "fusion.json").write_text(
        json.dumps(
            {
                "text_weights": [float(w) for w in chosen.tolist()],
                "labels": labels,
                "kind": "per_class" if better_per_class else "scalar",
                "fitted_on": len(fit_rows),
                "beats_text_only": helps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {image_dir}/fusion.json "
        f"({'per-class' if better_per_class else 'scalar'} weights)"
    )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if helps else 1


if __name__ == "__main__":
    raise SystemExit(main())
