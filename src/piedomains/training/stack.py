#!/usr/bin/env python3
r"""Find the best way to combine two probability vectors into one label.

**The framing.** Late fusion is a small supervised problem: two 47-dimensional probability
vectors in, one label out, ~1,700 examples to fit on. So it should be attacked the way any
such problem is -- try several model families, select by cross-validation, report on a
held-out split -- rather than by assuming a functional form.

**What was wrong with the first attempt.** It fitted a per-class *weight*, which can only
say "trust text 0.62, image 0.38 for `news`". Then a `LogisticRegression` over the
concatenated probabilities, which is linear and therefore still cannot represent "the image
model says `adult` confidently **and** the text model says `shopping`". That interaction is
the entire reason a screenshot might beat the text on a page -- and both of those models
are blind to it by construction. This tries models that are not.

**Why this is cheap.** `fuse.py` writes ``probabilities.npz``: both matrices and the
targets, 1.3 MB. Computing them needs the paired screenshots and an hour of GPU; fitting a
combiner on them needs neither. Every experiment here runs in seconds.

**The honest risk.** 1,704 fitting examples across 47 classes is ~36 per class, and a
gradient booster will happily memorise that. Cross-validation is exactly what catches it:
if the nonlinear models do not beat the linear one *out of fold*, that is the answer and it
gets reported as the answer.

Usage::

    uv run python -m piedomains.training.stack --probabilities reports/probabilities.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .metrics import accuracy, macro_f1, per_class_report

#: Classes where per-class weighting gave the screenshot real weight. Reported separately
#: because an aggregate over 47 classes averages away an effect concentrated in four.
VISUAL_CLASSES: tuple[str, ...] = ("adult", "news", "socialnet", "games-online")

#: Minimum macro-F1 gain over text-only that counts as evidence rather than noise, at
#: n=1,742 where the standard error on accuracy is about 0.011.
REQUIRED_GAIN = 0.02


def build_features(text_p: Any, image_p: Any) -> Any:
    """Turn two probability vectors into a feature matrix.

    The raw 94 probabilities, plus the summaries a person would actually look at when
    deciding which model to believe: how confident each one is, how spread its
    distribution is, and whether they agree at all.

    Args:
        text_p: ``(n, classes)`` text probabilities.
        image_p: ``(n, classes)`` image probabilities.

    Returns:
        Any: ``(n, 2 * classes + 5)`` feature matrix.
    """
    import numpy as np

    eps = 1e-12
    text_max = text_p.max(axis=1, keepdims=True)
    image_max = image_p.max(axis=1, keepdims=True)
    text_entropy = -(text_p * np.log(text_p + eps)).sum(axis=1, keepdims=True)
    image_entropy = -(image_p * np.log(image_p + eps)).sum(axis=1, keepdims=True)
    agree = (
        (text_p.argmax(axis=1) == image_p.argmax(axis=1)).astype(float).reshape(-1, 1)
    )

    return np.hstack(
        [text_p, image_p, text_max, image_max, text_entropy, image_entropy, agree]
    )


def candidates() -> dict[str, Any]:
    """The model families worth trying on a problem this size.

    Returns:
        dict[str, Any]: Name to unfitted estimator.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    return {
        # The linear baseline: what the first stacker was. Kept so the comparison is
        # explicit rather than assumed.
        "logistic": LogisticRegression(max_iter=2000, C=1.0),
        # Interactions natively, which is the whole point.
        "boosted": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0
        ),
        # A smooth nonlinear alternative; different inductive bias from trees.
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128,), max_iter=600, alpha=1e-3, random_state=42
        ),
    }


def cross_validate(model: Any, features: Any, targets: Any, labels: list[str]) -> float:
    """Score a candidate by out-of-fold macro-F1.

    Stratified because several classes have under 20 examples and an unstratified fold
    could contain none of them.

    Args:
        model: An unfitted estimator.
        features: Feature matrix.
        targets: Gold class indices.
        labels: Ordered class names.

    Returns:
        float: Out-of-fold macro-F1.
    """
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out_of_fold = np.zeros_like(targets)
    for train_idx, test_idx in splitter.split(features, targets):
        # clone() so each fold starts unfitted; sklearn's stub returns a broad union.
        fold: Any = clone(model)
        fold.fit(features[train_idx], targets[train_idx])
        out_of_fold[test_idx] = fold.predict(features[test_idx])
    return macro_f1(
        [labels[int(t)] for t in targets], [labels[int(p)] for p in out_of_fold]
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Select a late-fusion combiner by CV")
    parser.add_argument(
        "--probabilities", required=True, help="probabilities.npz from fuse.py"
    )
    parser.add_argument("--out", default="", help="Write the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Sweep combiner families and report the best against text-only.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Zero only if the best combiner clears text-only macro-F1 by REQUIRED_GAIN.
    """
    args = build_parser().parse_args(argv)

    import numpy as np

    data = np.load(args.probabilities, allow_pickle=False)
    labels = [str(x) for x in data["labels"]]
    fit_x = build_features(data["fit_text"], data["fit_image"])
    test_x = build_features(data["test_text"], data["test_image"])
    fit_y, test_y = data["fit_targets"], data["test_targets"]
    print(
        f"fitting on {len(fit_y):,} paired domains, scoring on {len(test_y):,}; "
        f"{fit_x.shape[1]} features, {len(labels)} classes"
    )

    truth = [labels[int(t)] for t in test_y]
    text_pred = [labels[int(i)] for i in data["test_text"].argmax(axis=1)]
    image_pred = [labels[int(i)] for i in data["test_image"].argmax(axis=1)]
    text_f1 = macro_f1(truth, text_pred)
    print(f"\ntext only  macro-F1 {text_f1:.4f}")
    print(f"image only macro-F1 {macro_f1(truth, image_pred):.4f}")

    print(f"\n{'combiner':12s} {'CV macro-F1':>12s} {'test macro-F1':>14s}")
    results: dict[str, Any] = {}
    for name, model in candidates().items():
        cv_f1 = cross_validate(model, fit_x, fit_y, labels)
        model.fit(fit_x, fit_y)
        pred = [labels[int(p)] for p in model.predict(test_x)]
        test_f1 = macro_f1(truth, pred)
        test_accuracy = accuracy(truth, pred)
        results[name] = {
            "cv_macro_f1": cv_f1,
            "test_macro_f1": test_f1,
            "test_accuracy": test_accuracy,
            "per_class": per_class_report(truth, pred),
        }
        print(f"{name:12s} {cv_f1:12.4f} {test_f1:14.4f}")

    # Selection on cross-validated score, never on the test split -- picking the winner by
    # test macro-F1 would make the reported number optimistically biased, which is a
    # mistake this project has already made once in fuse.py.
    best = max(results, key=lambda k: results[k]["cv_macro_f1"])
    gain = results[best]["test_macro_f1"] - text_f1
    print(f"\nselected by CV: {best}")
    print(f"gain over text-only: {gain:+.4f} macro-F1 (need +{REQUIRED_GAIN})")

    print(f"\n{'class where images earned weight':34s} {'f1':>6s} {'n':>5s}")
    for name in VISUAL_CLASSES:
        row = results[best]["per_class"].get(name)
        if row:
            print(f"{name:34s} {row['f1']:6.3f} {int(row['support']):5d}")

    report = {
        "text_only_macro_f1": text_f1,
        "selected": best,
        "gain": gain,
        "required_gain": REQUIRED_GAIN,
        "clears_band": gain >= REQUIRED_GAIN,
        "candidates": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if gain >= REQUIRED_GAIN:
        print(f"\n{best} beats text by {gain:.4f} -- ship it.")
        return 0
    print(
        f"\nNo combiner clears the band (best {gain:+.4f} against +{REQUIRED_GAIN}).\n"
        "Report the number and ship text-only."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
