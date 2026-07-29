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

sys.path.insert(0, str(Path(__file__).parent))

from metrics import macro_f1, per_class_report
from train_text import pick_device, read_jsonl


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
    from train_text import TextDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    temperature = load_calibrated(model_dir)

    loader = DataLoader(
        TextDataset(rows, tokenizer, labels, 128), batch_size=batch_size, shuffle=False
    )
    chunks = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            chunks.append(torch.softmax(logits.cpu() / temperature, dim=-1))
    return torch.cat(chunks)


def image_probabilities(
    model_dir: Path,
    rows: list[dict[str, Any]],
    image_dir: Path,
    labels: list[str],
    batch_size: int,
) -> Any:
    """Score domains with the image model, temperature applied.

    Args:
        model_dir: Image checkpoint directory.
        rows: Records carrying ``domain``.
        image_dir: Directory of ``<domain>.jpg``.
        labels: Ordered class names.
        batch_size: Inference batch size.

    Returns:
        Any: A ``(n, classes)`` tensor of calibrated probabilities.
    """
    import torch
    from torch.utils.data import DataLoader
    from train_image import ScreenshotDataset
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = pick_device()
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
    model.eval()
    temperature = load_calibrated(model_dir)

    loader = DataLoader(
        ScreenshotDataset(rows, image_dir, labels, processor),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks = []
    with torch.no_grad():
        for batch in loader:
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            chunks.append(torch.softmax(logits.cpu() / temperature, dim=-1))
    return torch.cat(chunks)


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
    """
    args = build_parser().parse_args(argv)

    import torch

    text_dir, image_dir = Path(args.text), Path(args.image)
    labels = json.loads((text_dir / "labels.json").read_text(encoding="utf-8"))
    image_labels = json.loads((image_dir / "labels.json").read_text(encoding="utf-8"))
    if labels != image_labels:
        # Fusing across different label orders silently permutes every class.
        raise SystemExit(
            "text and image models disagree on the label set; "
            f"{len(labels)} vs {len(image_labels)} classes"
        )

    text_data, image_data = Path(args.text_data), Path(args.image_data)
    available = {p.stem for p in (image_data / "images").glob("*.jpg")}

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
            image_dir, rows, image_data / "images", labels, args.batch_size
        )
        targets = torch.tensor([index[r["category"]] for r in rows])
        results[name] = (text_p, image_p, targets)

    fit_text, fit_image, fit_targets = results["fit"]
    scalar = fit_weight(fit_text, fit_image, fit_targets, per_class=False)
    per_class = fit_weight(fit_text, fit_image, fit_targets, per_class=True)

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
        "scalar_text_weight": float(scalar.item()),
        "paired_fit": len(fit_rows),
        "paired_test": len(test_rows),
    }

    print(f"\n{'model':20s} {'accuracy':>9s} {'macro-F1':>9s}")
    for key in ("text_only", "image_only", "fused_scalar", "fused_per_class"):
        r = report[key]
        print(f"{key:20s} {r['accuracy']:9.3f} {r['macro_f1']:9.3f}")
    print(
        f"\nfitted text weight: {scalar.item():.3f} "
        f"({1 - scalar.item():.3f} to the image model)"
    )

    best_fused = max(
        report["fused_scalar"]["macro_f1"], report["fused_per_class"]["macro_f1"]
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

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if helps else 1


if __name__ == "__main__":
    raise SystemExit(main())
