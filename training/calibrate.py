#!/usr/bin/env python
r"""Fit a temperature scalar so reported confidence means something.

The shipped model calibrates with 39 per-class ``IsotonicRegression`` pickles
applied elementwise and never renormalized, which has three problems, all of
them live:

1. **They do not work at all.** Every one of the 39 unpickles under
   scikit-learn 1.9 and then predicts ``NaN``, so all are discarded at load and
   reported confidences are raw model outputs. The calibration layer users
   were told about has not run for some time.
2. **Elementwise isotonic does not produce a distribution.** Mapping each class
   score independently and not renormalizing means the vector does not sum to
   1, so ``confidence`` is not a probability and ``argmax`` is not the argmax
   of any distribution.
3. **Pickled estimators are a compatibility surface.** They are unpinned joblib
   pickles against ``scikit-learn>=1.3``, which is exactly how (1) happened.

Temperature scaling replaces all of it with one learned scalar ``T``, applied
as ``softmax(logits / T)``. It always yields a proper normalized distribution,
never changes which class wins (so accuracy is untouched by construction), and
serializes as a single float in a JSON file. It is the standard answer -- Guo
et al., *On Calibration of Modern Neural Networks*, ICML 2017.

Usage::

    uv run --group train python training/calibrate.py \
        --model models/text-v2 --data data/prepared

Writes ``calibration.json`` next to the weights and reports ECE before and
after, so the improvement is a measured number rather than an assumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from metrics import expected_calibration_error
from train_text import TextDataset, pick_device, read_jsonl


def collect_logits(model, loader, device: str) -> tuple[Any, Any]:
    """Run the model over a split and keep the raw logits.

    Args:
        model: The classifier.
        loader: DataLoader over the split.
        device: Torch device string.

    Returns:
        tuple: ``(logits, labels)`` as CPU tensors.
    """
    import torch

    model.eval()
    chunks, golds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            chunks.append(logits.cpu())
            golds.append(batch["labels"])
    return torch.cat(chunks), torch.cat(golds)


def fit_temperature(logits, targets, *, max_iter: int = 200) -> float:
    """Find the temperature minimizing NLL on held-out logits.

    A single scalar, optimized with LBFGS. Fitting on the validation split
    rather than the training split is the whole point: temperature learned on
    data the model has already memorized would come out near 1 and calibrate
    nothing.

    Args:
        logits: Model logits, shape ``(n, classes)``.
        targets: Gold class indices, shape ``(n,)``.
        max_iter: LBFGS iteration cap.

    Returns:
        float: The fitted temperature.
    """
    import torch

    log_t = torch.zeros(1, requires_grad=True)  # optimize log T to keep T > 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)  # pyright: ignore[reportArgumentType]
    return float(log_t.exp().item())


def ece_of(logits, targets, labels: list[str], temperature: float) -> float:
    """Compute ECE at a given temperature.

    Args:
        logits: Model logits.
        targets: Gold class indices.
        labels: Ordered category names.
        temperature: Scaling to apply.

    Returns:
        float: Expected calibration error.
    """
    import torch

    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidence, predicted = probabilities.max(dim=-1)
    correct = [
        gold == pred
        for gold, pred in zip(targets.tolist(), predicted.tolist(), strict=True)
    ]
    return expected_calibration_error(correct, confidence.tolist())


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="train_text.py output directory")
    parser.add_argument(
        "--data", required=True, help="prepare_text.py output directory"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Fit and record the temperature.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status.
    """
    args = build_parser().parse_args(argv)

    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(args.model)
    data = Path(args.data)
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    def loader(split: str):
        return DataLoader(
            TextDataset(
                read_jsonl(data / f"{split}.jsonl"), tokenizer, labels, args.max_length
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )

    print("collecting validation logits...")
    val_logits, val_targets = collect_logits(model, loader("val"), device)
    temperature = fit_temperature(val_logits, val_targets)

    print("collecting test logits...")
    test_logits, test_targets = collect_logits(model, loader("test"), device)
    before = ece_of(test_logits, test_targets, labels, 1.0)
    after = ece_of(test_logits, test_targets, labels, temperature)

    print(f"\ntemperature: {temperature:.4f}")
    print(f"test ECE: {before:.4f} -> {after:.4f}")
    if after > before:
        print(
            "WARNING: calibration got worse on the test split. Report this rather "
            "than shipping the temperature."
        )

    (model_dir / "calibration.json").write_text(
        json.dumps(
            {
                "method": "temperature_scaling",
                "temperature": temperature,
                "ece_before": before,
                "ece_after": after,
                "fitted_on": "val",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {model_dir}/calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
