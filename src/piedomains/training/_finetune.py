"""Fine-tuning shared by the text and screenshot trainers.

Both trainers fit a Hugging Face classifier the same way: AdamW with linear warmup,
gradient accumulation and clipping, validation macro-F1 after every epoch, a checkpoint
only when that improves, early stopping, and a final score of the checkpoint on disk
against the held-out split. They differ only in the dataset, the preprocessor and the
model's input fields, which the trainers pass in.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .metrics import macro_f1, per_class_report


def pick_device() -> str:
    """Choose the fastest available torch device.

    Returns:
        str: ``cuda``, ``mps`` or ``cpu``.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: Path, written_by: str) -> list[dict[str, Any]]:
    """Read a JSONL split.

    Args:
        path: File to read.
        written_by: The script that writes it, named in the error.

    Returns:
        list[dict[str, Any]]: One record per line.

    Raises:
        SystemExit: If the file is missing, naming the step that makes it.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found -- run {written_by} first")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_split(
    model: Any,
    loader: Any,
    device: str,
    labels: list[str],
    input_keys: Sequence[str],
) -> tuple[float, list[str], list[str]]:
    """Score the model over a loader.

    Args:
        model: The classifier.
        loader: A DataLoader over a split.
        device: Torch device string.
        labels: Ordered class names.
        input_keys: Batch fields the model takes as keyword inputs.

    Returns:
        tuple: ``(macro_f1, truth, predicted)``.
    """
    import torch

    model.eval()
    truth: list[str] = []
    predicted: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(**{key: batch[key].to(device) for key in input_keys}).logits
            for gold, pred in zip(
                batch["labels"].tolist(), logits.argmax(-1).cpu().tolist(), strict=True
            ):
                truth.append(labels[gold])
                predicted.append(labels[pred])
    return macro_f1(truth, predicted), truth, predicted


def save_checkpoint(
    out: Path,
    model: Any,
    preprocessor: Any,
    labels: list[str],
    config: Any,
    state: dict,
) -> None:
    """Write weights, preprocessor, labels and run state.

    A checkpoint does not record which domains it trained on:
    :func:`piedomains.training.splits.split_of` derives the split from the domain, so
    anyone can recompute it.

    Args:
        out: Directory to write into.
        model: The classifier.
        preprocessor: Its tokenizer or image processor.
        labels: Ordered class names.
        config: The run configuration, a dataclass.
        state: Epoch/score bookkeeping for resuming.
    """
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    preprocessor.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (out / "config.json.piedomains").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    (out / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def train_and_evaluate(
    *,
    model: Any,
    preprocessor: Any,
    labels: list[str],
    config: Any,
    device: str,
    out: Path,
    loaders: tuple[Any, Any, Any],
    input_keys: Sequence[str],
    load_best: Callable[[Path, Any], Any],
    start_epoch: int = 0,
    best_f1: float = 0.0,
) -> int:
    """Fine-tune, keep the best epoch, and score it on the held-out split.

    Args:
        model: The classifier, already on ``device``.
        preprocessor: Its tokenizer or image processor, saved with each checkpoint.
        labels: Ordered class names.
        config: The run configuration: ``lr``, ``weight_decay``, ``warmup_ratio``,
            ``grad_accum``, ``epochs`` and ``patience``.
        device: Torch device string.
        out: Directory checkpoints and ``test_metrics.json`` are written to.
        loaders: The train, validation and test DataLoaders.
        input_keys: Batch fields the model takes as keyword inputs.
        load_best: Reloads the saved checkpoint, given ``out`` and the in-memory
            model to fall back on.
        start_epoch: First epoch to run, when resuming.
        best_f1: Validation macro-F1 to beat, when resuming.

    Returns:
        int: Process exit status.
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    train_loader, val_loader, test_loader = loaders
    stale = 0
    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum)
    total_steps = steps_per_epoch * max(1, config.epochs - start_epoch)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )

    for epoch in range(start_epoch, config.epochs):
        model.train()
        started = time.time()
        running = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            output = model(
                **{key: batch[key].to(device) for key in input_keys},
                labels=batch["labels"].to(device),
            )
            (output.loss / config.grad_accum).backward()
            running += output.loss.item()

            if (step + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # MPS does not release cached blocks aggressively; on a small unified-memory
            # machine that accumulation is enough to wedge the process in swap.
            if step % 50 == 0 and device == "mps":
                torch.mps.empty_cache()

            if step % 100 == 0:
                done = step + 1
                print(
                    f"  epoch {epoch + 1} step {done}/{len(train_loader)} "
                    f"loss {running / done:.4f}",
                    flush=True,
                )

        val_f1, _, _ = evaluate_split(model, val_loader, device, labels, input_keys)
        mins = (time.time() - started) / 60
        print(
            f"epoch {epoch + 1}: loss {running / len(train_loader):.4f}, "
            f"val macro-F1 {val_f1:.4f} ({mins:.1f} min)"
        )

        improved = val_f1 > best_f1
        if improved:
            best_f1, stale = val_f1, 0
        else:
            stale += 1

        # Only overwrite on an improvement. Saving unconditionally leaves the *last*
        # epoch on disk rather than the best, which is silently wrong once the model
        # starts overfitting -- and it does: the first text run peaked at epoch 3 and
        # fell at epoch 4.
        if improved:
            save_checkpoint(
                out,
                model,
                preprocessor,
                labels,
                config,
                {"epoch": epoch + 1, "best_f1": best_f1, "last_val_f1": val_f1},
            )
            print(f"  checkpoint saved to {out} (new best)")
        else:
            print(f"  not saved: {val_f1:.4f} does not beat {best_f1:.4f}")

        if stale >= config.patience:
            print(f"no improvement in {stale} epochs; stopping")
            break

    # Score the checkpoint on disk, not the model in memory. With best-epoch-only
    # checkpointing, `model` holds the *last* epoch's weights whenever validation peaked
    # earlier or early stopping fired -- so test_metrics.json described a model nobody
    # would ever load. The published image model was saved at epoch 4 and reported epoch
    # 5's score.
    best = load_best(out, model)
    test_f1, truth, predicted = evaluate_split(
        best, test_loader, device, labels, input_keys
    )
    report = per_class_report(truth, predicted)
    accuracy = sum(t == p for t, p in zip(truth, predicted, strict=True)) / max(
        1, len(truth)
    )

    print(f"\nheld-out test: accuracy {accuracy:.4f}, macro-F1 {test_f1:.4f}")
    print(f"{'category':22s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}")
    for name, row in sorted(report.items(), key=lambda kv: -kv[1]["support"]):
        print(
            f"{name:22s} {row['precision']:6.3f} {row['recall']:6.3f} "
            f"{row['f1']:6.3f} {int(row['support']):6d}"
        )

    (out / "test_metrics.json").write_text(
        json.dumps(
            {"accuracy": accuracy, "macro_f1": test_f1, "per_class": report}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}/test_metrics.json")
    return 0
