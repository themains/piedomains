#!/usr/bin/env python3
"""Fine-tune a vision backbone to classify a website from its homepage screenshot.

Replaces a ResNet50 that reported 52.9% at training time and, in production, labelled
Khan Academy and Yahoo as `porn`. Two things were wrong with it, and both are fixed here
rather than tuned around:

* **The backbone was frozen.** ``base_model.trainable = False`` meant only a linear head
  was ever fitted, on ImageNet features that know about dogs and cars and nothing about
  page layout. This fine-tunes the whole network.
* **The serving path divided pixels by 255** before handing them to a graph that already
  baked in ``resnet50.preprocess_input``, so every image arrived as a near-constant
  negative array. Preprocessing here is the model's own, applied identically in training
  and inference.

Usage::

    uv run --group train python training/train_image.py \
        --data data/images-224 --out models/image-v1

**What a screenshot can and cannot say.** At 224px a page is unreadable — the model sees
layout, colour and gross structure, not text. That is a real ceiling, and if image-only
macro-F1 is poor the first lever is resolution (``--size 384``, roughly 3x the compute),
not more epochs.

The reported number that matters is macro-F1: the class distribution is long-tailed and
accuracy flatters a model that only learns `adult` and `shopping`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .metrics import macro_f1, per_class_report

#: ImageNet-pretrained ViT. Base rather than large because the bottleneck here is label
#: quality and 224px resolution, not model capacity.
DEFAULT_MODEL = "google/vit-base-patch16-224-in21k"


@dataclass
class TrainConfig:
    """Everything that determines the run, recorded alongside the weights."""

    model_name: str = DEFAULT_MODEL
    image_size: int = 224
    batch_size: int = 32
    grad_accum: int = 1
    epochs: int = 4
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    #: Stop when val macro-F1 has not improved for this many epochs.
    patience: int = 2


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a split written by ``prepare_images.py``.

    Args:
        path: File to read.

    Returns:
        list[dict[str, Any]]: One record per line.

    Raises:
        SystemExit: If the file is missing, naming the step that makes it.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found -- run prepare_images.py first")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ScreenshotDataset:
    """Screenshots paired with labels, decoded on demand.

    Map-style by protocol -- it defines ``__len__`` and ``__getitem__``, which is all
    ``DataLoader`` requires at runtime -- rather than subclassing
    ``torch.utils.data.Dataset``, which would force a module-level ``torch`` import into a
    file that is also imported just to print ``--help``. torch's stub is stricter than its
    behaviour, so the call sites carry a narrow ignore.

    Decoding in ``__getitem__`` rather than up front keeps memory flat: the resized
    corpus is ~5 GB on disk and would not fit comfortably in RAM as tensors.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        image_dir: Path,
        labels: list[str],
        processor: Any,
        *,
        train: bool = False,
    ):
        """Bind rows to an image directory and a preprocessor.

        Args:
            rows: Records with ``domain`` and ``category``.
            image_dir: Directory of ``<domain>.jpg`` files.
            labels: Ordered class names; index is the class id.
            processor: The model's own image processor.
            train: Whether to apply training-time augmentation.
        """
        self.rows = rows
        self.image_dir = image_dir
        self.index = {name: i for i, name in enumerate(labels)}
        self.processor = processor
        self.train = train

    def __len__(self) -> int:
        """Number of examples.

        Returns:
            int: Row count.
        """
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        """Load and preprocess one screenshot.

        Args:
            i: Row index.

        Returns:
            dict: ``pixel_values`` and ``labels`` tensors.
        """
        import torch
        from PIL import Image

        row = self.rows[i]
        path = self.image_dir / f"{row['domain']}.jpg"
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # A missing or corrupt file becomes a black frame rather than killing the
            # epoch. prepare_images.py already filters these; this is belt and braces.
            img = Image.new("RGB", (self.processor.size["height"],) * 2)

        # Horizontal flip only. Rotation and heavy colour jitter are wrong here: page
        # layout is not rotation-invariant, and colour scheme is signal.
        if self.train and random.random() < 0.5:  # noqa: S311 -- augmentation
            from PIL import ImageOps

            img = ImageOps.mirror(img)

        encoded = self.processor(images=img, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.tensor(self.index[row["category"]], dtype=torch.long),
        }


def evaluate_split(
    model: Any, loader: Any, device: str, labels: list[str]
) -> tuple[float, list[str], list[str]]:
    """Score the model over a loader.

    Args:
        model: The classifier.
        loader: A DataLoader over a split.
        device: Torch device string.
        labels: Ordered class names.

    Returns:
        tuple: ``(macro_f1, truth, predicted)``.
    """
    import torch

    model.eval()
    truth: list[str] = []
    predicted: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            for gold, pred in zip(
                batch["labels"].tolist(), logits.argmax(-1).cpu().tolist(), strict=True
            ):
                truth.append(labels[gold])
                predicted.append(labels[pred])
    return macro_f1(truth, predicted), truth, predicted


def save_checkpoint(
    out: Path,
    model: Any,
    processor: Any,
    labels: list[str],
    config: TrainConfig,
    state: dict,
) -> None:
    """Write weights, preprocessor, labels and run state.

    Args:
        out: Directory to write into.
        model: The classifier.
        processor: Its image processor.
        labels: Ordered class names.
        config: The run configuration.
        state: Epoch/score bookkeeping for resuming.
    """
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (out / "config.json.piedomains").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    (out / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a screenshot classifier")
    parser.add_argument("--data", required=True, help="prepare_images.py output")
    parser.add_argument("--out", required=True, help="Where to write the model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base vision backbone")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Train on N examples only, to check the loop",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue from the checkpoint in --out"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Fine-tune the backbone and report held-out macro-F1.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status.
    """
    args = build_parser().parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        get_linear_schedule_with_warmup,
    )

    config = TrainConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    torch.manual_seed(config.seed)
    # Python's RNG too, not just torch's. ScreenshotDataset's horizontal flip calls
    # random.random(), which torch.manual_seed does not touch -- so two runs with the same
    # --seed saw different augmentation and diverged. Two SigLIP2 runs that should have
    # shared their first three epochs peaked at val macro-F1 0.3953 and 0.3705, which was
    # large enough to be mistaken for an effect of epoch count.
    random.seed(config.seed)

    data = Path(args.data)
    out = Path(args.out)
    labels = json.loads((data / "labels.json").read_text(encoding="utf-8"))
    splits = {n: read_jsonl(data / f"{n}.jsonl") for n in ("train", "val", "test")}
    if args.limit:
        splits["train"] = splits["train"][: args.limit]
        for n in ("val", "test"):
            splits[n] = splits[n][: max(1, args.limit // 8)]

    device = pick_device()
    print(f"device: {device}")
    print(f"classes: {len(labels)}")
    print(f"train/val/test: {'/'.join(str(len(splits[n])) for n in splits)}")

    source = (
        str(out)
        if (args.resume and (out / "state.json").exists())
        else config.model_name
    )
    processor = AutoImageProcessor.from_pretrained(source)
    model = AutoModelForImageClassification.from_pretrained(
        source,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
        ignore_mismatched_sizes=True,
    ).to(device)

    start_epoch, best_f1, stale = 0, 0.0, 0
    if args.resume and (out / "state.json").exists():
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        start_epoch, best_f1 = state.get("epoch", 0), state.get("best_f1", 0.0)
        print(f"resuming from epoch {start_epoch} (best macro-F1 {best_f1:.4f})")

    image_dir = data / "images"

    def loader(name: str, shuffle: bool) -> Any:
        return DataLoader(
            ScreenshotDataset(  # pyright: ignore[reportArgumentType]
                splits[name], image_dir, labels, processor, train=shuffle
            ),
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=(device == "cuda"),
        )

    train_loader = loader("train", True)
    val_loader = loader("val", False)
    test_loader = loader("test", False)

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
                pixel_values=batch["pixel_values"].to(device),
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

        val_f1, _, _ = evaluate_split(model, val_loader, device, labels)
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
        # starts overfitting -- and it does.
        if improved:
            save_checkpoint(
                out,
                model,
                processor,
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

    test_f1, truth, predicted = evaluate_split(model, test_loader, device, labels)
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


if __name__ == "__main__":
    raise SystemExit(main())
