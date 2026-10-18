#!/usr/bin/env python3
"""Fine-tune a vision backbone to classify a website from its homepage screenshot.

Replaces a ResNet50 that reported 52.9% at training time and, in production, labeled
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
layout, color and gross structure, not text. That is a real ceiling, and if image-only
macro-F1 is poor the first lever is resolution (``--size 384``, roughly 3x the compute),
not more epochs.

The reported number that matters is macro-F1: the class distribution is long-tailed and
accuracy flatters a model that only learns `adult` and `shopping`.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._finetune import pick_device, read_jsonl, train_and_evaluate

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


class ScreenshotDataset:
    """Screenshots paired with labels, decoded on demand.

    Map-style by protocol -- it defines ``__len__`` and ``__getitem__``, which is all
    ``DataLoader`` requires at runtime -- rather than subclassing
    ``torch.utils.data.Dataset``, which would force a module-level ``torch`` import into a
    file that is also imported just to print ``--help``. torch's stub is stricter than its
    behavior, so the call sites carry a narrow ignore.

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

        # Horizontal flip only. Rotation and heavy color jitter are wrong here: page
        # layout is not rotation-invariant, and color scheme is signal.
        if self.train and random.random() < 0.5:  # noqa: S311 -- augmentation
            from PIL import ImageOps

            img = ImageOps.mirror(img)

        encoded = self.processor(images=img, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.tensor(self.index[row["category"]], dtype=torch.long),
        }

    # Record what this model actually trained on. Checking the *split files* is not
    # enough: a checkpoint outlives the splits it was fitted against, and when the text
    # corpus was re-prepared, 73% of the new test domains turned out to be in the old
    # training set. The split files agreed with each other perfectly -- both had been
    # rebuilt from the same hash -- while the model had seen three quarters of the
    # data it was about to be scored on. Only the model can answer what the model saw.


def load_best(out: Path, fallback: Any) -> Any:
    """Load the saved checkpoint, so metrics describe the artifact.

    Args:
        out: Directory the best epoch was written to.
        fallback: The in-memory model, used when nothing was ever saved.

    Returns:
        Any: The model to evaluate, on the same device as ``fallback``.
    """
    from transformers import AutoModelForImageClassification

    if not (out / "config.json").exists():
        print("no checkpoint on disk; scoring the in-memory model")
        return fallback
    device = next(fallback.parameters()).device
    best = AutoModelForImageClassification.from_pretrained(out).to(device)
    best.eval()
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    print(
        f"scoring the saved checkpoint (epoch {state['epoch']}, "
        f"val macro-F1 {state['best_f1']:.4f})"
    )
    return best


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
    splits = {
        n: read_jsonl(data / f"{n}.jsonl", "prepare_images.py")
        for n in ("train", "val", "test")
    }
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

    start_epoch, best_f1 = 0, 0.0
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

    return train_and_evaluate(
        model=model,
        preprocessor=processor,
        labels=labels,
        config=config,
        device=device,
        out=out,
        loaders=(train_loader, val_loader, test_loader),
        input_keys=("pixel_values",),
        load_best=load_best,
        start_epoch=start_epoch,
        best_f1=best_f1,
    )


if __name__ == "__main__":
    raise SystemExit(main())
