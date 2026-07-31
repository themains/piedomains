#!/usr/bin/env python3
r"""Train one classifier that reads the page and the screenshot together.

**Why this rather than late fusion.** Late fusion combined the two models' output
*probabilities* with fitted weights. It could reweight two finished opinions and nothing
else -- it cannot represent "when the screenshot looks like a parking page, distrust the
text", because by the time it runs both models have already committed. It bought +0.001
macro-F1 on 1,742 paired domains and is not worth shipping.

**But the screenshot is not useless, and the fitted weights say where.** Per-class late
fusion is exactly "trust whichever modality is better for this class", and it gave the
image model real weight on visually distinctive categories::

    adult               0.500      hobby/games-online  0.278
    news                0.380      drugs               0.210
    socialnet           0.338

The median weight was 1.000 though: 39 of 47 classes gave it nothing, so the gain on the
handful that benefit diluted to nothing overall. A joint head over *representations* can
concentrate what probability-level mixing cannot.

**Frozen towers first.** 13,633 paired training examples against ~200M parameters overfits
readily, and a head over frozen encoders trains in minutes rather than hours. Unfreeze at
low LR only if the frozen run shows the interaction is there at all.

**The bar, declared before the run.** At n=1,742 the standard error on accuracy is ≈0.011,
so a difference under ~0.02 is not evidence. This exits non-zero unless the joint model
clears text-only macro-F1 + 0.02. It also reports the subset where the image model earned
fusion weight, because an aggregate over 47 classes hides an effect concentrated in four.

Usage::

    uv run python -m piedomains.training.train_joint \\
        --text models/text-v5 --image models/image-v1 \\
        --data data/prepared-minimal --images data/images-224 --out models/joint-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..checkpoints import eager_config
from .metrics import macro_f1, per_class_report
from .train_text import pick_device, read_jsonl

#: Classes where per-class late fusion gave the screenshot model real weight. Reported
#: separately because an aggregate over 47 classes hides an effect concentrated in four.
VISUAL_CLASSES: tuple[str, ...] = (
    "adult",
    "news",
    "socialnet",
    "hobby/games-online",
)

#: Minimum macro-F1 gain over text-only that counts as evidence rather than noise.
REQUIRED_GAIN = 0.02


class PairedDataset:
    """A domain's page text and its screenshot, decoded together.

    Map-style by protocol -- ``__len__`` and ``__getitem__`` -- rather than subclassing
    ``torch.utils.data.Dataset``, which would force a module-level torch import into a
    file that is also imported to print ``--help``.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        image_dir: Path,
        labels: list[str],
        tokenizer: Any,
        processor: Any,
        max_length: int,
    ):
        """Bind paired rows to both preprocessors.

        Args:
            rows: Records carrying ``domain``, ``category`` and ``text``.
            image_dir: Directory of ``<domain>.jpg``.
            labels: Ordered class names.
            tokenizer: The text model's tokenizer.
            processor: The image model's processor.
            max_length: Text truncation length.
        """
        self.rows = rows
        self.image_dir = image_dir
        self.index = {name: i for i, name in enumerate(labels)}
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        """Number of paired examples.

        Returns:
            int: Row count.
        """
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        """Encode one domain's text and screenshot.

        Args:
            i: Row index.

        Returns:
            dict: ``input_ids``, ``attention_mask``, ``pixel_values`` and ``labels``.
        """
        import torch
        from PIL import Image

        row = self.rows[i]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        path = self.image_dir / f"{row['domain']}.jpg"
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.processor.size["height"],) * 2)
        pixels = self.processor(images=img, return_tensors="pt")["pixel_values"]

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": pixels.squeeze(0),
            "labels": torch.tensor(self.index[row["category"]], dtype=torch.long),
        }


def build_model(text_dir: Path, image_dir: Path, n_classes: int, freeze: bool) -> Any:
    """Assemble the two-tower classifier.

    Both towers keep their trained weights and lose their classification heads; a small
    MLP over the concatenated pooled representations produces the label.

    Args:
        text_dir: Trained text checkpoint.
        image_dir: Trained image checkpoint.
        n_classes: Number of output classes.
        freeze: Whether to freeze both towers and train only the head.

    Returns:
        Any: The assembled ``torch.nn.Module``.
    """
    import torch
    from torch import nn
    from transformers import AutoModel

    text_tower = AutoModel.from_pretrained(text_dir, config=eager_config(str(text_dir)))
    image_tower = AutoModel.from_pretrained(image_dir)
    if hasattr(image_tower, "vision_model"):
        image_tower = image_tower.vision_model

    if freeze:
        for tower in (text_tower, image_tower):
            for param in tower.parameters():
                param.requires_grad = False

    text_width = text_tower.config.hidden_size
    image_width = getattr(image_tower.config, "hidden_size", None) or 768

    class TwoTower(nn.Module):
        """Text and image encoders feeding one classification head."""

        def __init__(self):
            """Wire the towers to the head."""
            super().__init__()
            self.text = text_tower
            self.image = image_tower
            self.head = nn.Sequential(
                nn.Linear(text_width + image_width, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, n_classes),
            )

        def forward(
            self, input_ids: Any, attention_mask: Any, pixel_values: Any
        ) -> Any:
            """Encode both modalities and classify.

            Args:
                input_ids: Tokenised text.
                attention_mask: Text attention mask.
                pixel_values: Preprocessed screenshot.

            Returns:
                Any: Class logits.
            """
            text_out = self.text(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            # Mean-pool over real tokens only; padding would otherwise drag the
            # representation toward whatever the pad embedding happens to be.
            mask = attention_mask.unsqueeze(-1).float()
            text_vec = (text_out * mask).sum(1) / mask.sum(1).clamp_min(1e-9)

            image_out = self.image(pixel_values=pixel_values)
            image_vec = getattr(image_out, "pooler_output", None)
            if image_vec is None:
                image_vec = image_out.last_hidden_state.mean(dim=1)

            return self.head(torch.cat([text_vec, image_vec], dim=-1))

    return TwoTower()


def score(
    model: Any, loader: Any, device: str, labels: list[str]
) -> tuple[float, float, list[str], list[str]]:
    """Evaluate the joint model over a split.

    Args:
        model: The classifier.
        loader: A DataLoader over paired examples.
        device: Torch device string.
        labels: Ordered class names.

    Returns:
        tuple: ``(accuracy, macro_f1, truth, predicted)``.
    """
    import torch

    model.eval()
    truth: list[str] = []
    predicted: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device),
            )
            for gold, pred in zip(
                batch["labels"].tolist(), logits.argmax(-1).cpu().tolist(), strict=True
            ):
                truth.append(labels[gold])
                predicted.append(labels[pred])
    accuracy = sum(t == p for t, p in zip(truth, predicted, strict=True)) / max(
        1, len(truth)
    )
    return accuracy, macro_f1(truth, predicted), truth, predicted


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Train a joint text+image classifier")
    parser.add_argument("--text", required=True, help="Trained text checkpoint")
    parser.add_argument("--image", required=True, help="Trained image checkpoint")
    parser.add_argument("--data", required=True, help="prepare_text.py output")
    parser.add_argument("--images", required=True, help="prepare_images.py output")
    parser.add_argument("--out", required=True, help="Where to write the checkpoint")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Fine-tune both towers. Off by default: 13,633 paired examples against "
        "~200M parameters overfits readily.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train the joint model and report whether it earns its place.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Zero only if the joint model clears text-only macro-F1 by REQUIRED_GAIN.

    Raises:
        SystemExit: If no domain has both a page and a screenshot.
    """
    args = build_parser().parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoTokenizer

    text_dir, image_dir = Path(args.text), Path(args.image)
    data, images = Path(args.data), Path(args.images)
    labels = json.loads((text_dir / "labels.json").read_text(encoding="utf-8"))

    available = {p.stem for p in (images / "images").glob("*.jpg")}

    def paired(split: str) -> list[dict[str, Any]]:
        return [
            r
            for r in read_jsonl(data / f"{split}.jsonl")
            if r["domain"].lower() in available
        ]

    train_rows, val_rows, test_rows = paired("train"), paired("val"), paired("test")
    print(
        f"paired: {len(train_rows):,} train, {len(val_rows):,} val, "
        f"{len(test_rows):,} test"
    )
    if not train_rows or not test_rows:
        raise SystemExit("no domains have both a page and a screenshot")

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(text_dir)
    processor = AutoImageProcessor.from_pretrained(image_dir)
    model = build_model(text_dir, image_dir, len(labels), freeze=not args.unfreeze).to(
        device
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"trainable parameters: {trainable:,} ({'head only' if not args.unfreeze else 'all'})"
    )

    def loader(rows: list[dict[str, Any]], shuffle: bool) -> Any:
        dataset: Any = PairedDataset(
            rows, images / "images", labels, tokenizer, processor, args.max_length
        )
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = loader(train_rows, True)
    val_loader = loader(val_rows, False)
    test_loader = loader(test_rows, False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    out = Path(args.out)
    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device),
            )
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch + 1} step {step} loss {total / (step + 1):.4f}")

        _, val_f1, _, _ = score(model, val_loader, device, labels)
        print(f"epoch {epoch + 1}: val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "joint.pt")
            (out / "labels.json").write_text(json.dumps(labels, indent=2))
            print(f"  saved to {out} (new best)")

    # Score the checkpoint on disk, not the model in memory: with best-epoch-only saving
    # they differ whenever validation peaked before the last epoch.
    if (out / "joint.pt").exists():
        # weights_only=True: the default unpickles arbitrary Python objects, and a
        # checkpoint is data, not code. This file holds tensors only.
        model.load_state_dict(
            torch.load(out / "joint.pt", map_location=device, weights_only=True)
        )
        print(f"scoring the saved checkpoint (val macro-F1 {best_f1:.4f})")

    accuracy, joint_f1, truth, predicted = score(model, test_loader, device, labels)
    report = per_class_report(truth, predicted)

    text_metrics = json.loads(
        (text_dir / "test_metrics.json").read_text(encoding="utf-8")
    )
    text_f1 = text_metrics["macro_f1"]
    gain = joint_f1 - text_f1

    print(f"\n{'model':16s} {'accuracy':>9s} {'macro-F1':>9s}")
    print(f"{'text only':16s} {text_metrics['accuracy']:9.3f} {text_f1:9.3f}")
    print(f"{'joint':16s} {accuracy:9.3f} {joint_f1:9.3f}")
    print(f"\ngain over text: {gain:+.4f} macro-F1 (need +{REQUIRED_GAIN})")

    # An aggregate over 47 classes hides an effect concentrated in four, so report the
    # classes where late fusion gave the screenshot real weight.
    print(f"\n{'class where images earned weight':34s} {'f1':>6s} {'n':>5s}")
    for name in VISUAL_CLASSES:
        row = report.get(name)
        if row:
            print(f"{name:34s} {row['f1']:6.3f} {int(row['support']):5d}")

    (out / "test_metrics.json").write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "macro_f1": joint_f1,
                "text_only_macro_f1": text_f1,
                "gain": gain,
                "required_gain": REQUIRED_GAIN,
                "clears_band": gain >= REQUIRED_GAIN,
                "per_class": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if gain >= REQUIRED_GAIN:
        print(f"\nJoint beats text by {gain:.4f} -- ship it.")
        return 0
    print(
        f"\nJoint does not clear the band ({gain:+.4f} against +{REQUIRED_GAIN}).\n"
        "Report the number and ship text-only; a 224px screenshot carries little the "
        "page text does not."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
