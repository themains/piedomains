#!/usr/bin/env python
r"""Fine-tune a multilingual encoder to classify page text into categories.

Replaces the shipped bag-of-embeddings model
(``Embedding(525473, 64) -> GlobalAveragePooling1D -> Dense``), which is 403 MB
of embedding table and reported 71.3% at training time. Two properties of that
model motivate the rewrite rather than a port:

* It is **English-only by construction**. The pipeline feeding it strips
  non-English words and applies NLTK English stopwords, so a non-English site
  degrades to noise. `mmBERT <https://github.com/JHU-CLSP/mmBERT>`_ is
  multilingual, which removes that ceiling rather than raising it.
* Averaged embeddings cannot use word order, so ``free shipping returns`` and
  ``free returns shipping`` are the same input. A transformer can tell them
  apart.

Checkpoints every epoch and resumes from the newest one, because this is meant
to survive a dropped Colab session without losing the run.

Usage::

    uv run --group train python training/train_text.py \
        --data data/prepared --out models/text-v2

The reported number that matters is macro-F1 on the held-out split, not
accuracy: the category distribution is long-tailed, and accuracy flatters a
model that only learns the head.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .metrics import macro_f1, per_class_report

#: Multilingual ModernBERT-architecture encoder.
#:
#: Small rather than base, on a measurement: on an M4 (16 GB unified memory) the
#: base model trains at 155 min/epoch against small's 21 -- a 7x gap for only
#: 2.2x the parameters (308M vs 141M), which is memory pressure, not compute.
#: Base is the better choice on a real GPU; pass ``--model jhu-clsp/mmBERT-base``
#: there.
DEFAULT_MODEL = "jhu-clsp/mmBERT-small"


@dataclass
class TrainConfig:
    """Everything that determines the run, recorded alongside the weights.

    Written to ``config.json`` next to the checkpoint so a result can be traced
    back to what produced it.
    """

    model_name: str = DEFAULT_MODEL
    max_length: int = 128
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
    """Read a JSONL split written by ``prepare_text.py``.

    Args:
        path: File to read.

    Returns:
        list[dict[str, Any]]: One dict per line.

    Raises:
        SystemExit: If the file does not exist, naming the step that makes it.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found -- run prepare_text.py first")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TextDataset:
    """Tokenized (text, label) pairs.

    Map-style by protocol -- it defines ``__len__`` and ``__getitem__``, which is all
    ``DataLoader`` requires at runtime -- rather than subclassing
    ``torch.utils.data.Dataset``, which would force a module-level ``torch`` import into a
    file that is also imported just to print ``--help``. torch's stub is stricter than its
    behaviour, so the call sites carry a narrow ignore.

    Tokenization is deferred to ``__getitem__`` rather than done up front: the
    corpus is large enough that pre-tokenizing every split costs more memory
    than the training step itself, and the tokenizer is fast enough that this
    does not bottleneck the GPU.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        labels: list[str],
        max_length: int,
    ):
        """Bind rows to a tokenizer and label vocabulary.

        Args:
            rows: Records with ``text`` and ``category``.
            tokenizer: A HuggingFace tokenizer.
            labels: Ordered category names; index is the class id.
            max_length: Token truncation length.
        """
        self.rows = rows
        self.tokenizer = tokenizer
        self.index = {name: i for i, name in enumerate(labels)}
        self.max_length = max_length

    def __len__(self) -> int:
        """Number of examples.

        Returns:
            int: Row count.
        """
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        """Tokenize one row.

        Args:
            i: Row index.

        Returns:
            dict: ``input_ids``, ``attention_mask`` and ``labels`` tensors.
        """
        import torch

        row = self.rows[i]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
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
        labels: Ordered category names.

    Returns:
        tuple: ``(macro_f1, truth, predicted)``.
    """
    import torch

    model.eval()
    truth: list[str] = []
    predicted: list[str] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            for gold, pred in zip(
                batch["labels"].tolist(), logits.argmax(-1).cpu().tolist(), strict=True
            ):
                truth.append(labels[gold])
                predicted.append(labels[pred])
    return macro_f1(truth, predicted), truth, predicted


def save_checkpoint(
    out: Path,
    model: Any,
    tokenizer: Any,
    labels: list[str],
    config: TrainConfig,
    state: dict,
) -> None:
    """Write weights, tokenizer, labels and run state.

    Args:
        out: Directory to write into.
        model: The classifier.
        tokenizer: Its tokenizer.
        labels: Ordered category names.
        config: The run configuration.
        state: Epoch/score bookkeeping for resuming.
    """
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
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
    parser = argparse.ArgumentParser(description="Fine-tune the text classifier")
    parser.add_argument(
        "--data", required=True, help="prepare_text.py output directory"
    )
    parser.add_argument("--out", required=True, help="Where to write the model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base encoder")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Train on this many examples only -- for verifying the loop runs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the checkpoint in --out if one is there",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Fine-tune the encoder and report held-out macro-F1.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit status.
    """
    args = build_parser().parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    config = TrainConfig(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    torch.manual_seed(config.seed)

    data = Path(args.data)
    out = Path(args.out)
    labels = json.loads((data / "labels.json").read_text(encoding="utf-8"))
    train_rows = read_jsonl(data / "train.jsonl")
    val_rows = read_jsonl(data / "val.jsonl")
    test_rows = read_jsonl(data / "test.jsonl")
    if args.limit:
        train_rows = train_rows[: args.limit]
        val_rows = val_rows[: max(1, args.limit // 8)]
        test_rows = test_rows[: max(1, args.limit // 8)]

    device = pick_device()
    print(f"device: {device}")
    print(f"classes: {len(labels)}")
    print(f"train/val/test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

    source = (
        str(out)
        if (args.resume and (out / "state.json").exists())
        else config.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
    ).to(device)

    start_epoch, best_f1, stale = 0, 0.0, 0
    if args.resume and (out / "state.json").exists():
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        start_epoch, best_f1 = state.get("epoch", 0), state.get("best_f1", 0.0)
        print(f"resuming from epoch {start_epoch} (best macro-F1 {best_f1:.4f})")

    make = lambda rows, shuffle: DataLoader(  # noqa: E731
        TextDataset(rows, tokenizer, labels, config.max_length),  # pyright: ignore[reportArgumentType]
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    train_loader = make(train_rows, True)
    val_loader = make(val_rows, False)
    test_loader = make(test_rows, False)

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
            out_ = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (out_.loss / config.grad_accum).backward()
            running += out_.loss.item()

            if (step + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # MPS does not release cached blocks aggressively, and on a 16GB
            # unified-memory machine that accumulation is enough to push the
            # process into swap and wedge it in uninterruptible IO wait.
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

        # Only overwrite on an improvement. Saving every epoch unconditionally
        # means the artifact left on disk is the *last* epoch, not the best one,
        # which is silently wrong whenever the model starts overfitting -- and it
        # does: val macro-F1 peaked at epoch 3 and fell at epoch 4 on the first
        # real run.
        if improved:
            save_checkpoint(
                out,
                model,
                tokenizer,
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
    print(f"{'category':18s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}")
    for name, row in sorted(report.items(), key=lambda kv: -kv[1]["support"]):
        print(
            f"{name:18s} {row['precision']:6.3f} {row['recall']:6.3f} "
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
