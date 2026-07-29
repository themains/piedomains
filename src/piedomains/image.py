#!/usr/bin/env python3
"""Screenshot-based classification.

Replaces a ResNet50 that reported 52.9% at training time with a frozen backbone — only a
linear head was ever fitted — and, in production, labelled Khan Academy and Yahoo as
``porn``. Two independent bugs produced that: the backbone was never fine-tuned, and the
serving path divided pixels by 255 before handing them to a graph that already baked in
``resnet50.preprocess_input``, so every image arrived as a near-constant negative array.

Both are gone. The backbone is fully fine-tuned (``training/train_image.py``) and
preprocessing is the model's own ``AutoImageProcessor``, so training and inference cannot
drift apart.

**A screenshot is a weak signal on its own, and that is the point.** At 224px a page is
unreadable — the model sees layout, colour and gross structure, not text. It exists to
say something the text model cannot, which matters most on pages carrying almost no text.
It is combined with text through calibrated late fusion, never used to overrule it
blindly.
"""

import json
import os
from pathlib import Path
from typing import Any

from .checkpoints import read_labels, read_temperature
from .constants import classes
from .content_processor import ContentProcessor
from .piedomains_logging import get_logger

logger = get_logger()

#: Where the fine-tuned screenshot model lives. Overridable with
#: ``PIEDOMAINS_IMAGE_MODEL`` (a Hub repo id or a local directory).
DEFAULT_IMAGE_MODEL = "soodoku/piedomains-image"


def resolve_image_model(latest: bool = False) -> str:
    """Decide which checkpoint to load.

    A local directory is preferred when configured, so a freshly trained model can be
    evaluated before it is published.

    Args:
        latest: Re-resolve from the Hub even if a copy is cached.

    Returns:
        str: A local path or a Hugging Face Hub repo id.
    """
    from .config import get_config

    configured = os.environ.get("PIEDOMAINS_IMAGE_MODEL") or get_config().get(
        "image_model"
    )
    if configured:
        return str(configured)
    if latest:
        logger.info("latest=True: re-resolving %s from the Hub", DEFAULT_IMAGE_MODEL)
    return DEFAULT_IMAGE_MODEL


class ImageClassifier:
    """Classify a website from a screenshot of its homepage."""

    def __init__(self, cache_dir: str | None = None, archive_date: str | None = None):
        """Initialize the classifier.

        Args:
            cache_dir: Directory for caching content.
            archive_date: Date for archive.org snapshots.
        """
        self.cache_dir = Path(cache_dir or "cache")
        self.archive_date = archive_date
        self.processor_cache = ContentProcessor(str(self.cache_dir), archive_date)
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"
        self._labels: list[str] = list(classes)
        #: Fitted by training/calibrate.py --modality image; 1.0 means no scaling.
        self._temperature: float = 1.0

    def load_models(self, latest: bool = False) -> None:
        """Load the classifier, its preprocessor, labels and temperature.

        Args:
            latest: Re-resolve the model even if one is already loaded.

        Raises:
            RuntimeError: If the model cannot be loaded. Deliberately fatal — the
                predecessor substituted a model returning zeros, so every domain came
                back as a confident-looking wrong answer.
        """
        if self._model is not None and not latest:
            return

        try:
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoImageProcessor,
                AutoModelForImageClassification,
            )

            from .text import _pick_device

            source = resolve_image_model(latest)
            self._processor = AutoImageProcessor.from_pretrained(source)
            self._model = AutoModelForImageClassification.from_pretrained(source)
            self._model.eval()
            self._device = _pick_device()
            self._model.to(self._device)

            self._labels = read_labels(source, self._model.config, classes)
            self._temperature = read_temperature(source)

            logger.info(
                f"Loaded image model from {source} on {self._device}: "
                f"{len(self._labels)} classes, temperature {self._temperature:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            raise RuntimeError(
                f"Could not load the screenshot classification model: {e}"
            ) from e

    def predict_proba(self, image_path: str | Path) -> dict[str, float] | None:
        """Score one screenshot into a calibrated probability distribution.

        Args:
            image_path: Path to the screenshot.

        Returns:
            dict[str, float] | None: Class probabilities summing to 1, or ``None`` when
            the image cannot be read.
        """
        import torch  # pyright: ignore[reportMissingImports]
        from PIL import Image

        path = Path(image_path)
        if not path.is_absolute():
            path = self.cache_dir / path
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not read screenshot {path}: {e}")
            return None

        encoded = self._processor(images=img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            logits = self._model(**encoded).logits[0]
        probabilities = torch.softmax(logits / self._temperature, dim=-1)
        return {name: float(probabilities[i]) for i, name in enumerate(self._labels)}

    def _classify_one(self, domain: str, image_path: str | Path | None) -> dict:
        """Build one result row for a domain.

        Args:
            domain: Domain being classified.
            image_path: Path to its screenshot, if there is one.

        Returns:
            dict: A result row, carrying an ``error`` when no screenshot was usable.
        """
        from .outcomes import ErrorCode, Stage

        result: dict[str, Any] = {
            "url": domain,
            "domain": domain,
            "image_path": str(image_path) if image_path else None,
            "model_used": "image/vit",
            "category": None,
            "confidence": None,
            "raw_predictions": None,
            "error": None,
        }
        if not image_path:
            result["error"] = "No screenshot available"
            result["error_code"] = ErrorCode.MISSING_SCREENSHOT.value
            result["stage"] = Stage.PROCESS.value
            return result

        scores = self.predict_proba(image_path)
        if scores is None:
            result["error"] = f"Screenshot could not be read: {image_path}"
            result["error_code"] = ErrorCode.MISSING_SCREENSHOT.value
            result["stage"] = Stage.PROCESS.value
            return result

        best = max(scores, key=lambda k: scores[k])
        result["category"] = best
        result["confidence"] = scores[best]
        result["raw_predictions"] = scores
        return result

    def classify(self, domains: list[str], latest: bool = False) -> list[dict]:
        """Classify domains from their cached screenshots.

        Args:
            domains: Domain names to classify.
            latest: Whether to re-resolve the model.

        Returns:
            list[dict]: One result row per domain.
        """
        self.load_models(latest)
        rows = []
        for domain in domains:
            path = self.cache_dir / "images" / f"{domain}.png"
            rows.append(self._classify_one(domain, path if path.exists() else None))
        return rows

    def classify_from_paths(
        self,
        data_paths: list[dict],
        output_file: str | None = None,
        latest: bool = False,
    ) -> list[dict]:
        """Classify domains from collected screenshot paths.

        Args:
            data_paths: Records carrying ``domain`` and ``image_path``.
            output_file: Optional path to write JSON results to.
            latest: Whether to re-resolve the model.

        Returns:
            list[dict]: One result row per record.
        """
        self.load_models(latest)
        rows = []
        for entry in data_paths:
            domain = entry.get("domain") or entry.get("url") or ""
            row = self._classify_one(domain, entry.get("image_path"))
            row["url"] = entry.get("url", domain)
            row["date_time_collected"] = entry.get("date_time_collected")
            rows.append(row)

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text(json.dumps({"results": rows}, indent=2))
            logger.info(f"Saved image classification results to {output_file}")
        return rows

    def classify_from_data(
        self,
        collection_data: dict,
        output_file: str | None = None,
        latest: bool = False,
    ) -> list[dict]:
        """Classify every successfully collected domain in a collection envelope.

        Args:
            collection_data: The envelope from ``DataCollector``.
            output_file: Optional path to write JSON results to.
            latest: Whether to re-resolve the model.

        Returns:
            list[dict]: One result row per collected domain.
        """
        entries = [
            d
            for d in collection_data.get("domains", [])
            if d.get("fetch_success") and d.get("image_path")
        ]
        return self.classify_from_paths(entries, output_file, latest)
