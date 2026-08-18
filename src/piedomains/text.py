#!/usr/bin/env python3
"""Text-based domain classification using HTML content analysis."""

import os
from datetime import UTC
from pathlib import Path
from typing import Any

from .blocking import is_thin
from .checkpoints import eager_config, read_labels, read_temperature
from .config import get_config
from .constants import classes
from .content_processor import ContentProcessor
from .labels import project, top_labels
from .outcomes import ErrorCode, Stage
from .piedomains_logging import get_logger

logger = get_logger()

#: Where the fine-tuned classifier lives. Overridable with
#: ``PIEDOMAINS_TEXT_MODEL`` (a Hub repo id or a local directory), which is what
#: `training/train_text.py` output is pointed at before it is published.
DEFAULT_TEXT_MODEL = "gojiberries/piedomains-text"
DEFAULT_TEXT_REVISION = "38d7e6403f911902f47b9218ce5c645a06dd02fe"


def _pick_device() -> str:
    """Choose the fastest available torch device.

    Returns:
        str: ``cuda``, ``mps`` or ``cpu``.
    """
    import torch  # pyright: ignore[reportMissingImports]

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_text_model(latest: bool = False) -> str:
    """Decide which checkpoint to load.

    A local directory is preferred when one is configured, so a freshly trained
    model can be evaluated before it is published anywhere.

    Args:
        latest: Bypass any local cache and re-resolve from the Hub.

    Returns:
        str: A local path or a Hugging Face Hub repo id.
    """
    configured = os.environ.get("PIEDOMAINS_TEXT_MODEL") or get_config().get(
        "text_model"
    )
    if configured:
        return str(configured)
    if latest:
        logger.info("latest=True: reloading pinned %s checkpoint", DEFAULT_TEXT_MODEL)
    return DEFAULT_TEXT_MODEL


class TextClassifier:
    """Text-based domain content classifier."""

    def __init__(self, cache_dir: str | None = None, archive_date: str | None = None):
        """Initialize text classifier.

        Args:
            cache_dir: Directory for caching content
            archive_date: Date for archive.org snapshots
        """
        self.cache_dir = Path(cache_dir or "cache")
        self.archive_date = archive_date
        self.processor = ContentProcessor(str(self.cache_dir), archive_date)
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._labels: list[str] = list(classes)
        #: Fitted by training/calibrate.py; 1.0 means no scaling.
        self._temperature: float = 1.0
        self._max_length: int = get_config().get("text_max_length", 256)

    def load_models(self, latest: bool = False) -> None:
        """Load the classifier, its tokenizer, labels and temperature.

        The model is a HuggingFace sequence-classification checkpoint written by
        ``training/train_text.py``: safetensors weights, a tokenizer, a
        ``labels.json`` giving class order, and a ``calibration.json`` holding
        the fitted temperature.

        Args:
            latest: Re-resolve the model even if one is already loaded.

        Raises:
            RuntimeError: If the model cannot be loaded. Deliberately fatal --
                this used to substitute a model returning all zeros, so every
                domain came back as ``adv`` at confidence 0.0, indistinguishable
                from a real prediction.
        """
        if self._model is not None and not latest:
            return

        try:
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            source = resolve_text_model(latest)
            revision = DEFAULT_TEXT_REVISION if source == DEFAULT_TEXT_MODEL else None
            self._tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                source,
                config=eager_config(source, revision=revision),
                revision=revision,
            )
            self._model.eval()
            self._device = _pick_device()
            self._model.to(self._device)

            self._labels = read_labels(
                source, self._model.config, classes, revision=revision
            )
            self._temperature = read_temperature(source, revision=revision)

            logger.info(
                f"Loaded text model from {source} on {self._device}: "
                f"{len(self._labels)} classes, temperature {self._temperature:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to load text models: {e}")
            raise RuntimeError(
                f"Could not load the text classification model: {e}"
            ) from e

    def classify(self, domains: list[str], latest: bool = False) -> list[dict]:
        """Classify domains using their cached HTML content.

        Args:
            domains: List of domain names to classify
            latest: Whether to use latest model version

        Returns:
            List of classification result dictionaries

        Example:
            >>> classifier = TextClassifier()
            >>> results = classifier.classify(["cnn.com", "bbc.com"])
            >>> print(results[0]["category"])
            news
        """
        self.load_models(latest)

        results = []

        for domain in domains:
            result = self._classify_single_domain(domain)
            results.append(result)

        return results

    @staticmethod
    def _model_input(domain: str, processed_text: str) -> str:
        """Build the model input exactly as training did.

        Training fed ``df['domain'] + ' ' + df['text']`` where ``domain`` was
        the name with its TLD stripped (notebooks/04_train_model.ipynb). Serving
        omitted the prefix entirely, so a feature the model was trained on was
        missing at inference.

        Args:
            domain: Domain name, e.g. ``"yahoo.com"``.
            processed_text: Cleaned page text.

        Returns:
            str: The text to feed the model.
        """
        stem = domain.rsplit(".", 1)[0] if "." in domain else domain
        return f"{stem} {processed_text}"

    def _classify_single_domain(self, domain: str) -> dict:
        """Classify a single domain using its cached HTML file."""
        html_path = self.cache_dir / "html" / f"{domain}.html"

        result = {
            "url": domain,
            "domain": domain,
            "text_path": str(html_path.relative_to(self.cache_dir)),
            "image_path": str(
                (self.cache_dir / "images" / f"{domain}.png").relative_to(
                    self.cache_dir
                )
            ),
            "date_time_collected": None,  # Would need to read from metadata if needed
            "model_used": "text/shallalist_ml",
            "category": None,
            "confidence": None,
            "categories": [],
            "raw_predictions": None,
            "reason": None,
            "error": None,
        }

        if not html_path.exists():
            result["error"] = f"HTML file not found: {html_path}"
            return result

        try:
            # Load and process HTML content
            html_content = html_path.read_text(encoding="utf-8")

            # Process HTML to text
            from .text_processor import TextProcessor

            processed_text = TextProcessor.process_html_to_text(html_content)
            self._score_or_refuse(result, domain, processed_text)

        except FileNotFoundError:
            result["error"] = f"HTML file not found: {html_path}"
        except Exception as e:
            result["error"] = f"Classification error: {e}"

        return result

    def _score_or_refuse(self, result: dict, domain: str, text: str) -> None:
        """Label the page, or record why it is not labelable.

        Below the token floor the model's output is dominated by its prior
        (``recreation`` 0.31, ``shopping`` 0.21, ``porn`` 0.19 on empty input),
        which is where results like ``facebook.com -> porn`` come from. Refusing
        is the honest answer; the alternative is a confident label built out of
        nothing.

        Args:
            result: Row to write into, mutated in place.
            domain: Domain being classified, prepended as a model feature.
            text: Processed page text.
        """
        if not text.strip():
            result["error"] = "No meaningful text extracted"
            result["error_code"] = ErrorCode.EMPTY_TEXT.value
            result["stage"] = Stage.PROCESS.value
            return

        floor = get_config().get("min_tokens", 30)
        if is_thin(text, min_tokens=floor):
            result["error"] = (
                f"only {len(text.split())} usable tokens, below the {floor}-token "
                "floor for a meaningful label"
            )
            result["error_code"] = ErrorCode.THIN_CONTENT.value
            result["stage"] = Stage.PROCESS.value
            return

        predictions = self._predict_text(self._model_input(domain, text))
        result["category"] = predictions.get("text_label")
        result["confidence"] = predictions.get("text_prob")
        result["categories"] = predictions.get("text_categories") or []
        result["raw_predictions"] = predictions.get("text_domain_probs")

    def classify_from_paths(
        self,
        data_paths: list[dict],
        output_file: str | None = None,
        latest: bool = False,
    ) -> list[dict]:
        """Classify domains using HTML files from collected data paths.

        Args:
            data_paths: List of dicts with domain data containing text_path, domain, etc.
            output_file: Optional path to save JSON results
            latest: Whether to use latest model version

        Returns:
            List of classification result dictionaries (JSON format)

        Example:
            >>> classifier = TextClassifier()
            >>> data = [{"domain": "cnn.com", "text_path": "html/cnn.com.html", ...}]
            >>> results = classifier.classify_from_paths(data)
            >>> print(results[0]["category"])
            news
        """
        self.load_models(latest)

        results = []

        for domain_data in data_paths:
            domain = domain_data.get("domain")
            text_path = domain_data.get("text_path")

            result = {
                "url": domain_data.get("url", domain),
                "domain": domain,
                "text_path": text_path,
                "image_path": domain_data.get("image_path"),
                "date_time_collected": domain_data.get("date_time_collected"),
                "model_used": "text/shallalist_ml",
                "category": None,
                "confidence": None,
                "categories": [],
                "raw_predictions": None,
                "reason": None,
                "error": None,
            }

            if not domain or not text_path:
                # Prefer the fetcher's own account of why there is no text. It
                # knows the page rendered 11 words; all this layer can see is an
                # absent path, and reporting that would name the symptom.
                result["error"] = (
                    domain_data.get("error") or "Missing domain or text_path"
                )
                result["error_code"] = (
                    domain_data.get("error_code") or ErrorCode.MISSING_INPUT_PATH.value
                )
                result["stage"] = Stage.PROCESS.value
                results.append(result)
                continue

            try:
                # Clean path resolution using pathlib
                if not Path(text_path).is_absolute():
                    html_path = self.cache_dir / text_path
                else:
                    html_path = Path(text_path)

                # Load HTML from file path
                html_content = html_path.read_text(encoding="utf-8")

                # Process HTML to text
                from .text_processor import TextProcessor

                processed_text = TextProcessor.process_html_to_text(html_content)
                self._score_or_refuse(result, domain, processed_text)

            except FileNotFoundError:
                result["error"] = f"HTML file not found: {text_path}"
            except Exception as e:
                result["error"] = f"Classification error: {e}"

            results.append(result)

        # Save results if output file specified
        if output_file:
            import json
            from datetime import datetime

            # Create results directory if needed
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

            # Add metadata
            output_data = {
                "inference_timestamp": datetime.now(UTC).isoformat(),
                "model_used": "text/shallalist_ml",
                "total_domains": len(data_paths),
                "successful": len([r for r in results if r["category"] is not None]),
                "failed": len([r for r in results if r["error"] is not None]),
                "results": results,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Saved classification results to {output_file}")

        return results

    def classify_from_data(
        self,
        collection_data: dict,
        output_file: str | None = None,
        latest: bool = False,
    ) -> list[dict]:
        """Classify domains using collection metadata from DataCollector.

        Args:
            collection_data: Collection metadata dict from DataCollector.collect()
            output_file: Optional path to save JSON results
            latest: Whether to use latest model version

        Returns:
            List of classification result dictionaries (JSON format)

        Example:
            >>> from piedomains import DataCollector
            >>> collector = DataCollector()
            >>> data = collector.collect(["cnn.com"])
            >>> classifier = TextClassifier()
            >>> results = classifier.classify_from_data(data)
        """
        # Extract domain data from collection metadata
        domains_data = collection_data.get("domains", [])

        # Filtered on fetch_success alone, matching the image path. Requiring
        # text_path here drops the row instead of explaining it: a page that
        # renders but yields almost no text succeeds with a screenshot and no
        # HTML, and dropping it turns "the page had 11 words" into a domain that
        # silently disappears from the results.
        valid_domains = [d for d in domains_data if d.get("fetch_success")]

        if not valid_domains:
            logger.warning("No valid domains with text data found in collection")
            return []

        return self.classify_from_paths(valid_domains, output_file, latest)

    def _predict_text(self, text: str) -> dict:
        """Score one page of text.

        Confidence is ``softmax(logits / T)`` with the temperature fitted by
        ``training/calibrate.py``. Unlike the elementwise isotonic calibration
        this replaces, the result is always a proper distribution: it sums to 1,
        and the argmax is the argmax of that distribution rather than of a
        vector of independently-mapped scores.

        Args:
            text: Cleaned page text, already carrying the domain prefix.

        Returns:
            dict: ``text_label``, ``text_prob``, the full ``text_domain_probs``
            distribution, and ``text_categories`` -- every label above the
            configured threshold. All ``None``/empty on failure.
        """
        try:
            import torch  # pyright: ignore[reportMissingImports]

            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=self._max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**encoded).logits[0]
            probabilities = torch.softmax(logits / self._temperature, dim=-1)

            # Project the checkpoint's own labels onto the documented space. Merged
            # classes sum -- renaming dict keys instead would keep whichever source came
            # last and silently discard the other's mass. Excluded classes are dropped and
            # the remainder renormalised, so this is still a distribution.
            names, groups = project(self._labels)
            summed = [sum(float(probabilities[i]) for i in group) for group in groups]
            total = sum(summed) or 1.0
            scores = {
                name: value / total for name, value in zip(names, summed, strict=True)
            }
            best = max(scores, key=lambda k: scores[k])
            logger.debug(f"Text prediction: {best} ({scores[best]:.3f})")

            # Built from the same distribution as `text_domain_probs`, so the reported
            # labels and the raw probabilities cannot disagree.
            ranked = top_labels(scores, get_config().get("multilabel_threshold", 0.10))

            return {
                "text_label": best,
                "text_prob": scores[best],
                "text_domain_probs": scores,
                "text_categories": ranked,
            }

        except Exception as e:
            logger.error(f"Text prediction failed: {e}")
            return {
                "text_label": None,
                "text_prob": None,
                "text_domain_probs": None,
                "text_categories": [],
            }
