#!/usr/bin/env python3
"""Screenshot-based classification — currently unavailable.

The screenshot model was a frozen-backbone ResNet50 in TensorFlow SavedModel
format. It is gone for two independent reasons, and neither is worth working
around:

* **TensorFlow had to go.** It ships no ``cp314`` wheels, which pinned the whole
  package below Python 3.14.
* **It did not work.** It reported 52.9% at training time with
  ``base_model.trainable = False`` — only a linear head was ever fitted — and in
  practice labelled Khan Academy and Yahoo as ``porn``. Separately, the serving
  path divided pixels by 255 before handing them to a graph that already baked in
  ``resnet50.preprocess_input``, so every image arrived as a near-constant
  negative array. The 52.9% was a training-time number users never got.

**Nothing measurable is lost in the meantime.** The "combined" path never merged
the two probability vectors: it returned the *text* label every time and only
averaged the confidences, so the image model could not change an answer, only
blur the number attached to it.

The class is kept so the import and the public surface stay stable, and so the
failure names itself instead of surfacing as a missing attribute somewhere
downstream. The replacement — a properly fine-tuned backbone plus late fusion
over both probability vectors — is the next piece of work.
"""

from pathlib import Path
from typing import Any

from .content_processor import ContentProcessor
from .piedomains_logging import get_logger

logger = get_logger()

_UNAVAILABLE = (
    "Image classification is unavailable in this version. The TensorFlow "
    "screenshot model was removed (it ships no Python 3.14 wheels, and it "
    "labelled Khan Academy as 'porn'); the PyTorch replacement is not trained "
    "yet. Use classify_by_text(), or classify(), which is text-only for now."
)


class ImageClassifier:
    """Placeholder for the screenshot classifier while it is being retrained.

    Every entry point raises :class:`NotImplementedError` with the reason.
    Screenshot *collection* is unaffected — ``DataCollector`` still captures and
    caches them, so no data is lost while the model is rebuilt.
    """

    MODELFN = "model/shallalist"

    def __init__(self, cache_dir: str | None = None, archive_date: str | None = None):
        """Initialize the placeholder.

        Constructing this is deliberately allowed: callers that build one up
        front should fail when they try to classify, not at import time.

        Args:
            cache_dir: Directory for caching content.
            archive_date: Date for archive.org snapshots.
        """
        self.cache_dir = Path(cache_dir or "cache")
        self.archive_date = archive_date
        self.processor = ContentProcessor(str(self.cache_dir), archive_date)
        self._model: Any = None

    def load_models(self, latest: bool = False) -> None:
        """Report that no image model is available.

        Args:
            latest: Accepted for interface compatibility; unused.

        Raises:
            NotImplementedError: Always, naming the path to use instead.
        """
        raise NotImplementedError(_UNAVAILABLE)

    def classify(self, domains: list[str], latest: bool = False) -> list[dict]:
        """Reject a screenshot classification request.

        Args:
            domains: Domains that would have been classified.
            latest: Accepted for interface compatibility; unused.

        Returns:
            list[dict]: Never returns.

        Raises:
            NotImplementedError: Always, naming the path to use instead.
        """
        raise NotImplementedError(_UNAVAILABLE)

    def classify_from_paths(
        self,
        data_paths: list[dict],
        output_file: str | None = None,
        latest: bool = False,
    ) -> list[dict]:
        """Reject a screenshot classification request.

        Args:
            data_paths: Collected screenshot paths.
            output_file: Where results would have been written.
            latest: Accepted for interface compatibility; unused.

        Returns:
            list[dict]: Never returns.

        Raises:
            NotImplementedError: Always, naming the path to use instead.
        """
        raise NotImplementedError(_UNAVAILABLE)

    def classify_from_data(
        self,
        collection_data: dict,
        output_file: str | None = None,
        latest: bool = False,
    ) -> list[dict]:
        """Reject a screenshot classification request.

        Args:
            collection_data: The collection envelope.
            output_file: Where results would have been written.
            latest: Accepted for interface compatibility; unused.

        Returns:
            list[dict]: Never returns.

        Raises:
            NotImplementedError: Always, naming the path to use instead.
        """
        raise NotImplementedError(_UNAVAILABLE)
