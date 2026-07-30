"""Read the files that ship alongside model weights.

Every model this package loads — text today, screenshots next — carries the same three
sidecars: `labels.json` for class order, `calibration.json` for the fitted temperature,
and for the ensemble, `fusion.json` for the mixing weights.

This module exists because that reading logic was written once for text and would
otherwise be written again for images. The first version checked ``Path(source) /
filename`` and nothing else, which is correct for a local directory and silently wrong
for a Hub repo id: the model loaded, logged success, and ran at temperature 1.0 with no
calibration at all. Nothing failed — it just quietly stopped doing the thing it was for.
A second copy would be a second chance to make that mistake.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .piedomains_logging import get_logger

__all__ = ["read_labels", "read_sidecar", "read_temperature"]

logger = get_logger()


def read_sidecar(source: str, filename: str) -> dict | None:
    """Read a JSON file shipped alongside the weights.

    ``source`` is either a local directory or a Hub repo id, and the difference matters:
    a repo id is not a path, so a local existence check always misses on it.

    Args:
        source: Local directory or Hugging Face Hub repo id.
        filename: Sidecar file to read.

    Returns:
        dict | None: The parsed contents, or ``None`` when absent.
    """
    local = Path(source) / filename
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    if Path(source).exists():
        return None  # a real local directory that simply lacks the file

    from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]
    from huggingface_hub.errors import (  # pyright: ignore[reportMissingImports]
        EntryNotFoundError,
    )

    try:
        path = hf_hub_download(repo_id=source, filename=filename)
    except (EntryNotFoundError, OSError, ValueError):
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_labels(source: str, config: Any, fallback: list[str]) -> list[str]:
    """Read class order for a checkpoint.

    ``labels.json`` is authoritative when present; otherwise the model's own ``id2label``
    is used, in index order. Reading the order off the checkpoint rather than a module
    constant is what stops a retrain from silently permuting every label.

    Args:
        source: Local directory or Hub repo id the model came from.
        config: The loaded model's config.
        fallback: Class list to use when the checkpoint carries no usable names.

    Returns:
        list[str]: Category names in class-index order.
    """
    payload = read_sidecar(source, "labels.json")
    if payload:
        return list(payload)

    id2label = getattr(config, "id2label", None) or {}
    if id2label and not all(str(v).startswith("LABEL_") for v in id2label.values()):
        return [id2label[i] for i in sorted(id2label, key=int)]

    logger.warning(
        "No labels.json and no usable id2label in %s; falling back to the built-in "
        "%d-class list. Verify this matches the checkpoint.",
        source,
        len(fallback),
    )
    return list(fallback)


def read_temperature(source: str) -> float:
    """Read the fitted calibration temperature.

    Args:
        source: Local directory or Hub repo id the model came from.

    Returns:
        float: The temperature, or ``1.0`` (no scaling) when absent.
    """
    payload = read_sidecar(source, "calibration.json")
    if payload:
        return float(payload.get("temperature", 1.0))
    logger.warning(
        "No calibration.json alongside %s: confidences are RAW softmax outputs, which "
        "are badly overconfident. Run training/calibrate.py.",
        source,
    )
    return 1.0
