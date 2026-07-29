"""Combine the text and screenshot models into one answer.

**What this replaces.** The package advertised an ensemble for years and did not have
one. It ran both models, returned the *text* label every time, and set confidence to
``(text_conf + image_conf) / 2`` — averaging an isotonic-calibrated, unnormalized text
score against a raw image softmax. The image model could not change an answer, only blur
the number attached to it.

Three things make this real:

* **Both inputs are calibrated.** Each model's probabilities come from
  ``softmax(logits / T)`` with its own fitted temperature. Mixing an uncalibrated vector
  with a calibrated one hands the decision to whichever model is more overconfident.
* **The weights are fitted, not assumed.** ``training/fuse.py`` fits them on domains that
  have both a page and a screenshot, and writes them to ``fusion.json``. There is no
  hard-coded 0.5.
* **It refuses rather than guesses.** No weights, or two models that disagree about the
  label set, is an error — not a silent fallback to averaging.

A missing screenshot degrades to text-only, which is the honest answer rather than a
half-informed blend.
"""

from __future__ import annotations

from dataclasses import dataclass

from .checkpoints import read_sidecar
from .piedomains_logging import get_logger

__all__ = ["FusionWeights", "fuse_probabilities", "load_fusion_weights"]

logger = get_logger()


@dataclass(frozen=True)
class FusionWeights:
    """How much of each modality to trust.

    Weights apply to the *text* side; the image side gets the remainder. A per-class
    vector lets the screenshot carry more where it is actually informative — plausibly
    ``adult`` and ``shopping`` more than ``government``.
    """

    #: Text weight, either one scalar or one per class.
    text: tuple[float, ...]
    #: Class order these weights were fitted against. Fusing across a different order
    #: would silently permute every class.
    labels: tuple[str, ...]

    def weight_for(self, index: int) -> float:
        """Return the text weight for one class.

        Args:
            index: Class index.

        Returns:
            float: Weight applied to the text probability.

        Raises:
            ValueError: If no weights were loaded at all.
        """
        if not self.text:
            raise ValueError("fusion weights are empty")
        if len(self.text) == 1:
            return self.text[0]
        return self.text[index]


def load_fusion_weights(source: str) -> FusionWeights | None:
    """Read fitted fusion weights shipped alongside a model.

    Args:
        source: Local directory or Hub repo id holding ``fusion.json``.

    Returns:
        FusionWeights | None: The weights, or ``None`` when absent.
    """
    payload = read_sidecar(source, "fusion.json")
    if not payload:
        return None
    return FusionWeights(
        text=tuple(float(w) for w in payload["text_weights"]),
        labels=tuple(payload["labels"]),
    )


def fuse_probabilities(
    text: dict[str, float] | None,
    image: dict[str, float] | None,
    weights: FusionWeights,
) -> dict[str, float]:
    """Blend two calibrated distributions into one.

    Args:
        text: Calibrated text probabilities, or ``None`` if unavailable.
        image: Calibrated image probabilities, or ``None`` if unavailable.
        weights: Fitted weights, carrying the class order they were fitted on.

    Returns:
        dict[str, float]: The blended distribution, renormalized to sum to 1.

    Raises:
        ValueError: If neither modality produced anything, or if either disagrees with
            the class order the weights were fitted against. Both are silent-corruption
            failures otherwise: a permuted label set mislabels everything while looking
            perfectly healthy.
    """
    present = text if text is not None else image
    if present is None:
        raise ValueError("fusion needs at least one modality")
    if tuple(present) != weights.labels:
        raise ValueError(
            "model label order does not match the order fusion weights were fitted "
            f"on ({len(present)} vs {len(weights.labels)} classes)"
        )

    if image is None:
        logger.debug("no screenshot; returning text-only rather than blending")
        return dict(present)
    if text is None:
        logger.debug("no page text; returning image-only rather than blending")
        return dict(present)

    blended = {}
    for i, name in enumerate(weights.labels):
        w = weights.weight_for(i)
        blended[name] = w * text[name] + (1.0 - w) * image[name]

    total = sum(blended.values())
    if total <= 0:
        raise ValueError("fused probabilities sum to zero")
    return {name: value / total for name, value in blended.items()}
