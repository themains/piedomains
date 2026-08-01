#!/usr/bin/env python3
"""The one way a screenshot becomes model input.

**Why this is a module and not three inlined resizes.** Preprocessing that differs between
training and serving is the single failure this project has already shipped. The previous
image model divided pixels by 255 before a graph that already applied
``resnet50.preprocess_input``; in production it labelled Khan Academy and Yahoo as ``porn``
while reporting 52.9% at training time. The model was fine. It was being shown different
pictures than it learned from.

A DRY audit found the same shape of bug live again, in three copies that had drifted, for
a 1280x1024 capture:

======================  =========================================================
where                   what it did
======================  =========================================================
``prepare_images.py``   centre-crop to 1024 square, then resize
``image.py`` (serving)  handed straight to ``SiglipImageProcessor``, which resizes
                        to 224x224 with no crop -- i.e. squashes
``capture_screenshots``  **left**-crop to 1024 square, then resize
======================  =========================================================

**Squash, do not crop -- and the evidence points the opposite way to intuition.** SigLIP 2
(arXiv:2502.14786 §2.4.2) pretrains "with non-aspect preserving resize", and its
``big_vision`` configs use the ``resize`` op (both dimensions forced) rather than
``resize_small`` (aspect-preserving). All 10B WebLI images went through a square squash. So
squashing *reproduces* the distribution the encoder was pretrained on, and cropping is the
domain shift -- the reverse of what "distortion is bad" suggests.

Cropping also loses the bottom of every page for nothing. At 1280x1024 the squash is a 25%
horizontal compression, which is mild; the aspect-ratio argument that motivates SigLIP 2's
NaFlex variant targets OCR, where glyph geometry has to survive, and at 224px this model
cannot read page text either way.

**Two knobs with better evidence than crop geometry, deliberately not taken here:**
resolution scales monotonically (SigLIP 2 Table 1: ImageNet 0-shot 78.2 at 224 to 81.2 at
512; StrucTexTv2 measured 89.5% to 92.5% on document classification over the same range),
and Homepage2Vec strips cookie and modal ``div`` elements before capture, since a consent
banner over the hero region is exactly this task's failure mode. Both are changes to what
is captured, not to how it is resized, and both want measuring on their own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover -- PIL is imported lazily at call time
    from PIL.Image import Image

__all__ = ["IMAGE_SIZE", "resize_for_model"]

#: Edge length the encoder expects. `google/siglip2-base-patch16-224` and the checkpoints
#: fine-tuned from it. Raising this means retraining, and is one of the two levers the
#: module docstring flags as better supported than crop geometry.
IMAGE_SIZE = 224


def resize_for_model(image: Image, size: int = IMAGE_SIZE) -> Image:
    """Resize a screenshot to the square the encoder expects.

    Every path that turns a screenshot into model input must call this -- corpus
    preparation, live inference, evaluation capture. When one of them stops, the model is
    shown a different picture than it was trained on, and no metric computed on the other
    path can see it.

    Aspect ratio is deliberately **not** preserved: this reproduces SigLIP 2's own
    pretraining transform. See the module docstring for the citation.

    Args:
        image: The source screenshot, any mode or aspect ratio.
        size: Edge length of the result, in pixels.

    Returns:
        Image: An RGB image of exactly ``size`` x ``size``.

    Example:
        >>> from PIL import Image as PILImage
        >>> resize_for_model(PILImage.new("RGB", (1280, 1024))).size
        (224, 224)
        >>> resize_for_model(PILImage.new("L", (500, 900)), 64).mode
        'RGB'
    """
    from PIL import Image as PILImage

    return image.convert("RGB").resize((size, size), PILImage.Resampling.LANCZOS)
