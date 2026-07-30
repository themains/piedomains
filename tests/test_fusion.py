#!/usr/bin/env python

"""Tests for combining the text and screenshot models.

The package claimed to have an ensemble for years and did not: it returned the text label
every time and averaged a calibrated, unnormalized text score against a raw image
softmax, so the image model could not change an answer. These tests pin the properties
that make the replacement real — weights are loaded rather than assumed, and a label-order
mismatch is refused rather than silently permuting every class.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from piedomains.fusion import FusionWeights, fuse_probabilities, load_fusion_weights

LABELS = ("news", "shopping", "adult")


class TestLoadingWeights(unittest.TestCase):
    """Weights are an artifact; absence must be visible, not papered over."""

    def test_reads_weights_from_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "fusion.json").write_text(
                json.dumps({"text_weights": [0.7], "labels": list(LABELS)})
            )
            weights = load_fusion_weights(tmp)
            self.assertIsNotNone(weights)
            self.assertEqual(weights.labels, LABELS)
            self.assertAlmostEqual(weights.weight_for(0), 0.7)

    def test_absent_weights_return_none_rather_than_a_default(self):
        """A guessed 0.5 is exactly the bug this replaces."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_fusion_weights(tmp))

    def test_per_class_weights_are_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "fusion.json").write_text(
                json.dumps({"text_weights": [0.9, 0.5, 0.1], "labels": list(LABELS)})
            )
            weights = load_fusion_weights(tmp)
            self.assertAlmostEqual(weights.weight_for(0), 0.9)
            self.assertAlmostEqual(weights.weight_for(2), 0.1)

    def test_a_hub_repo_id_is_not_a_path(self):
        """The mistake that made calibration silently do nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(tmp) / "fusion.json"
            downloaded.write_text(
                json.dumps({"text_weights": [0.6], "labels": list(LABELS)})
            )
            with patch(
                "huggingface_hub.hf_hub_download", return_value=str(downloaded)
            ) as download:
                weights = load_fusion_weights("owner/model")
            download.assert_called_once()
            self.assertAlmostEqual(weights.weight_for(0), 0.6)


class TestBlending(unittest.TestCase):
    """The arithmetic, and what it refuses to do."""

    def setUp(self):
        self.scalar = FusionWeights(text=(0.5,), labels=LABELS)
        self.text = {"news": 0.6, "shopping": 0.3, "adult": 0.1}
        self.image = {"news": 0.2, "shopping": 0.7, "adult": 0.1}

    def test_blend_is_weighted_and_normalized(self):
        got = fuse_probabilities(self.text, self.image, self.scalar)
        self.assertAlmostEqual(sum(got.values()), 1.0, places=6)
        self.assertAlmostEqual(got["news"], 0.4, places=6)
        self.assertAlmostEqual(got["shopping"], 0.5, places=6)

    def test_the_image_model_can_change_the_answer(self):
        """The single property the old 'ensemble' did not have.

        Text says news, image says shopping strongly enough to win. The old code
        returned the text label unconditionally.
        """
        confident_image = {"news": 0.05, "shopping": 0.9, "adult": 0.05}
        got = fuse_probabilities(self.text, confident_image, self.scalar)
        self.assertEqual(max(got, key=lambda k: got[k]), "shopping")

    def test_per_class_weights_apply_per_class(self):
        weights = FusionWeights(text=(1.0, 0.0, 0.5), labels=LABELS)
        got = fuse_probabilities(self.text, self.image, weights)
        raw = {"news": 0.6, "shopping": 0.7, "adult": 0.1}
        total = sum(raw.values())
        for name, value in raw.items():
            self.assertAlmostEqual(got[name], value / total, places=6)

    def test_missing_screenshot_degrades_to_text(self):
        got = fuse_probabilities(self.text, None, self.scalar)
        self.assertEqual(got, self.text)

    def test_missing_text_degrades_to_image(self):
        got = fuse_probabilities(None, self.image, self.scalar)
        self.assertEqual(got, self.image)

    def test_neither_modality_is_an_error(self):
        with self.assertRaises(ValueError):
            fuse_probabilities(None, None, self.scalar)

    def test_label_order_mismatch_is_refused(self):
        """Silent permutation mislabels everything while looking healthy."""
        wrong_order = {"shopping": 0.6, "news": 0.3, "adult": 0.1}
        with self.assertRaises(ValueError) as caught:
            fuse_probabilities(wrong_order, self.image, self.scalar)
        self.assertIn("label order", str(caught.exception))

    def test_a_class_the_image_model_cannot_emit_does_not_crash(self):
        """This crashed every fused call in production.

        The shipped screenshot model covers 42 of the 47 classes -- five had too few
        screenshots to train on -- and `fuse_probabilities` indexed the image dict by all
        47, so `use_screenshots=True` raised KeyError('aggressive') for every domain.
        training/fuse.py had the projection; the inference path never got it.
        """
        weights = FusionWeights(text=(0.5, 1.0, 0.5), labels=LABELS)
        partial = {"news": 0.4, "adult": 0.6}  # no "shopping"
        got = fuse_probabilities(self.text, partial, weights)
        self.assertAlmostEqual(sum(got.values()), 1.0, places=6)
        self.assertEqual(set(got), set(LABELS))

    def test_pinning_an_uncovered_class_stops_it_being_shrunk(self):
        """Zero image mass at weight < 1 quietly halves a class the image model cannot see.

        `training/fuse.py` pins those classes to text-only for exactly this reason. The
        comparison is the point: the same inputs, differing only in whether `shopping`
        is pinned.
        """
        text = {"news": 0.2, "shopping": 0.45, "adult": 0.35}
        image = {"news": 0.9, "adult": 0.1}  # confident, and silent on shopping

        pinned = fuse_probabilities(
            text, image, FusionWeights(text=(0.5, 1.0, 0.5), labels=LABELS)
        )
        unpinned = fuse_probabilities(
            text, image, FusionWeights(text=(0.5, 0.5, 0.5), labels=LABELS)
        )
        # Halved before renormalisation (0.45 -> 0.225); 0.367 vs 0.225 after, because
        # the pinned blend sums to more than 1 before it is normalised.
        self.assertGreater(pinned["shopping"], unpinned["shopping"])
        self.assertAlmostEqual(pinned["shopping"], 0.367, places=2)
        self.assertAlmostEqual(unpinned["shopping"], 0.225, places=2)

    def test_a_different_class_count_is_refused(self):
        five = FusionWeights(text=(0.5,), labels=(*LABELS, "porn", "finance"))
        with self.assertRaises(ValueError):
            fuse_probabilities(self.text, self.image, five)

    def test_empty_weights_are_refused(self):
        with self.assertRaises(ValueError):
            FusionWeights(text=(), labels=LABELS).weight_for(0)


if __name__ == "__main__":
    unittest.main()
