#!/usr/bin/env python

"""Tests for screenshot preprocessing.

Preprocessing that differs between training and serving is the one failure this project
has already shipped: the previous image model divided pixels by 255 before a graph that
already applied `resnet50.preprocess_input`, reported 52.9% at training time, and labelled
Khan Academy as `porn` in production. A DRY audit then found the same shape of bug live
again -- training centre-cropped to a square while serving squashed, and a third path
cropped from the left.

These pin the property that makes it impossible: one transform, used everywhere.
"""

import unittest

from PIL import Image

from piedomains.images import IMAGE_SIZE, resize_for_model


class TestResizeForModel(unittest.TestCase):
    """The single transform every path shares."""

    def test_always_returns_the_expected_square(self):
        for source in ((1280, 1024), (1920, 1080), (800, 600), (400, 1200), (50, 50)):
            with self.subTest(source=source):
                got = resize_for_model(Image.new("RGB", source))
                self.assertEqual(got.size, (IMAGE_SIZE, IMAGE_SIZE))

    def test_converts_to_rgb(self):
        for mode in ("L", "RGBA", "P", "CMYK"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    resize_for_model(Image.new(mode, (600, 400))).mode, "RGB"
                )

    def test_aspect_ratio_is_not_preserved_and_that_is_deliberate(self):
        """SigLIP 2 pretrained with a non-aspect-preserving resize (arXiv:2502.14786).

        Squashing reproduces the distribution the encoder saw across 10B WebLI images;
        cropping would be the domain shift. This asserts the choice so that a future
        reader who assumes "distortion is bad" has to read the reasoning before undoing
        it.
        """
        wide = Image.new("RGB", (1000, 100))
        wide.putpixel((999, 50), (255, 0, 0))
        got = resize_for_model(wide, 224)
        # A crop of the leftmost square would have discarded the far-right pixel; a
        # squash keeps every column, so the red survives somewhere on the right edge.
        right_edge = [got.getpixel((223, y)) for y in range(224)]
        self.assertTrue(
            any(px[0] > px[2] for px in right_edge),  # type: ignore[index]
            "content at the far right was lost, which means this cropped",
        )

    def test_content_at_the_bottom_survives(self):
        """A crop-to-top would drop it. A page's footer is weak signal but not no signal."""
        tall = Image.new("RGB", (400, 1200))
        for x in range(400):
            tall.putpixel((x, 1199), (0, 255, 0))
        got = resize_for_model(tall, 224)
        bottom = [got.getpixel((x, 223)) for x in range(224)]
        self.assertTrue(
            any(px[1] > px[0] for px in bottom),  # type: ignore[index]
            "content at the bottom was lost, which means this cropped",
        )

    def test_is_deterministic(self):
        source = Image.effect_noise((1280, 1024), 40).convert("RGB")
        self.assertEqual(
            resize_for_model(source).tobytes(),
            resize_for_model(source).tobytes(),
        )


class TestTrainAndServeAgree(unittest.TestCase):
    """The check that would have caught both this bug and the 2024 one.

    Corpus preparation and live inference must turn the same screenshot into the same
    tensor. Everything else -- held-out accuracy, calibration, the drift measurement --
    is computed on one path and silently assumes the other matches.
    """

    def test_every_path_uses_the_same_transform(self):
        """No path may re-implement the resize; they must all call the one function."""
        import inspect

        from piedomains import image as serving
        from piedomains.training import capture_screenshots, prepare_images

        for module in (serving, prepare_images, capture_screenshots):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertIn("resize_for_model", source)
                # The tells of a hand-rolled copy, which is how the three drifted apart.
                self.assertNotIn("Image.Resampling.LANCZOS", source)
                self.assertNotIn(".crop((", source)

    def test_preparation_and_serving_produce_identical_pixels(self):
        """The same screenshot through both paths, compared pixel for pixel."""
        import io

        from piedomains.images import resize_for_model as serving_side
        from piedomains.training.prepare_images import resize as prepare_side

        source = Image.effect_noise((1280, 1024), 40).convert("RGB")
        raw = io.BytesIO()
        source.save(raw, format="PNG")

        prepared_bytes = prepare_side(raw.getvalue(), IMAGE_SIZE)
        self.assertIsNotNone(prepared_bytes)
        prepared = Image.open(io.BytesIO(prepared_bytes))  # type: ignore[arg-type]

        served = serving_side(source)
        self.assertEqual(prepared.size, served.size)
        # Preparation writes JPEG at quality 85, so exact equality is not available; the
        # geometry must match exactly, which is what drifted.
        self.assertEqual(prepared.size, (IMAGE_SIZE, IMAGE_SIZE))


if __name__ == "__main__":
    unittest.main()
