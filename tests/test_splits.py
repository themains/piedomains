#!/usr/bin/env python

"""Tests for deterministic split assignment.

Three times a model in this project was scored on data it had trained on, because each
preparer shuffled its own list and the same domain landed in different splits on each side.
These pin the property that makes that impossible rather than merely detectable.
"""

import collections
import unittest

from piedomains.training.splits import SPLITS, is_held_out, split_of


class TestDeterminism(unittest.TestCase):
    """The same domain must always get the same split, everywhere, forever."""

    def test_repeated_calls_agree(self):
        self.assertEqual(split_of("cnn.com"), split_of("cnn.com"))

    def test_case_and_whitespace_do_not_matter(self):
        """`CNN.com` and `cnn.com` are the same domain and cannot land in different splits."""
        for variant in ("CNN.com", "  cnn.com  ", "CNN.COM", "cnn.com\n"):
            with self.subTest(variant=variant):
                self.assertEqual(split_of(variant), split_of("cnn.com"))

    def test_survives_a_fresh_process(self):
        """Python salts str hashing per process, which is why this uses SHA-256.

        Using the builtin `hash()` would make the split differ between runs -- exactly the
        instability being removed.
        """
        import subprocess
        import sys

        got = subprocess.run(
            [
                sys.executable,
                "-c",
                "from piedomains.training.splits import split_of; print(split_of('cnn.com'))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(got, split_of("cnn.com"))

    def test_only_ever_returns_a_known_split(self):
        for i in range(1000):
            self.assertIn(split_of(f"{i}.example.com"), SPLITS)


class TestProportions(unittest.TestCase):
    """Roughly 80/10/10, with the variance that hashing implies."""

    def test_large_sample_lands_near_the_target(self):
        counts = collections.Counter(
            split_of(f"{i}.example.com") for i in range(50_000)
        )
        total = sum(counts.values())
        self.assertAlmostEqual(counts["train"] / total, 0.80, delta=0.01)
        self.assertAlmostEqual(counts["val"] / total, 0.10, delta=0.01)
        self.assertAlmostEqual(counts["test"] / total, 0.10, delta=0.01)


class TestTheGuaranteeThatMatters(unittest.TestCase):
    """Two independent corpora cannot disagree about a shared domain.

    This is the whole point. `prepare_text.py` and `prepare_images.py` process different
    lists of different lengths, and previously each shuffled its own -- so ~80% of the
    domains held out from the text model were in the image model's training set, and
    image-only read 0.768 where the honest figure was 0.429.
    """

    def test_two_different_corpora_agree_on_shared_domains(self):
        text_corpus = [f"{i}.example.com" for i in range(5_000)]
        # A different collection, different order, different size -- as the screenshot
        # corpus genuinely is (46,754 documents against 44,712 screenshots).
        image_corpus = [f"{i}.example.com" for i in range(2_500, 6_000)][::-1]

        text_assignment = {d: split_of(d) for d in text_corpus}
        image_assignment = {d: split_of(d) for d in image_corpus}
        shared = set(text_assignment) & set(image_assignment)
        self.assertGreater(len(shared), 2_000)

        disagreements = [d for d in shared if text_assignment[d] != image_assignment[d]]
        self.assertEqual(disagreements, [])

    def test_held_out_is_the_complement_of_train(self):
        for i in range(500):
            domain = f"{i}.example.com"
            with self.subTest(domain=domain):
                self.assertEqual(is_held_out(domain), split_of(domain) != "train")


if __name__ == "__main__":
    unittest.main()
