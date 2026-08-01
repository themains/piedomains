#!/usr/bin/env python

"""Tests for multi-label selection.

The label set is not mutually exclusive: `shopping` says what a site does, `automobile`
says what it is about, and a car dealership is both. Reporting only the argmax throws away
a true answer whenever two axes overlap, so these pin the behaviour that stops it.
"""

import unittest

from piedomains.labels import top_labels


class TestTopLabels(unittest.TestCase):
    """Selecting which labels to report from a distribution."""

    def test_returns_everything_above_the_threshold(self):
        scores = {"automobile": 0.52, "shopping": 0.31, "news": 0.09, "adult": 0.08}
        got = top_labels(scores, 0.10)
        self.assertEqual([e["category"] for e in got], ["automobile", "shopping"])

    def test_ordered_by_probability_descending(self):
        scores = {"a": 0.2, "b": 0.5, "c": 0.3}
        got = top_labels(scores, 0.10)
        self.assertEqual([e["category"] for e in got], ["b", "c", "a"])
        probs = [e["probability"] for e in got]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_argmax_survives_even_below_the_threshold(self):
        """A classified row reporting nothing is a worse answer than a weak one.

        With 44 classes a genuinely uncertain page can leave every probability under the
        floor. The caller can read the number and decide; an empty list would just hide
        that the model had an opinion.
        """
        scores = {"a": 0.04, "b": 0.03, "c": 0.02}
        got = top_labels(scores, 0.10)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["category"], "a")

    def test_first_entry_is_always_the_argmax(self):
        scores = {"x": 0.11, "y": 0.7, "z": 0.19}
        self.assertEqual(top_labels(scores, 0.10)[0]["category"], "y")

    def test_threshold_is_inclusive(self):
        got = top_labels({"a": 0.6, "b": 0.10}, 0.10)
        self.assertEqual(len(got), 2)

    def test_empty_distribution_yields_nothing(self):
        self.assertEqual(top_labels({}, 0.10), [])

    def test_a_high_threshold_collapses_to_one_label(self):
        scores = {"a": 0.4, "b": 0.35, "c": 0.25}
        self.assertEqual(len(top_labels(scores, 0.9)), 1)

    def test_probabilities_are_carried_through_unchanged(self):
        """No renormalising: these must match `raw_predictions` exactly.

        Both are built from the same distribution, and a caller comparing the two would
        be right to treat any difference as a bug.
        """
        scores = {"a": 0.62, "b": 0.28}
        got = top_labels(scores, 0.10)
        self.assertAlmostEqual(got[0]["probability"], 0.62)
        self.assertAlmostEqual(got[1]["probability"], 0.28)


class TestConfiguredThreshold(unittest.TestCase):
    """The floor is configuration, not a constant."""

    def test_default_is_the_measured_choice(self):
        """0.10 buys 86.6% recall at 1.35 labels per domain on held-out data."""
        from piedomains.config import get_config

        self.assertEqual(get_config().get("multilabel_threshold"), 0.10)


if __name__ == "__main__":
    unittest.main()
