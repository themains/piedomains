#!/usr/bin/env python

"""Tests for multi-label selection.

The label set is not mutually exclusive: `shopping` says what a site does, `automobile`
says what it is about, and a car dealership is both. Reporting only the argmax throws away
a true answer whenever two axes overlap, so these pin the behaviour that stops it.
"""

import unittest

from piedomains.labels import project, top_labels


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


class TestProjection(unittest.TestCase):
    """Mapping a checkpoint's labels onto the space the package documents.

    A checkpoint outlives its taxonomy. The shipped model emits 47 prefixed labels while
    `constants.classes` documents 44 flat ones, so without this the package describes a
    vocabulary its own model cannot speak.
    """

    def test_merged_classes_are_grouped_not_renamed(self):
        """The whole point: two sources, one label, both indices kept.

        Renaming dict keys instead would silently keep whichever source came last and
        throw the other's probability away -- a wrong model that looks like a working one.
        """
        names, groups = project(["news", "webradio", "radiotv"])
        self.assertEqual(names, ["news", "radiotv"])
        self.assertEqual(groups, [[0], [1, 2]])

    def test_all_three_merges(self):
        for sources, expected in (
            (["hobby/games-misc", "hobby/games-online"], "games"),
            (["radiotv", "webradio"], "radiotv"),
            (["downloads", "warez"], "downloads"),
        ):
            with self.subTest(expected=expected):
                names, groups = project(sources)
                self.assertEqual(names, [expected])
                self.assertEqual(groups, [[0, 1]])

    def test_split_parents_flatten(self):
        """`hobby/` and `recreation/` are filing structure, not part of the answer."""
        names, _ = project(
            ["hobby/cooking", "recreation/sports", "recreation/travel", "hobby/pets"]
        )
        self.assertEqual(names, ["cooking", "sports", "travel", "pets"])

    def test_excluded_classes_are_dropped(self):
        """`aggressive` has no target: 9 training documents and recall 0.111."""
        names, groups = project(["news", "aggressive", "shopping"])
        self.assertEqual(names, ["news", "shopping"])
        self.assertEqual(groups, [[0], [2]])

    def test_current_labels_are_untouched(self):
        """Projecting an already-current label set must be the identity."""
        current = ["news", "shopping", "cooking", "sports", "parked"]
        names, groups = project(current)
        self.assertEqual(names, current)
        self.assertEqual(groups, [[i] for i in range(len(current))])

    def test_the_shipped_checkpoint_projects_into_the_documented_space(self):
        """Nothing invented, and only `unavailable` is beyond this checkpoint's reach."""
        import json
        from pathlib import Path

        from piedomains.constants import classes

        manifest = Path("models/text-v5/labels.json")
        if not manifest.exists():
            self.skipTest("local training checkpoint not present")
        names, groups = project(json.loads(manifest.read_text()))
        self.assertEqual(sorted(set(names) - set(classes)), [])
        self.assertEqual(sorted(set(classes) - set(names)), ["unavailable"])
        self.assertEqual(
            sum(len(g) for g in groups), 46
        )  # 47 sources less `aggressive`

    def test_probability_mass_is_conserved_under_grouping(self):
        """Summing the groups must not lose or duplicate mass."""
        labels = ["news", "webradio", "radiotv", "aggressive"]
        probs = [0.5, 0.2, 0.2, 0.1]
        _, groups = project(labels)
        summed = [sum(probs[i] for i in g) for g in groups]
        self.assertEqual(len(summed), 2)
        self.assertAlmostEqual(summed[0], 0.5)
        self.assertAlmostEqual(summed[1], 0.4)  # webradio + radiotv, not one of them
        # `aggressive` contributes nothing; the caller renormalises the rest back to 1.
        self.assertAlmostEqual(sum(summed), 0.9)

    def test_empty_input(self):
        self.assertEqual(project([]), ([], []))


if __name__ == "__main__":
    unittest.main()
