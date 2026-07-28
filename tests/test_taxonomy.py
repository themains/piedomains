#!/usr/bin/env python

"""Tests for the Shallalist -> training-label mapping.

The mapping encodes one rule -- *is it visible in the page text?* -- and these
tests pin the three consequences of applying it, because each one was a
measured failure before it was a design decision:

* classes a page never states were producing 31% of evaluation errors,
* `recreation` was 98% travel-or-sports and could not be applied consistently,
* `porn`/`sex`/`models` competed for the same pages three ways.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from taxonomy import EXCLUDED, map_category, target_classes


class TestExcluded(unittest.TestCase):
    """Classes a page does not state are not in the label space."""

    def test_infrastructure_classes_are_excluded(self):
        """These describe who runs a site, not what it says.

        A homepage selling handmade goods reads identically whether or not the
        operator also runs trackers -- which is how `etsy` came back `spyware`.
        """
        for category in ("adv", "tracker", "spyware"):
            with self.subTest(category=category):
                self.assertIsNone(map_category(category))

    def test_redirector_is_excluded_as_an_outcome(self):
        """A redirector page has no content by construction."""
        self.assertIsNone(map_category("redirector"))
        self.assertIn("redirector", EXCLUDED)

    def test_original_by_name_drops_still_apply(self):
        for category in ("chat", "hacking", "webtv"):
            with self.subTest(category=category):
                self.assertIsNone(map_category(category))


class TestSplitting(unittest.TestCase):
    """Parents that were never categories keep their children."""

    def test_recreation_keeps_its_children(self):
        """98% of `recreation` was travel or sports; the label meant nothing."""
        self.assertEqual(map_category("recreation/sports"), "recreation/sports")
        self.assertEqual(map_category("recreation/travel"), "recreation/travel")

    def test_hobby_keeps_its_children(self):
        self.assertEqual(map_category("hobby/pets"), "hobby/pets")
        self.assertEqual(map_category("hobby/games-online"), "hobby/games-online")

    def test_other_parents_still_collapse(self):
        """Splitting these would produce classes too small to learn."""
        self.assertEqual(map_category("finance/banking"), "finance")
        self.assertEqual(map_category("automobile/cars"), "automobile")
        self.assertEqual(map_category("science/astronomy"), "science")
        self.assertEqual(map_category("education/schools"), "education")

    def test_real_estate_is_promoted_out_of_finance(self):
        """IAB, Cloudflare and WebOrganizer all treat it as top-level."""
        self.assertEqual(map_category("finance/realestate"), "realestate")


class TestMerging(unittest.TestCase):
    """Categories describing one thing become one label."""

    def test_adult_categories_merge(self):
        for category in ("porn", "sex", "models", "sex/lingerie"):
            with self.subTest(category=category):
                self.assertEqual(map_category(category), "adult")

    def test_sexual_health_is_not_adult_content(self):
        """Shallalist files sex/education under `sex`; inheriting that is wrong.

        Conflating sexual health with adult content is the failure mode that
        makes filters block the resources people most need. Shallalist's own
        description concedes the category "can be misdetected as porn".
        """
        self.assertEqual(map_category("sex/education"), "education")
        self.assertNotEqual(map_category("sex/education"), "adult")


class TestTargetClasses(unittest.TestCase):
    """The resolved label set."""

    def test_deduplicates_and_sorts(self):
        got = target_classes(["porn", "sex", "models", "recreation/sports"])
        self.assertEqual(got, ["adult", "recreation/sports"])

    def test_excluded_classes_leave_no_trace(self):
        self.assertEqual(target_classes(["adv", "tracker", "spyware"]), [])

    def test_a_realistic_slice(self):
        got = target_classes(
            ["news", "shopping", "adv", "porn", "finance/banking", "recreation/travel"]
        )
        self.assertEqual(
            got, ["adult", "finance", "news", "recreation/travel", "shopping"]
        )


if __name__ == "__main__":
    unittest.main()
