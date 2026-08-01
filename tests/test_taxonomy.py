#!/usr/bin/env python

"""Tests for the Shallalist -> training-label mapping.

The mapping encodes one rule -- *is it visible in the page text?* -- and these
tests pin the three consequences of applying it, because each one was a
measured failure before it was a design decision:

* classes a page never states were producing 31% of evaluation errors,
* `recreation` was 98% travel-or-sports and could not be applied consistently,
* `porn`/`sex`/`models` competed for the same pages three ways.
"""

import unittest

from piedomains.training.taxonomy import EXCLUDED, map_category, target_classes


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

    def test_aggressive_is_excluded_for_want_of_data(self):
        """Not invisible -- just too rare to learn.

        9 held-out documents and recall 0.111, while the pages that did attract the
        label were politics and humor. macro-F1 weighted it the same as `parked`'s 378.
        """
        self.assertIsNone(map_category("aggressive"))
        self.assertIn("aggressive", EXCLUDED)


class TestSplitting(unittest.TestCase):
    """Parents that were never categories keep their children."""

    def test_recreation_keeps_its_children(self):
        """98% of `recreation` was travel or sports; the label meant nothing."""
        self.assertEqual(map_category("sports"), "sports")
        self.assertEqual(map_category("travel"), "travel")

    def test_hobby_keeps_its_children(self):
        self.assertEqual(map_category("pets"), "pets")
        self.assertEqual(map_category("gardening"), "gardening")

    def test_games_children_merge_despite_hobby_being_split(self):
        """A split parent does not protect a child that asks an unanswerable question.

        Whether a game is played online is a fact about delivery, not subject, and the
        page rarely says. It cost `hobby/games-misc` 34.4% of itself to
        `hobby/games-online` on held-out data -- the most expensive single distinction
        left in the taxonomy.

        Note the asymmetry, which a bulk rename got wrong once already: the *inputs* keep
        their `hobby/` prefix because they are raw Shallalist category names, fixed by the
        upstream data. Only the label we emit is flat.
        """
        self.assertEqual(map_category("hobby/games-online"), "games")
        self.assertEqual(map_category("hobby/games-misc"), "games")

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

    def test_delivery_mechanism_splits_collapse(self):
        """A web-only station and a broadcast one produce the same homepage.

        `webradio` lost 31.8% of itself to `radiotv` on held-out data. The split asks
        how the signal reaches the listener, which the page does not say.
        """
        self.assertEqual(map_category("webradio"), "radiotv")
        self.assertEqual(map_category("radiotv"), "radiotv")

    def test_legality_splits_collapse(self):
        """Warez is downloads the licence did not permit.

        That is a fact about the files, not about the text, and a warez site does not
        announce it. `warez` lost 30.8% of itself to `downloads`.
        """
        self.assertEqual(map_category("warez"), "downloads")
        self.assertEqual(map_category("downloads"), "downloads")


class TestTargetClasses(unittest.TestCase):
    """The resolved label set."""

    def test_deduplicates_and_sorts(self):
        got = target_classes(["porn", "sex", "models", "sports"])
        self.assertEqual(got, ["adult", "sports"])

    def test_excluded_classes_leave_no_trace(self):
        self.assertEqual(target_classes(["adv", "tracker", "spyware"]), [])

    def test_a_realistic_slice(self):
        """Raw Shallalist names in, flat sorted labels out."""
        got = target_classes(
            ["news", "shopping", "adv", "porn", "finance/banking", "recreation/travel"]
        )
        self.assertEqual(got, ["adult", "finance", "news", "shopping", "travel"])


class TestLeakGuard(unittest.TestCase):
    """`confusions.refuse_if_seen` must not clear what it cannot verify.

    The first version of this guard fell back to comparing the split directory's own
    train and test lists when a checkpoint carried no manifest. Those are disjoint by
    construction, so it passed a checkpoint that had trained on 78.9% of the split it was
    about to be scored on -- reproducing, inside the leak detector, the exact bug the
    leak detector exists to catch.
    """

    def setUp(self):
        """Build a split directory and two checkpoint directories."""
        import json
        import tempfile
        from pathlib import Path

        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "prepared"
        self.data.mkdir()
        for split, domains in (
            ("train", ["a.com", "b.com"]),
            ("test", ["c.com", "d.com"]),
        ):
            (self.data / f"{split}.jsonl").write_text(
                "\n".join(
                    json.dumps({"domain": d, "category": "news", "text": "x"})
                    for d in domains
                ),
                encoding="utf-8",
            )
        self.bare = self.tmp / "bare"
        self.bare.mkdir()
        self.honest = self.tmp / "honest"
        self.honest.mkdir()
        (self.honest / "train_domains.json").write_text(
            json.dumps(["c.com", "zzz.com"]), encoding="utf-8"
        )

    def tearDown(self):
        """Remove the temporary tree."""
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_manifest_refuses_rather_than_passing(self):
        """Absence of evidence is not evidence of disjointness."""
        from piedomains.training.confusions import refuse_if_seen

        with self.assertRaises(SystemExit) as caught:
            refuse_if_seen(self.bare, self.data)
        self.assertIn("train_domains.json", str(caught.exception))

    def test_no_manifest_can_be_overridden_explicitly(self):
        """The override exists, and requires being asked for."""
        from piedomains.training.confusions import refuse_if_seen

        refuse_if_seen(self.bare, self.data, assume_disjoint=True)

    def test_a_real_overlap_is_caught(self):
        """`c.com` is in both the manifest and the test split."""
        from piedomains.training.confusions import refuse_if_seen

        with self.assertRaises(SystemExit) as caught:
            refuse_if_seen(self.honest, self.data)
        self.assertIn("c.com", str(caught.exception))


class TestBenchmarkMapStaysInSync(unittest.TestCase):
    """The Curlie map must not key on labels the model cannot emit.

    This is a silent failure, which is why it is worth a test: a stale key does not
    raise, it just never fires, and the benchmark quietly scores fewer domains while
    still reporting a number. Merging `games-misc` and `games-online` into
    `games` orphaned two keys at exactly the moment the map was about to be used to
    judge the retrain.
    """

    def test_no_key_is_absent_from_the_label_set(self):
        from piedomains.constants import classes
        from piedomains.training.validate_curlie import TO_CURLIE

        orphans = sorted(set(TO_CURLIE) - set(classes))
        self.assertEqual(orphans, [], f"map keys no model can emit: {orphans}")

    def test_domains_with_no_site_have_no_curlie_topic(self):
        """`parked` and `unavailable` are unmappable on purpose, not by omission."""
        from piedomains.training.validate_curlie import TO_CURLIE

        self.assertNotIn("parked", TO_CURLIE)
        self.assertNotIn("unavailable", TO_CURLIE)


if __name__ == "__main__":
    unittest.main()
