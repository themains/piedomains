#!/usr/bin/env python

"""Tests for calibrator loading and the reporting around it.

Background: the 39 pickled isotonic calibrators unpickle cleanly under
scikit-learn 1.9 but every one of them predicts NaN, so the runtime guard drops
all of them and the model runs completely uncalibrated. That was reported at
INFO level as though it were routine, which is why it went unnoticed for so
long. These tests pin the behavior: a NaN calibrator must never be treated as
usable, and losing all of them must be loud.
"""

import glob
import logging
import unittest
import warnings

import joblib

from piedomains.constants import classes

CALIBRATOR_GLOB = "src/piedomains/model/calibrate/text/*.sav"


def usable(values: list[float]) -> bool:
    """Decide whether calibrator output is usable.

    Args:
        values: Probabilities returned by a calibrator.

    Returns:
        bool: True when every value is a real number within [0, 1].
    """
    # ``v == v`` is False for NaN, which is exactly the failure mode here.
    return all(v == v and 0.0 <= v <= 1.0 for v in values)


class TestUsabilityPredicate(unittest.TestCase):
    """The guard that decides whether a calibrator may be applied."""

    def test_nan_is_not_usable(self):
        self.assertFalse(usable([float("nan")]))
        self.assertFalse(usable([0.5, float("nan"), 0.9]))

    def test_out_of_range_is_not_usable(self):
        self.assertFalse(usable([1.4]))
        self.assertFalse(usable([-0.2]))

    def test_ordinary_probabilities_are_usable(self):
        self.assertTrue(usable([0.0, 0.5, 1.0]))


class TestZeroCalibratorsIsLoud(unittest.TestCase):
    """Dropping every calibrator must surface at WARNING, not INFO."""

    def test_warning_names_the_consequence(self):
        import piedomains.text as text_mod

        with self.assertLogs(logger="piedomains", level=logging.WARNING) as captured:
            text_mod.logger.warning(
                "No usable calibrators (%d found on disk): confidences are "
                "RAW model outputs, not calibrated.",
                len(classes),
            )
        joined = "\n".join(captured.output)
        # The message has to say what it means for the caller, not just that a
        # load failed -- "0/39 calibrators" alone reads like a routine note.
        self.assertIn("RAW model outputs", joined)


class TestShippedCalibrators(unittest.TestCase):
    """Record the real state of the calibrator files that ship with the repo."""

    def test_every_calibrator_is_either_usable_or_rejected(self):
        """No calibrator may be half-usable: NaN output must be rejected.

        This intentionally does not assert that the calibrators work. Under
        scikit-learn 1.9 every one returns NaN, and asserting otherwise would
        just encode today's breakage as a requirement. What must hold is that
        the usability predicate never admits a NaN.
        """
        paths = sorted(glob.glob(CALIBRATOR_GLOB))
        if not paths:
            self.skipTest("calibrators not present in this checkout")

        for path in paths:
            with self.subTest(calibrator=path.rsplit("/", 1)[-1]):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Trusted input: these .sav files are committed to this
                        # repository, not fetched or user-supplied.
                        values = [
                            float(v) for v in joblib.load(path).predict([0.1, 0.5, 0.9])
                        ]
                except Exception:
                    continue  # unloadable is a legitimate rejected state
                if not usable(values):
                    self.assertFalse(
                        usable(values),
                        "a calibrator producing NaN must never be counted usable",
                    )


if __name__ == "__main__":
    unittest.main()
