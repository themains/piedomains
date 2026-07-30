#!/usr/bin/env python

"""Tests for the console script and the bundled training scripts.

The console script has its own history worth guarding: ``classify_domains`` was declared
in ``pyproject.toml`` pointing at ``piedomains.domain:main``, a module that has never
existed, so the entry point was broken in every published release until 0.7.0.
"""

import unittest
from pathlib import Path

from piedomains.cli import build_parser, main, training_scripts_dir


class TestTrainingScriptsShip(unittest.TestCase):
    """Every accuracy number in the README comes out of these scripts.

    A figure nobody can re-run is a figure taken on faith, so they install alongside the
    package rather than living only in the repository. ``pyproject.toml`` force-includes
    ``training/`` into the wheel; these tests fail if that is ever dropped.
    """

    def test_the_directory_is_found(self):
        self.assertTrue(training_scripts_dir().is_dir())

    def test_the_scripts_behind_every_published_number_are_present(self):
        here = training_scripts_dir()
        for name in (
            "train_text.py",  # the text model
            "train_image.py",  # the screenshot model
            "prepare_text.py",  # corpus -> splits
            "prepare_images.py",  # tarballs -> 224px, and --respect-splits
            "calibrate.py",  # temperature scaling
            "fuse.py",  # the late-fusion measurement and its gate
            "evaluate.py",  # tests/eval/labels.csv
            "validate_curlie.py",  # the independent out-of-sample check
        ):
            with self.subTest(script=name):
                self.assertTrue((here / name).exists(), f"{name} does not ship")

    def test_the_kaggle_kernels_ship_too(self):
        """The image model cannot be reproduced on a laptop; the kernels are the recipe."""
        kaggle = training_scripts_dir() / "kaggle"
        self.assertTrue((kaggle / "train_image_kaggle.py").exists())
        self.assertTrue((kaggle / "push_kernel.py").exists())

    def test_missing_scripts_raise_rather_than_returning_a_dead_path(self):
        from unittest.mock import patch

        with (
            patch.object(Path, "exists", return_value=False),
            self.assertRaises(SystemExit) as caught,
        ):
            training_scripts_dir()
        self.assertIn("training scripts not found", str(caught.exception))


class TestCli(unittest.TestCase):
    """The console script itself."""

    def test_training_scripts_flag_prints_a_real_directory(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["--training-scripts"])
        self.assertEqual(code, 0)
        self.assertTrue(Path(buffer.getvalue().strip()).is_dir())

    def test_no_domains_is_an_error_not_a_silent_success(self):
        with self.assertRaises(SystemExit):
            main([])

    def test_parser_defaults(self):
        args = build_parser().parse_args(["example.com"])
        self.assertEqual(args.domains, ["example.com"])
        self.assertEqual(args.method, "combined")
        self.assertEqual(args.output, "text")


if __name__ == "__main__":
    unittest.main()
