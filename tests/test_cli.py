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

    A figure nobody can re-run is a figure taken on faith, so they are a subpackage that
    installs with the library rather than living only in the repository. These tests fail
    if that is ever undone.
    """

    def test_it_is_a_real_subpackage_not_a_build_time_copy(self):
        """Importing it must not require torch, or `import piedomains` breaks for users."""
        import piedomains.training

        self.assertTrue(hasattr(piedomains.training, "__path__"))

    def test_every_script_is_an_importable_module(self):
        """`python -m piedomains.training.x` is the documented way to run these."""
        import importlib

        for name in ("metrics", "taxonomy", "evaluate", "fuse"):
            with self.subTest(module=name):
                importlib.import_module(f"piedomains.training.{name}")

    def test_the_directory_is_found(self):
        self.assertTrue(training_scripts_dir().is_dir())

    def test_the_scripts_behind_every_published_number_are_present(self):
        here = training_scripts_dir()
        for name in (
            "train_text.py",  # the text model
            "train_image.py",  # the screenshot model
            "prepare_text.py",  # corpus -> splits
            "prepare_images.py",  # tarballs -> 224px
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

    def test_kaggle_model_override_carries_its_immutable_revision(self):
        """A custom Hub repo must not inherit another model's commit SHA."""
        from piedomains.training.kaggle.push_kernel import (
            DEFAULT_SCRIPT,
            build_source,
        )

        source = build_source(
            None,
            DEFAULT_SCRIPT,
            image_model=("example/model", "deadbeef"),
        )
        self.assertIn(
            "IMAGE_MODEL: tuple[str, str] | None = ('example/model', 'deadbeef')",
            source,
        )

    def test_kaggle_model_override_requires_a_revision(self):
        """The kernel generator refuses an unpinned Hub checkpoint."""
        from piedomains.training.kaggle.push_kernel import main as push_kernel_main

        with self.assertRaises(SystemExit) as caught:
            push_kernel_main(["--image-model", "example/model", "--dry-run"])
        self.assertEqual(caught.exception.code, 2)

    def test_missing_scripts_raise_rather_than_returning_a_dead_path(self):
        from unittest.mock import patch

        with (
            patch.object(Path, "exists", return_value=False),
            self.assertRaises(SystemExit) as caught,
        ):
            training_scripts_dir()
        self.assertIn("training scripts not found", str(caught.exception))

    def test_no_sys_path_mutation_survives(self):
        """The scripts imported each other through sys.path before they were a package.

        That resolved by accident of process state: it worked when run as a script from
        the right directory and failed as an import, which is how `fuse.py` ended up with
        two function-local imports that would have broken on install.
        """
        for script in training_scripts_dir().rglob("*.py"):
            with self.subTest(script=script.name):
                self.assertNotIn("sys.path.insert", script.read_text(encoding="utf-8"))


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
        self.assertEqual(args.output, "text")

    def test_the_default_method_is_text_not_combined(self):
        """Screenshots are opt-in, and the CLI has to agree with the library.

        Fusing gains +0.001 macro-F1 over text alone on 1,742 paired domains, with a
        fitted text weight of 0.973. Defaulting to `combined` would load a 350MB vision
        model on every invocation to buy noise.
        """
        self.assertEqual(build_parser().parse_args(["x.com"]).method, "text")
        self.assertEqual(
            build_parser().parse_args(["x.com", "--method", "combined"]).method,
            "combined",
        )


if __name__ == "__main__":
    unittest.main()
