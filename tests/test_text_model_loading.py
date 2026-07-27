#!/usr/bin/env python

"""Tests for resolving the text model and its sidecar files.

The sidecar tests exist because of a real bug: `labels.json` and
`calibration.json` were read with a bare ``Path(source) / filename`` check,
which is correct for a local directory and silently wrong for a Hub repo id.
The model loaded, logged success, and ran with temperature 1.0 -- so the
calibration this release is largely about did nothing at all. The failure was
invisible because every other signal said the model was fine.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from piedomains.text import (
    DEFAULT_TEXT_MODEL,
    _read_labels,
    _read_temperature,
    _sidecar,
    resolve_text_model,
)


class TestModelResolution(unittest.TestCase):
    """Which checkpoint gets loaded."""

    def test_defaults_to_the_published_model(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("PIEDOMAINS_TEXT_MODEL", None)
            self.assertEqual(resolve_text_model(), DEFAULT_TEXT_MODEL)

    def test_environment_override_wins(self):
        """A freshly trained model has to be loadable before it is published."""
        with patch.dict("os.environ", {"PIEDOMAINS_TEXT_MODEL": "models/text-v3"}):
            self.assertEqual(resolve_text_model(), "models/text-v3")


class TestSidecarFiles(unittest.TestCase):
    """Reading files that sit next to the weights, locally or on the Hub."""

    def test_reads_from_a_local_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "calibration.json").write_text(
                json.dumps({"temperature": 2.5})
            )
            self.assertEqual(_read_temperature(tmp), 2.5)

    def test_missing_file_in_a_local_directory_is_not_a_hub_lookup(self):
        """A local dir without the file must not hit the network."""
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("huggingface_hub.hf_hub_download") as download,
        ):
            self.assertIsNone(_sidecar(tmp, "calibration.json"))
            download.assert_not_called()

    def test_reads_from_a_hub_repo_id(self):
        """The bug: a repo id is not a path, so the local check always missed."""
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(tmp) / "calibration.json"
            downloaded.write_text(json.dumps({"temperature": 3.4161}))
            with patch(
                "huggingface_hub.hf_hub_download", return_value=str(downloaded)
            ) as download:
                self.assertAlmostEqual(_read_temperature("owner/model"), 3.4161)
            download.assert_called_once()

    def test_absent_calibration_falls_back_to_no_scaling(self):
        with patch("huggingface_hub.hf_hub_download", side_effect=OSError("404")):
            self.assertEqual(_read_temperature("owner/model"), 1.0)


class TestLabels(unittest.TestCase):
    """Class order must come from the checkpoint, not a module constant."""

    def test_labels_json_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "labels.json").write_text(json.dumps(["news", "shopping"]))
            config = SimpleNamespace(id2label={0: "wrong", 1: "alsowrong"})
            self.assertEqual(_read_labels(tmp, config), ["news", "shopping"])

    def test_falls_back_to_id2label_in_index_order(self):
        """Order matters: getting it wrong renames every class silently."""
        with tempfile.TemporaryDirectory() as tmp:
            config = SimpleNamespace(id2label={1: "shopping", 0: "news", 2: "porn"})
            self.assertEqual(_read_labels(tmp, config), ["news", "shopping", "porn"])

    def test_placeholder_id2label_is_rejected(self):
        """`LABEL_0` means the checkpoint carries no real names."""
        from piedomains.constants import classes

        with tempfile.TemporaryDirectory() as tmp:
            config = SimpleNamespace(id2label={0: "LABEL_0", 1: "LABEL_1"})
            self.assertEqual(_read_labels(tmp, config), list(classes))


if __name__ == "__main__":
    unittest.main()
