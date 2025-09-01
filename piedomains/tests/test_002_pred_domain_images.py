#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for piedomain with images

"""

import unittest
import pytest
from piedomains import domain


class TestPredDomainImages(unittest.TestCase):
    def setUp(self):
        self.domains = ["yahoo.com", "khanacademy.org"]
        # self.true_labels = ["news", "finance"]

    def tearDown(self):
        pass

    # test if domains is None and image_path is None
    def test_domains_path_none(self):
        try:
            domain.pred_shalla_cat_with_images()
        except Exception as e:
            self.assertTrue(str(e) == "Provide list of Domains, or for offline provide image_path")

    # test if domains is None and image_path is not None but not a directory
    def test_domains_none_path_not_dir(self):
        try:
            domain.pred_shalla_cat_with_images(image_path="./test")
        except Exception as e:
            self.assertTrue(str(e) == "./test does not exist")

    @pytest.mark.ml
    def test_pred_label(self):
        import os
        test_dir = os.path.dirname(__file__)
        image_path = os.path.join(test_dir, "images")
        odf = domain.pred_shalla_cat_with_images(image_path=image_path)
        self.assertIn("image_label", odf.columns)
        self.assertIn("image_prob", odf.columns)
        self.assertIn("image_domain_probs", odf.columns)
        self.assertIn("used_domain_screenshot", odf.columns)
        self.assertTrue(odf.iloc[0]["used_domain_screenshot"])
        self.assertTrue(odf.iloc[1]["used_domain_screenshot"])


if __name__ == "__main__":
    unittest.main()
