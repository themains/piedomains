#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for piedomain with images
    
"""

import unittest
from piedomains import domain


class TestPredDomainImages(unittest.TestCase):
    def setUp(self):
        self.domains = ["yahoo.com", "khanacademy.org"]
        # self.true_labels = ["news", "finance"]

    def tearDown(self):
        pass

    def test_pred_label(self):
        odf = domain.pred_shalla_cat_with_images(self.domains, image_path="./images")
        self.assertIn("image_label", odf.columns)
        self.assertIn("image_prob", odf.columns)
        self.assertIn("image_domain_probs", odf.columns)
        self.assertIn("used_domain_screenshot", odf.columns)
        # self.assertTrue(odf.iloc[0]["pred_label"] == self.true_labels[0])
        # self.assertTrue(odf.iloc[1]["pred_label"] == self.true_labels[1])
        # self.assertTrue(odf.iloc[2]["used_domain_content"] == False)


if __name__ == "__main__":
    unittest.main()
