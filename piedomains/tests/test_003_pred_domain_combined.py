#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for piedomain with text and images

"""

import unittest
from piedomains import domain


class TestPredDomainCombined(unittest.TestCase):
    def setUp(self):
        self.domains = ["yahoo.com", "khanacademy.org"]
        # self.true_labels = ["news", "finance"]

    def tearDown(self):
        pass

    def test_pred_label(self):
        odf = domain.pred_shalla_cat(self.domains, html_path="./html", image_path="./images")
        self.assertIn("text_label", odf.columns)
        self.assertIn("text_prob", odf.columns)
        self.assertIn("text_domain_probs", odf.columns)
        self.assertIn("used_domain_text", odf.columns)
        self.assertIn("image_label", odf.columns)
        self.assertIn("image_prob", odf.columns)
        self.assertIn("image_domain_probs", odf.columns)
        self.assertIn("used_domain_screenshot", odf.columns)
        self.assertIn("label", odf.columns)
        self.assertIn("label_prob", odf.columns)
        self.assertIn("combined_domain_probs", odf.columns)


if __name__ == "__main__":
    unittest.main()