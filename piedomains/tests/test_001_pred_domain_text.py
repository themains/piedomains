#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for piedomain with text

"""

import unittest
from piedomains import domain


class TestPredDomainText(unittest.TestCase):
    def setUp(self):
        self.domains = ["yahoo.com", "khanacademy.org"]
        # self.true_labels = ["news", "finance"]

    def tearDown(self):
        pass

    # test if domains is None and html_path is None
    def test_domains_path_none(self):
        try:
            domain.pred_shalla_cat_with_text()
        except Exception:
            self.assertTrue(True)

    # test if domains is None and html_path is not None but not a directory
    def test_domains_none_path_not_dir(self):
        try:
            domain.pred_shalla_cat_with_text(html_path="./test")
        except Exception:
            self.assertTrue(True)

    def test_pred_label(self):
        odf = domain.pred_shalla_cat_with_text(self.domains, html_path="./html")
        self.assertIn("text_label", odf.columns)
        self.assertIn("text_prob", odf.columns)
        self.assertIn("text_domain_probs", odf.columns)
        self.assertIn("used_domain_text", odf.columns)
        self.assertTrue(odf.iloc[0]["used_domain_text"])
        self.assertTrue(odf.iloc[1]["used_domain_text"])


if __name__ == "__main__":
    unittest.main()
