#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for piedomain with text

"""

import unittest
import pytest
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
        except Exception as e:
            self.assertTrue(str(e) == "Provide list of Domains, or for offline provide html_path")

    # test if domains is None and html_path is not None but not a directory
    def test_domains_none_path_not_dir(self):
        try:
            domain.pred_shalla_cat_with_text(html_path="./test")
        except Exception as e:
            self.assertTrue(str(e) == "./test does not exist")

    @pytest.mark.ml
    def test_pred_label(self):
        import os
        test_dir = os.path.dirname(__file__)
        html_path = os.path.join(test_dir, "html")
        odf = domain.pred_shalla_cat_with_text(html_path=html_path)
        self.assertIn("text_label", odf.columns)
        self.assertIn("text_prob", odf.columns)
        self.assertIn("text_domain_probs", odf.columns)
        self.assertIn("used_domain_text", odf.columns)
        self.assertTrue(odf.iloc[0]["used_domain_text"])
        self.assertTrue(odf.iloc[1]["used_domain_text"])


if __name__ == "__main__":
    unittest.main()
