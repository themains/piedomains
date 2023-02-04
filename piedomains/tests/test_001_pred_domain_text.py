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

    def test_pred_label(self):
        odf = domain.pred_shalla_cat_with_text(self.domains, html_path="./html")
        self.assertIn("text_label", odf.columns)
        self.assertIn("text_prob", odf.columns)
        self.assertIn("text_domain_probs", odf.columns)
        self.assertIn("used_domain_text", odf.columns)


if __name__ == "__main__":
    unittest.main()