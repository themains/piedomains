#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for piedomain

"""

import unittest
from piedomains import domain


class TestPredDomain(unittest.TestCase):
    def setUp(self):
        self.domains = [
            "forbes.com",
            "marketwatch.com",
        ]
        self.true_labels = ["news", "finance"]

    def tearDown(self):
        pass

    def test_pred_label(self):
        odf = domain.pred_shalla_cat(self.domains)
        self.assertIn("pred_label", odf.columns)
        self.assertIn("all_domain_probs", odf.columns)
        self.assertTrue(odf.iloc[0]["pred_label"] == self.true_labels[0])
        self.assertTrue(odf.iloc[1]["pred_label"] == self.true_labels[1])


if __name__ == "__main__":
    unittest.main()