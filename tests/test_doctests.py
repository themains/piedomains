#!/usr/bin/env python

"""Run the doctests that are genuinely executable.

The docs build deliberately does not execute bare ``>>>`` blocks (see
docs/conf.py): most API docstrings illustrate usage that needs a live network,
a downloaded model, or an LLM key. The modules listed here are pure, so their
examples are real and worth keeping honest — if someone edits the category list
without updating the count in its docstring, this fails.

Note: the unittest ``load_tests`` protocol is deliberately not used, because
pytest ignores it — the doctests would be collected by ``python -m unittest``
and silently skipped under pytest, which is how this suite runs.
"""

import doctest
import unittest

from piedomains import constants, outcomes

PURE_MODULES = [constants, outcomes]


class TestPureModuleDoctests(unittest.TestCase):
    """Execute the doctests of modules that import cleanly and hit no I/O."""

    def test_doctests_pass(self):
        """Every example in a pure module must produce its documented output."""
        for module in PURE_MODULES:
            with self.subTest(module=module.__name__):
                results = doctest.testmod(module, verbose=False, report=False)
                self.assertEqual(
                    results.failed,
                    0,
                    f"{results.failed} doctest failure(s) in {module.__name__}",
                )
                self.assertGreater(
                    results.attempted,
                    0,
                    f"{module.__name__} is listed as doctested but ran no examples",
                )


if __name__ == "__main__":
    unittest.main()
