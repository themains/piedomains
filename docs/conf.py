"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

configure(globals())

# Render Google-style "Attributes:" sections as :ivar: fields. Without this,
# napoleon emits a separate `.. attribute::` for each entry, which collides with
# the same dataclass fields picked up by :undoc-members: — nine duplicate-object
# warnings on LLMConfig alone, and CI builds docs with -W.
napoleon_use_ivar = True

# Do not auto-execute bare ">>>" blocks. The API docstrings use them to
# illustrate usage — classifying a live domain, downloading a remote checkpoint,
# calling an LLM — none of which can or should run in a docs build; all 103 of
# them failed with NameError because each example is a fragment, not a script.
#
# Genuinely executable examples are still verified, by tests/test_doctests.py,
# and anything wrapped in an explicit `.. doctest::` directive still runs here.
doctest_test_doctest_blocks = ""
