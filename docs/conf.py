"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

configure(globals())

# Render Google-style "Attributes:" sections as :ivar: fields. Without this,
# napoleon emits a separate `.. attribute::` for each entry, which collides with
# the same dataclass fields picked up by :undoc-members: — nine duplicate-object
# warnings on LLMConfig alone, and CI builds docs with -W.
napoleon_use_ivar = True
