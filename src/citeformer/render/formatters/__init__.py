"""Home-grown CSL formatters.

Each concrete formatter is a ``CitationFormatter`` subclass that takes a
CSL-JSON item (our ``Source.metadata``) and a 1-indexed citation number and
emits the inline marker + bibliography entry for its specific style. No
citeproc-py involvement; quirks-free output owned by us.

Adding a new style is the skill at ``.claude/skills/add-citation-format/``.
The short version:

1. Create ``src/citeformer/render/formatters/<name>.py`` with a subclass.
2. Register it in ``_REGISTRY`` below (with any canonical aliases).
3. Add fixtures to ``tests/unit/test_formatters.py`` covering at least the
   four canonical CSL types (book / article-journal / chapter / thesis).

Public API:

- ``CitationFormatter`` — the base ABC.
- ``Author`` — CSL-author record used across formatters.
- ``get_formatter(name)`` — registry lookup.
- ``available_formatters()`` — list of canonical style names.
"""

from __future__ import annotations

from citeformer.render.formatters._base import Author, CitationFormatter
from citeformer.render.formatters.apa import APAFormatter
from citeformer.render.formatters.chicago import ChicagoAuthorDateFormatter
from citeformer.render.formatters.ieee import IEEEFormatter
from citeformer.render.formatters.mla import MLAFormatter
from citeformer.render.formatters.nature import NatureFormatter
from citeformer.render.formatters.vancouver import VancouverFormatter

_REGISTRY: dict[str, type[CitationFormatter]] = {
    "apa": APAFormatter,
    "apa-7": APAFormatter,
    "mla": MLAFormatter,
    "mla-9": MLAFormatter,
    "chicago": ChicagoAuthorDateFormatter,
    "chicago-author-date": ChicagoAuthorDateFormatter,
    "ieee": IEEEFormatter,
    "nature": NatureFormatter,
    "vancouver": VancouverFormatter,
}

# Canonical / primary names (without aliases). Order matters for docs + tests.
_CANONICAL: tuple[str, ...] = (
    "apa-7",
    "mla-9",
    "chicago-author-date",
    "ieee",
    "nature",
    "vancouver",
)


def get_formatter(name: str) -> CitationFormatter:
    """Look up and instantiate a formatter by style name.

    Args:
        name: Style identifier (canonical or alias). Case-insensitive.

    Returns:
        A fresh `CitationFormatter` instance. Formatters are stateless, so
        this is cheap to call per-generation.

    Raises:
        ValueError: If ``name`` isn't in the registry.
    """
    try:
        formatter_class = _REGISTRY[name.lower()]
    except KeyError as e:
        raise ValueError(
            f"Unknown citation style {name!r}. Available: {sorted(set(_REGISTRY))}"
        ) from e
    return formatter_class()


def available_formatters() -> list[str]:
    """Return the canonical list of built-in style identifiers (no aliases)."""
    return list(_CANONICAL)


__all__ = [
    "APAFormatter",
    "Author",
    "ChicagoAuthorDateFormatter",
    "CitationFormatter",
    "IEEEFormatter",
    "MLAFormatter",
    "NatureFormatter",
    "VancouverFormatter",
    "available_formatters",
    "get_formatter",
]
