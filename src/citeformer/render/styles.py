"""Style-name resolution.

Since the home-grown-formatter rewrite (see
``docs/decisions/004-citeproc-rewrite.md``), "style loading" is just a
formatter lookup — no more CSL XML parsing. This module stays for API
compatibility with P3 callers and for future use by the optional
``citeproc-compat`` extra.
"""

from __future__ import annotations

from citeformer.render.formatters import available_formatters, get_formatter

__all__ = [
    "bundled_style_names",
    "get_formatter",
    "style_citation_format",
]


def bundled_style_names() -> list[str]:
    """Return the canonical list of bundled style identifiers.

    Aliases (``"apa"`` → ``"apa-7"``, etc.) are omitted. Any name in this
    list — or an alias — works with `render_references` and `get_formatter`.
    """
    return available_formatters()


def style_citation_format(name: str) -> str:
    """Classify a bundled style as ``"author-date"`` / ``"author"`` / ``"numeric"``.

    Useful for downstream tooling that wants to know what kind of inline
    marker a style produces without actually rendering.

    Args:
        name: Bundled style identifier (canonical or alias).

    Returns:
        The `citation_format` class attribute of the resolved formatter.
    """
    return get_formatter(name).citation_format
