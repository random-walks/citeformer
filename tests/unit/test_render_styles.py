"""Tests for `citeformer.render.styles` — name registry + classification.

Style loading used to mean "parse a CSL XML file via citeproc-py"; since the
home-grown rewrite, it means "look up a CitationFormatter class". These
tests pin the public name list + the `citation_format` classification.
"""

from __future__ import annotations

import pytest

from citeformer.render import (
    bundled_style_names,
    get_formatter,
    style_citation_format,
)

_EXPECTED_BUNDLED = [
    "apa-7",
    "mla-9",
    "chicago-author-date",
    "ieee",
    "nature",
    "vancouver",
]


def test_bundled_style_names_are_stable_list() -> None:
    assert bundled_style_names() == _EXPECTED_BUNDLED


def test_get_formatter_returns_same_kind_for_aliases() -> None:
    assert type(get_formatter("apa")) is type(get_formatter("apa-7"))
    assert type(get_formatter("mla")) is type(get_formatter("mla-9"))
    assert type(get_formatter("chicago")) is type(get_formatter("chicago-author-date"))


def test_get_formatter_is_case_insensitive() -> None:
    assert type(get_formatter("APA-7")) is type(get_formatter("apa-7"))


def test_get_formatter_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown citation style"):
        get_formatter("does-not-exist")


def test_style_citation_format_classifies_bundled_styles() -> None:
    # APA and Chicago are author-date; MLA is author; IEEE / Nature / Vancouver are numeric.
    assert style_citation_format("apa-7") == "author-date"
    assert style_citation_format("mla-9") == "author"
    assert style_citation_format("chicago-author-date") == "author-date"
    assert style_citation_format("ieee") == "numeric"
    assert style_citation_format("nature") == "numeric"
    assert style_citation_format("vancouver") == "numeric"


def test_formatter_instances_are_stateless() -> None:
    """Formatters are created fresh per call; no instance state persists."""
    f1 = get_formatter("ieee")
    f2 = get_formatter("ieee")
    assert f1 is not f2  # fresh instances
    assert f1.name == f2.name == "ieee"
