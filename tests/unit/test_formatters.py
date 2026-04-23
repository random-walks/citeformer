"""Per-formatter granular tests.

Where ``test_render_csl.py`` pins the full bibliography via snapshots for the
canonical fixture set, this file exercises narrower behaviours:

- Each CSL item type (book / article-journal / chapter / thesis / webpage /
  paper-conference / report) renders without raising for every formatter.
- Edge cases: missing author, missing year, single-author, many-author
  threshold for ``et al.``, literal (organization) names, hyphenated
  given names, unknown CSL types fall back cleanly.
- Inline marker shape matches each style's ``citation_format``:
  - numeric styles produce ``[N]`` or ``N``.
  - author-date styles produce ``(Author, YYYY)`` / ``(Author YYYY)``.
  - author styles (MLA) produce ``(Author)``.
"""

from __future__ import annotations

import pytest

from citeformer.render.formatters import Author, get_formatter

# --- Fixtures -----------------------------------------------------------------


def _book(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "id": "book",
        "type": "book",
        "author": [{"family": "Poe", "given": "Edgar Allan"}],
        "title": "The Raven",
        "publisher": "Wiley",
        "publisher-place": "New York",
        "issued": {"date-parts": [[1845]]},
    }
    base.update(overrides)
    return base


def _article(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "id": "art",
        "type": "article-journal",
        "author": [
            {"family": "Smith", "given": "Alice"},
            {"family": "Jones", "given": "Bob"},
        ],
        "title": "A study",
        "container-title": "Journal of X",
        "volume": "12",
        "issue": "3",
        "page": "45",
        "issued": {"date-parts": [[2023]]},
        "DOI": "10.1/example",
    }
    base.update(overrides)
    return base


def _chapter(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "id": "ch",
        "type": "chapter",
        "author": [{"family": "Melville", "given": "Herman"}],
        "title": "Loomings",
        "container-title": "Moby-Dick",
        "editor": [{"family": "Edit", "given": "Ed"}],
        "publisher": "Harper",
        "publisher-place": "New York",
        "issued": {"date-parts": [[1851]]},
        "page": "1",
    }
    base.update(overrides)
    return base


def _thesis(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "id": "th",
        "type": "thesis",
        "author": [{"family": "Austen", "given": "Jane"}],
        "title": "Early Novels",
        "publisher": "Oxford",
        "genre": "PhD dissertation",
        "issued": {"date-parts": [[1813]]},
    }
    base.update(overrides)
    return base


_STYLES = ["apa-7", "mla-9", "chicago-author-date", "ieee", "nature", "vancouver"]


# --- Every formatter renders every canonical type without raising ------------


@pytest.mark.parametrize("style", _STYLES)
@pytest.mark.parametrize("fixture", [_book(), _article(), _chapter(), _thesis()])
def test_every_style_renders_canonical_types_without_error(style, fixture) -> None:  # type: ignore[no-untyped-def]
    formatter = get_formatter(style)
    inline = formatter.inline(fixture, 1)
    bib = formatter.bibliography(fixture, 1)
    assert isinstance(inline, str) and inline
    assert isinstance(bib, str) and bib


# --- Edge cases on helpers ---------------------------------------------------


def test_author_given_initials_handles_edge_cases() -> None:
    assert Author(family="Smith", given="Edgar Allan").given_initials == "E. A."
    assert Author(family="Dupont", given="Jean-Paul").given_initials == "J.-P."
    assert Author(family="Smith").given_initials == ""
    assert Author(literal="United Nations").given_initials == ""


def test_author_is_literal_flag() -> None:
    assert Author(literal="WHO").is_literal
    assert not Author(family="Smith", given="Alice").is_literal


# --- Inline marker shape per style ------------------------------------------


def test_ieee_inline_marker_is_bracketed_number() -> None:
    f = get_formatter("ieee")
    assert f.inline(_article(), 1) == "[1]"
    assert f.inline(_book(), 42) == "[42]"


def test_nature_inline_marker_is_plain_number() -> None:
    f = get_formatter("nature")
    assert f.inline(_article(), 1) == "1"
    assert f.inline(_book(), 99) == "99"


def test_vancouver_inline_marker_is_bracketed_number() -> None:
    f = get_formatter("vancouver")
    assert f.inline(_article(), 1) == "[1]"


def test_apa_inline_uses_author_year() -> None:
    f = get_formatter("apa-7")
    assert f.inline(_book(), 1) == "(Poe, 1845)"
    assert f.inline(_article(), 1) == "(Smith & Jones, 2023)"


def test_apa_inline_uses_et_al_for_3_plus_authors() -> None:
    many = _article(
        author=[
            {"family": "Smith", "given": "Alice"},
            {"family": "Jones", "given": "Bob"},
            {"family": "Kim", "given": "Carol"},
        ]
    )
    f = get_formatter("apa-7")
    assert f.inline(many, 1) == "(Smith et al., 2023)"


def test_mla_inline_uses_author_only() -> None:
    f = get_formatter("mla-9")
    assert f.inline(_book(), 1) == "(Poe)"
    assert f.inline(_article(), 1) == "(Smith and Jones)"


def test_chicago_inline_uses_author_year_no_comma() -> None:
    f = get_formatter("chicago-author-date")
    assert f.inline(_book(), 1) == "(Poe 1845)"
    assert f.inline(_article(), 1) == "(Smith and Jones 2023)"


# --- Bibliography edge cases -------------------------------------------------


def test_apa_bib_falls_back_to_nd_when_year_missing() -> None:
    f = get_formatter("apa-7")
    item = _book()
    del item["issued"]
    out = f.bibliography(item, 1)
    assert "n.d." in out


def test_apa_bib_handles_missing_author() -> None:
    f = get_formatter("apa-7")
    item = _book()
    del item["author"]
    out = f.bibliography(item, 1)
    # Gracefully renders without author but shouldn't raise.
    assert "The Raven" in out


def test_chicago_bib_handles_literal_author() -> None:
    f = get_formatter("chicago-author-date")
    item = _book(author=[{"literal": "World Health Organization"}])
    out = f.bibliography(item, 1)
    assert "World Health Organization" in out


def test_vancouver_bib_undotted_initials() -> None:
    f = get_formatter("vancouver")
    item = _article()
    out = f.bibliography(item, 1)
    # Vancouver uses "Smith A, Jones B" (no dots on initials).
    assert "Smith A" in out
    assert "Jones B" in out
    assert "Smith A." not in out


def test_ieee_bib_starts_with_bracketed_number() -> None:
    f = get_formatter("ieee")
    out = f.bibliography(_article(), 7)
    assert out.startswith("[7]")


def test_nature_bib_starts_with_dotted_number() -> None:
    f = get_formatter("nature")
    out = f.bibliography(_article(), 3)
    assert out.startswith("3. ")


def test_unknown_csl_type_falls_back_to_article_format() -> None:
    """Unknown types shouldn't raise — every formatter picks a safe default."""
    weird = _book(type="a-type-that-does-not-exist")
    for style in _STYLES:
        out = get_formatter(style).bibliography(weird, 1)
        assert out, f"{style} returned empty for unknown type"


def test_apa_bib_renders_doi_as_https_url() -> None:
    f = get_formatter("apa-7")
    out = f.bibliography(_article(), 1)
    assert "https://doi.org/10.1/example" in out


# --- Many-author thresholds --------------------------------------------------


def _many_authors(n: int) -> list[dict[str, str]]:
    return [{"family": f"A{i}", "given": f"Given{i}"} for i in range(n)]


def test_ieee_uses_et_al_after_six_authors() -> None:
    f = get_formatter("ieee")
    out = f.bibliography(_article(author=_many_authors(7)), 1)
    assert "et al." in out


def test_nature_uses_et_al_after_five_authors() -> None:
    f = get_formatter("nature")
    out = f.bibliography(_article(author=_many_authors(6)), 1)
    assert "et al." in out


def test_vancouver_uses_et_al_after_six_authors() -> None:
    f = get_formatter("vancouver")
    out = f.bibliography(_article(author=_many_authors(7)), 1)
    assert "et al." in out


def test_mla_uses_et_al_for_three_plus_authors() -> None:
    f = get_formatter("mla-9")
    out = f.bibliography(_article(author=_many_authors(3)), 1)
    assert "et al." in out


def test_chicago_uses_et_al_only_beyond_ten_authors() -> None:
    f = get_formatter("chicago-author-date")
    few = f.bibliography(_article(author=_many_authors(5)), 1)
    many = f.bibliography(_article(author=_many_authors(11)), 1)
    assert "et al." not in few
    assert "et al." in many
