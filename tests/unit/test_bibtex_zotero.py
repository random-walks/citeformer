"""Tests for BibTeX + Zotero CSL-JSON ingest adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from citeformer import Source, load_bibtex, load_zotero_csl
from citeformer.metadata.bibtex import (
    BIBTEX_TYPE_MAP,
    bibtex_to_csl_json,
    parse_bibtex,
)

# --- BibTeX ------------------------------------------------------------------


_SIMPLE_BIB = """
@article{vaswani2017,
    author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki},
    title = {Attention Is All You Need},
    journal = {NeurIPS},
    year = 2017,
    volume = 30,
    pages = {5998-6008},
    doi = {10.48550/arXiv.1706.03762}
}

@book{strunk2000,
    author = {Strunk, William and White, E. B.},
    title = {The Elements of Style},
    publisher = {Macmillan},
    year = 2000,
    edition = {4th},
    address = {New York}
}

@inproceedings{devlin2019,
    author = "Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina",
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    booktitle = "NAACL-HLT",
    year = "2019",
    month = jun,
    pages = "4171--4186"
}
"""


def test_parse_bibtex_extracts_all_entries() -> None:
    entries = parse_bibtex(_SIMPLE_BIB)
    assert len(entries) == 3
    assert entries[0]["__key"] == "vaswani2017"
    assert entries[0]["__type"] == "article"
    assert entries[1]["__key"] == "strunk2000"
    assert entries[2]["__key"] == "devlin2019"


def test_bibtex_to_csl_json_maps_types() -> None:
    entries = parse_bibtex(_SIMPLE_BIB)
    csl = [bibtex_to_csl_json(e) for e in entries]
    assert csl[0]["type"] == "article-journal"  # @article
    assert csl[1]["type"] == "book"
    assert csl[2]["type"] == "paper-conference"  # @inproceedings


def test_bibtex_to_csl_json_preserves_id() -> None:
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(_SIMPLE_BIB)]
    assert [it["id"] for it in items] == ["vaswani2017", "strunk2000", "devlin2019"]


def test_bibtex_author_list_splits_on_and() -> None:
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(_SIMPLE_BIB)]
    authors = items[0]["author"]
    assert len(authors) == 3
    assert authors[0] == {"family": "Vaswani", "given": "Ashish"}
    assert authors[1] == {"family": "Shazeer", "given": "Noam"}
    assert authors[2] == {"family": "Parmar", "given": "Niki"}


def test_bibtex_author_given_family_convention() -> None:
    """BibTeX with ``Given Family`` (no comma) flips to CSL ``family/given``."""
    raw = "@misc{x, author = {Alice Smith and Bob Jones}, year = 2020}"
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(raw)]
    authors = items[0]["author"]
    assert authors[0] == {"family": "Smith", "given": "Alice"}
    assert authors[1] == {"family": "Jones", "given": "Bob"}


def test_bibtex_year_and_month_compose_issued() -> None:
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(_SIMPLE_BIB)]
    assert items[0]["issued"] == {"date-parts": [[2017]]}
    assert items[1]["issued"] == {"date-parts": [[2000]]}
    # Devlin entry had `month = jun` → should be month=6.
    assert items[2]["issued"] == {"date-parts": [[2019, 6]]}


def test_bibtex_maps_common_fields() -> None:
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(_SIMPLE_BIB)]
    article = items[0]
    assert article["title"] == "Attention Is All You Need"
    assert article["container-title"] == "NeurIPS"
    assert article["volume"] == "30"
    assert article["page"] == "5998-6008"
    assert article["DOI"] == "10.48550/arXiv.1706.03762"

    book = items[1]
    assert book["publisher"] == "Macmillan"
    assert book["edition"] == "4th"
    assert book["publisher-place"] == "New York"


def test_bibtex_nested_braces_preserved() -> None:
    """``{BERT}`` inside a title keeps its brace-protected casing."""
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(_SIMPLE_BIB)]
    # The home-grown parser strips one outer pair of braces but keeps inner ones.
    assert "BERT" in items[2]["title"]


def test_bibtex_unknown_fields_roundtrip_through_custom() -> None:
    raw = """
@article{x,
    author = {Doe, Jane},
    title = {Some Title},
    year = 2020,
    eprint = {2020.12345},
    eprinttype = {arxiv},
    shortjournal = {J. Thingy}
}
"""
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(raw)]
    assert "custom" in items[0]
    assert items[0]["custom"]["eprint"] == "2020.12345"
    assert items[0]["custom"]["eprinttype"] == "arxiv"


def test_bibtex_skips_macro_and_preamble_blocks() -> None:
    """``@string``, ``@preamble``, ``@comment`` are skipped without error."""
    raw = """
@string{jcss = "Journal of Computer and System Sciences"}
@preamble{"\\newcommand{\\foo}{bar}"}
@article{legit,
    author = {Real, Author},
    title = {A Real Paper},
    year = 2020
}
@comment{this is a note}
"""
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(raw)]
    assert len(items) == 1
    assert items[0]["id"] == "legit"


def test_bibtex_entry_types_map_to_csl() -> None:
    """Every mapped type produces the expected CSL type."""
    for btype, csl_type in BIBTEX_TYPE_MAP.items():
        raw = f"@{btype}{{k, title = {{T}}, year = 2020}}"
        items = [bibtex_to_csl_json(e) for e in parse_bibtex(raw)]
        assert items[0]["type"] == csl_type, f"@{btype} → expected {csl_type}"


def test_bibtex_unknown_type_falls_back_to_document() -> None:
    raw = "@madeuptype{k, title = {T}, year = 2020}"
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(raw)]
    assert items[0]["type"] == "document"


def test_load_bibtex_accepts_path_or_string(tmp_path: Path) -> None:
    # Path
    f = tmp_path / "test.bib"
    f.write_text(_SIMPLE_BIB, encoding="utf-8")
    from_path = load_bibtex(f)
    assert len(from_path) == 3

    # String
    from_string = load_bibtex(_SIMPLE_BIB)
    assert len(from_string) == 3

    # Same content regardless of source.
    assert from_path == from_string


def test_source_from_bibtex_builds_one_source_per_entry(tmp_path: Path) -> None:
    f = tmp_path / "lib.bib"
    f.write_text(_SIMPLE_BIB, encoding="utf-8")
    sources = Source.from_bibtex(f)
    assert len(sources) == 3
    assert all(isinstance(s, Source) for s in sources)
    assert sources[0].metadata["id"] == "vaswani2017"
    # content is empty on BibTeX sources — metadata only.
    assert sources[0].content == ""


def test_bibtex_in_press_year_is_dropped() -> None:
    """A non-numeric year becomes no ``issued`` rather than a crash."""
    raw = "@article{x, author = {Doe, J.}, title = {Preprint}, year = {in press}}"
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(raw)]
    assert "issued" not in items[0]


def test_bibtex_handles_empty_input() -> None:
    assert parse_bibtex("") == []
    assert load_bibtex("") == []


def test_bibtex_author_single_token_emits_literal() -> None:
    """``{Plato}`` with no comma / space is a literal (probably an org or mono-name)."""
    raw = "@book{plato, author = {Plato}, title = {The Republic}, year = -380}"
    items = [bibtex_to_csl_json(e) for e in parse_bibtex(raw)]
    assert items[0]["author"] == [{"literal": "Plato"}]


# --- Zotero CSL-JSON ----------------------------------------------------------


_ZOTERO_EXPORT = [
    {
        "id": "item-1",
        "type": "article-journal",
        "author": [{"family": "Smith", "given": "Alice"}],
        "title": "A Paper",
        "container-title": "Journal of X",
        "issued": {"date-parts": [["2021", "5"]]},  # Zotero stringifies ints
        "DOI": "10.1234/x.2021",
    },
    {
        "id": "item-2",
        "type": "book",
        "author": [{"family": "Jones", "given": "Bob"}],
        "title": "A Book",
        "publisher": "Penguin",
        "issued": {"date-parts": [[1999]]},
    },
    {
        # Duplicate id — should be dedupeable
        "id": "item-1",
        "type": "article-journal",
        "title": "Duplicate",
    },
    {
        # Null field — should be dropped
        "id": "item-3",
        "type": "webpage",
        "title": "Site",
        "URL": "https://example.com",
        "accessed": None,
    },
]


def test_load_zotero_csl_accepts_iterable() -> None:
    items = load_zotero_csl(_ZOTERO_EXPORT)
    # Duplicate "item-1" should be dropped.
    ids = [it["id"] for it in items]
    assert ids == ["item-1", "item-2", "item-3"]


def test_load_zotero_csl_dedupe_can_be_disabled() -> None:
    items = load_zotero_csl(_ZOTERO_EXPORT, dedupe=False)
    ids = [it["id"] for it in items]
    assert ids.count("item-1") == 2


def test_load_zotero_csl_normalises_stringified_date_parts() -> None:
    items = load_zotero_csl(_ZOTERO_EXPORT)
    assert items[0]["issued"] == {"date-parts": [[2021, 5]]}


def test_load_zotero_csl_drops_null_fields() -> None:
    items = load_zotero_csl(_ZOTERO_EXPORT)
    # item-3 has accessed=None which should be dropped.
    assert "accessed" not in items[2]


def test_load_zotero_csl_filter_fn() -> None:
    items = load_zotero_csl(
        _ZOTERO_EXPORT,
        filter_fn=lambda it: it.get("type") == "book",
    )
    assert len(items) == 1
    assert items[0]["id"] == "item-2"


def test_load_zotero_csl_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps(_ZOTERO_EXPORT), encoding="utf-8")
    items = load_zotero_csl(path)
    assert [it["id"] for it in items] == ["item-1", "item-2", "item-3"]


def test_load_zotero_csl_rejects_top_level_non_list() -> None:
    with pytest.raises(ValueError, match="top-level list"):
        load_zotero_csl('{"not": "a list"}')


def test_source_from_zotero_yields_sources() -> None:
    sources = Source.from_zotero(_ZOTERO_EXPORT)
    assert len(sources) == 3
    assert all(isinstance(s, Source) for s in sources)
    assert sources[0].metadata["DOI"] == "10.1234/x.2021"
    assert sources[0].content == ""


def test_source_from_zotero_forwards_filter_fn() -> None:
    sources = Source.from_zotero(
        _ZOTERO_EXPORT,
        filter_fn=lambda it: it.get("type") == "webpage",
    )
    assert len(sources) == 1
    assert sources[0].metadata["id"] == "item-3"
