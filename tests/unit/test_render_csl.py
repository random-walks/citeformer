"""Tests for `citeformer.render` — bibliography rendering per style.

Snapshot-pins the rendered output for a canonical set of citations across all
six home-grown formatters (APA 7, MLA 9, Chicago author-date, IEEE, Nature,
Vancouver — see ADR-004, `docs/decisions/004-citeproc-rewrite.md`). A change
to any formatter that shifts the output flags up as a snapshot diff — forcing
a deliberate §10.2 refresh ceremony (see `docs/reference/contracts.md`).
"""

from __future__ import annotations

from citeformer import Citation, Source
from citeformer.render import render_references


def _canonical_sources() -> list[Source]:
    """Four canonical sources covering book / article / chapter / thesis."""
    return [
        Source(
            metadata={
                "id": "poe-raven",
                "type": "book",
                "author": [{"family": "Poe", "given": "Edgar Allan"}],
                "title": "The Raven and Other Poems",
                "publisher": "Wiley and Putnam",
                "publisher-place": "New York",
                "issued": {"date-parts": [[1845]]},
            },
            content="Once upon a midnight dreary...",
        ),
        Source(
            metadata={
                "id": "smith2023",
                "type": "article-journal",
                "author": [
                    {"family": "Smith", "given": "Alice"},
                    {"family": "Jones", "given": "Bob"},
                ],
                "title": "Constrained Decoding for RAG",
                "container-title": "Journal of Applied AI",
                "volume": "12",
                "issue": "3",
                # Single page — historical: citeproc-py 0.9.2 had a Chicago
                # page-range bug (UnboundLocalError), so this fixture used a
                # single page. The home-grown formatters (ADR-004) don't have
                # the bug; the fixture keeps the single page so the pinned
                # snapshots stay stable. Page *ranges* are covered by
                # test_csl_suite.py.
                "page": "45",
                "issued": {"date-parts": [[2023, 4]]},
                "DOI": "10.1234/jaai.2023.12.3.45",
            },
            content="We show that ...",
        ),
        Source(
            metadata={
                "id": "melville-chapter",
                "type": "chapter",
                "author": [{"family": "Melville", "given": "Herman"}],
                "title": "Loomings",
                "container-title": "Moby-Dick",
                "publisher": "Harper & Brothers",
                "issued": {"date-parts": [[1851]]},
                "page": "1",
            },
            content="Call me Ishmael...",
        ),
        Source(
            metadata={
                "id": "austen-thesis",
                "type": "thesis",
                "author": [{"family": "Austen", "given": "Jane"}],
                "title": "A Critical Edition of Early Novels",
                "publisher": "University of Oxford",
                "genre": "PhD dissertation",
                "issued": {"date-parts": [[1813]]},
            },
            content="It is a truth universally acknowledged...",
        ),
    ]


def _full_citations() -> list[Citation]:
    """One citation per canonical source, in order."""
    return [Citation(span=(i * 10, i * 10 + 3), source_id=i + 1) for i in range(4)]


# --- Behavioral tests ---------------------------------------------------------


def test_render_references_returns_one_per_unique_source_id() -> None:
    sources = _canonical_sources()
    cs = [
        Citation(span=(0, 3), source_id=1),
        Citation(span=(10, 13), source_id=1),  # duplicate — should collapse
        Citation(span=(20, 23), source_id=3),
    ]
    refs = render_references(sources, cs, style_name="apa-7")
    ids = [r.source_id for r in refs]
    assert ids == [1, 3]


def test_render_references_preserves_ascending_source_id_order() -> None:
    sources = _canonical_sources()
    cs = [
        Citation(span=(0, 3), source_id=4),
        Citation(span=(10, 13), source_id=1),
        Citation(span=(20, 23), source_id=2),
    ]
    refs = render_references(sources, cs, style_name="ieee")
    assert [r.source_id for r in refs] == [1, 2, 4]


def test_render_references_skips_out_of_range_ids() -> None:
    """Grammar-level enforcement prevents this in P2+; tests belt-and-suspenders."""
    sources = _canonical_sources()  # 4 sources → ids 1..4
    cs = [
        Citation(span=(0, 3), source_id=1),
        Citation(span=(10, 13), source_id=99),  # out of range
    ]
    refs = render_references(sources, cs, style_name="apa-7")
    assert [r.source_id for r in refs] == [1]


def test_render_references_returns_empty_when_no_citations() -> None:
    assert render_references(_canonical_sources(), [], style_name="apa-7") == []


def test_render_references_inline_marker_reflects_style_format() -> None:
    sources = _canonical_sources()
    cs = _full_citations()

    apa_refs = render_references(sources, cs, style_name="apa-7")
    # APA is author-date; inline should contain an author surname.
    assert "Poe" in apa_refs[0].inline_marker
    assert "1845" in apa_refs[0].inline_marker

    ieee_refs = render_references(sources, cs, style_name="ieee")
    # IEEE is numeric; the home-grown formatter's inline marker is the
    # bracketed number ("[1]"), never an author-year form.
    assert "Poe" not in ieee_refs[0].inline_marker


def test_render_references_full_entry_contains_author_and_title() -> None:
    sources = _canonical_sources()
    cs = _full_citations()
    refs = render_references(sources, cs, style_name="apa-7")
    assert "Poe" in refs[0].rendered
    assert "Raven" in refs[0].rendered
    # Smith+Jones appears on the second source (DOI + journal article).
    assert "Smith" in refs[1].rendered
    assert "Journal of Applied AI" in refs[1].rendered


# --- Snapshots (§10.2 / bundled-style contract) -------------------------------


def test_render_snapshot_apa_7(data_regression) -> None:  # type: ignore[no-untyped-def]
    refs = render_references(_canonical_sources(), _full_citations(), "apa-7")
    data_regression.check(_refs_dict(refs))


def test_render_snapshot_mla_9(data_regression) -> None:  # type: ignore[no-untyped-def]
    refs = render_references(_canonical_sources(), _full_citations(), "mla-9")
    data_regression.check(_refs_dict(refs))


def test_render_snapshot_chicago_author_date(data_regression) -> None:  # type: ignore[no-untyped-def]
    refs = render_references(_canonical_sources(), _full_citations(), "chicago-author-date")
    data_regression.check(_refs_dict(refs))


def test_render_snapshot_ieee(data_regression) -> None:  # type: ignore[no-untyped-def]
    refs = render_references(_canonical_sources(), _full_citations(), "ieee")
    data_regression.check(_refs_dict(refs))


def test_render_snapshot_nature(data_regression) -> None:  # type: ignore[no-untyped-def]
    refs = render_references(_canonical_sources(), _full_citations(), "nature")
    data_regression.check(_refs_dict(refs))


def test_render_snapshot_vancouver(data_regression) -> None:  # type: ignore[no-untyped-def]
    refs = render_references(_canonical_sources(), _full_citations(), "vancouver")
    data_regression.check(_refs_dict(refs))


# --- Helpers ------------------------------------------------------------------


def _refs_dict(refs: list) -> list[dict]:
    return [
        {
            "source_id": r.source_id,
            "inline_marker": r.inline_marker,
            "rendered": r.rendered,
        }
        for r in refs
    ]
