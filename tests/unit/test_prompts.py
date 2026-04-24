"""Tests for `citeformer.prompts.build_rag_prompt`.

The helper is string-in / string-out. These tests pin the shape so
downstream users (benchmarks, examples, library consumers) can rely on
the ordering and section markers.
"""

from __future__ import annotations

import pytest

from citeformer import Source, build_rag_prompt


def _src(ident: str, title: str, family: str | None = None) -> Source:
    metadata: dict = {"id": ident, "type": "article-journal", "title": title}
    if family:
        metadata["author"] = [{"family": family}]
    return Source(metadata=metadata, content="")


def test_requires_non_empty_query() -> None:
    with pytest.raises(ValueError, match="non-empty `query`"):
        build_rag_prompt(query="  ", sources=[_src("a", "T")])


def test_requires_non_empty_sources() -> None:
    with pytest.raises(ValueError, match="at least 1 source"):
        build_rag_prompt(query="Anything.", sources=[])


def test_numbers_sources_one_indexed() -> None:
    sources = [
        _src("a", "Title A", family="Alpha"),
        _src("b", "Title B", family="Beta"),
        _src("c", "Title C", family="Gamma"),
    ]
    out = build_rag_prompt(query="Compare.", sources=sources)
    assert "[1] Alpha: Title A" in out
    assert "[2] Beta: Title B" in out
    assert "[3] Gamma: Title C" in out
    # No zero-indexing and no out-of-range numbering.
    assert "[0]" not in out
    assert "[4]" not in out


def test_handles_sources_without_authors() -> None:
    out = build_rag_prompt(query="Summarize.", sources=[_src("x", "Anonymous work")])
    assert "[1] Anonymous work" in out


def test_two_authors_uses_ampersand() -> None:
    sources = [
        Source(
            metadata={
                "id": "x",
                "type": "book",
                "title": "Joint Work",
                "author": [{"family": "Smith"}, {"family": "Jones"}],
            },
            content="",
        )
    ]
    out = build_rag_prompt(query="Test.", sources=sources)
    assert "Smith & Jones: Joint Work" in out


def test_four_plus_authors_uses_et_al() -> None:
    sources = [
        Source(
            metadata={
                "id": "x",
                "type": "book",
                "title": "Big Group Paper",
                "author": [
                    {"family": "A"},
                    {"family": "B"},
                    {"family": "C"},
                    {"family": "D"},
                ],
            },
            content="",
        )
    ]
    out = build_rag_prompt(query="Test.", sources=sources)
    assert "et al." in out


def test_literal_org_name_used_when_family_missing() -> None:
    sources = [
        Source(
            metadata={
                "id": "x",
                "type": "report",
                "title": "Report",
                "author": [{"literal": "OpenAI"}],
            },
            content="",
        )
    ]
    out = build_rag_prompt(query="Test.", sources=sources)
    assert "OpenAI: Report" in out


def test_system_renders_at_top() -> None:
    out = build_rag_prompt(
        query="Q.",
        sources=[_src("a", "T")],
        system="You are a terse cataloger.",
    )
    assert out.startswith("You are a terse cataloger.")


def test_cite_hint_defaults_are_present_and_suppressible() -> None:
    with_hint = build_rag_prompt(query="Q.", sources=[_src("a", "T")])
    assert "[N]" in with_hint
    without = build_rag_prompt(query="Q.", sources=[_src("a", "T")], cite_hint=None)
    assert "[N]" not in without


def test_example_line_renders_when_provided() -> None:
    out = build_rag_prompt(
        query="Q.",
        sources=[_src("a", "T")],
        example="The sky is blue [1].",
    )
    assert "Example: The sky is blue [1]." in out


def test_answer_prefix_configurable_and_suppressible() -> None:
    survey = build_rag_prompt(
        query="Q.",
        sources=[_src("a", "T")],
        answer_prefix="Survey:",
    )
    assert survey.rstrip().endswith("Survey:")
    no_prefix = build_rag_prompt(
        query="Q.",
        sources=[_src("a", "T")],
        answer_prefix=None,
    )
    assert not no_prefix.rstrip().endswith("Answer:")
    assert not no_prefix.rstrip().endswith("Survey:")


def test_task_section_contains_query_verbatim() -> None:
    q = "Explain self-attention in one sentence."
    out = build_rag_prompt(query=q, sources=[_src("a", "T")])
    assert q in out


def test_blank_line_section_separators() -> None:
    """Blank lines between sections keep the prompt humane to debug."""
    out = build_rag_prompt(
        query="Q.",
        sources=[_src("a", "T")],
        system="S.",
        example="E [1].",
    )
    # Each section pair separated by exactly one blank line (two newlines).
    assert "\n\n" in out
    assert "\n\n\n" not in out  # no triple newlines from empty sections
