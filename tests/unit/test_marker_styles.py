"""End-to-end tests for the ``marker_style`` plumbing.

Checks the parse-regex / mock-backend / orchestrator wiring across all four
:class:`~citeformer.core.MarkerStyle` variants — the grammar layer is
exercised by ``tests/unit/test_grammar_builder.py``; this file makes sure
``Citeformer.generate()`` end-to-end produces and parses markers correctly
regardless of which style is chosen.
"""

from __future__ import annotations

import pytest

from citeformer import Citeformer, MarkerStyle, MockBackend, Policy, Source


@pytest.fixture
def sources() -> list[Source]:
    return [
        Source(
            metadata={
                "id": f"s{i}",
                "type": "book",
                "title": f"Book {i}",
                "author": [{"family": f"Author{i}"}],
            },
            content=f"Content {i}",
        )
        for i in range(1, 4)
    ]


@pytest.mark.parametrize(
    "marker_style,expected_marker",
    [
        (MarkerStyle.BRACKET, "[1]"),
        (MarkerStyle.PAREN, "(1)"),
        (MarkerStyle.CURLY, "{1}"),
        (MarkerStyle.CARET, "^1"),
    ],
)
def test_mock_backend_echoes_chosen_marker_style(
    sources: list[Source], marker_style: MarkerStyle, expected_marker: str
) -> None:
    """The MockBackend fallback uses the configured marker shape."""
    cf = Citeformer(
        backend=MockBackend(),
        citation_policy=Policy.AUTO,
        marker_style=marker_style,
    )
    result = cf.generate(prompt="hi", sources=sources)
    assert expected_marker in result.text


@pytest.mark.parametrize("marker_style", list(MarkerStyle))
def test_parse_recognises_chosen_marker(sources: list[Source], marker_style: MarkerStyle) -> None:
    """A generation end-to-end produces ≥1 Citation regardless of marker shape."""
    cf = Citeformer(
        backend=MockBackend(),
        citation_policy=Policy.AUTO,
        marker_style=marker_style,
    )
    result = cf.generate(prompt="hi", sources=sources)
    assert len(result.citations) == 1
    assert result.citations[0].source_id == 1


def test_parse_does_not_match_wrong_marker(sources: list[Source]) -> None:
    """Parsing with BRACKET style must NOT pick up a ``(1)`` marker."""
    cf_paren = Citeformer(
        backend=MockBackend(responses={"hi": "Paren mock (1)."}),
        citation_policy=Policy.AUTO,
        marker_style=MarkerStyle.BRACKET,  # wrong style for this output
    )
    result = cf_paren.generate(prompt="hi", sources=sources)
    assert result.citations == []


def test_per_call_override_wins_over_default(sources: list[Source]) -> None:
    """Passing ``marker_style=`` on ``generate()`` overrides the orchestrator's default."""
    cf = Citeformer(
        backend=MockBackend(),
        citation_policy=Policy.AUTO,
        marker_style=MarkerStyle.BRACKET,
    )
    result = cf.generate(
        prompt="hi",
        sources=sources,
        marker_style=MarkerStyle.CURLY,
    )
    assert "{1}" in result.text
    assert "[1]" not in result.text


def test_streaming_honours_marker_style(sources: list[Source]) -> None:
    """The streaming finalize path uses the stream's chosen marker style."""
    cf = Citeformer(
        backend=MockBackend(),
        citation_policy=Policy.AUTO,
        marker_style=MarkerStyle.PAREN,
    )
    stream = cf.stream(prompt="hi", sources=sources)
    chunks = list(stream)
    assert "".join(chunks)  # non-empty
    result = stream.finalize()
    assert "(1)" in result.text
    assert len(result.citations) == 1
    assert result.citations[0].source_id == 1


@pytest.mark.parametrize("marker_style", list(MarkerStyle))
def test_references_rendered_same_regardless_of_marker(
    sources: list[Source], marker_style: MarkerStyle
) -> None:
    """Marker style is orthogonal to CSL-rendered bibliography output."""
    cf = Citeformer(
        backend=MockBackend(),
        citation_policy=Policy.AUTO,
        marker_style=marker_style,
    )
    result = cf.generate(prompt="hi", sources=sources)
    assert len(result.references) == 1
    # Bibliography output is driven by the CSL style, not the marker style.
    assert "Author1" in result.references[0].rendered
