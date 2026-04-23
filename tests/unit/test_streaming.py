"""Tests for `Citeformer.stream()` + `StreamingResult`.

Uses `MockBackend.stream()` (real ML backends are covered in integration
tests). Pins the contract: chunk iteration yields text, `.text` reflects
consumed chunks, `.finalize()` produces a complete `GenerationResult` that
matches what `generate()` would have returned.
"""

from __future__ import annotations

import pytest

from citeformer import Citeformer, Policy, Source
from citeformer.backends import MockBackend
from citeformer.citeformer import StreamingResult


def _sources() -> list[Source]:
    return [
        Source(
            metadata={
                "id": f"src-{i}",
                "type": "book",
                "title": f"Book {i}",
                "author": [{"family": f"Author{i}"}],
                "issued": {"date-parts": [[2000 + i]]},
            },
            content=f"Content {i}",
        )
        for i in range(1, 4)
    ]


def test_stream_returns_streaming_result() -> None:
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.stream(prompt="Hello", sources=_sources())
    assert isinstance(stream, StreamingResult)


def test_stream_yields_multiple_chunks_for_nontrivial_output() -> None:
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.stream(prompt="Write something longer than a few chars", sources=_sources())
    chunks = list(stream)
    # MockBackend splits into 10-char chunks; the prompt above produces >1.
    assert len(chunks) > 1
    # Every chunk is a non-empty string.
    assert all(isinstance(c, str) and c for c in chunks)


def test_stream_accumulated_text_matches_joined_chunks() -> None:
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.stream(prompt="Hi there", sources=_sources())
    collected: list[str] = []
    for chunk in stream:
        collected.append(chunk)
        # `.text` should always equal the running join.
        assert stream.text == "".join(collected)
    assert stream.text == "".join(collected)


def test_finalize_returns_generation_result_with_citations_and_references() -> None:
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.stream(prompt="Write a mock", sources=_sources())
    # Consume half the stream manually.
    next(stream)
    # `.finalize()` should exhaust the remaining chunks and build the result.
    result = stream.finalize()
    assert result.text == stream.text
    assert result.text  # non-empty
    # MockBackend's fallback emits `[1]`, so we expect at least one citation.
    assert len(result.citations) >= 1
    assert all(1 <= c.source_id <= 3 for c in result.citations)
    # Every cited source should have a rendered reference.
    cited_ids = {c.source_id for c in result.citations}
    rendered_ids = {r.source_id for r in result.references}
    assert cited_ids == rendered_ids


def test_finalize_is_idempotent() -> None:
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.stream(prompt="Test", sources=_sources())
    first = stream.finalize()
    second = stream.finalize()
    assert first is second  # cached, same instance


def test_finalize_without_iterating_exhausts_stream() -> None:
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.stream(prompt="Test", sources=_sources())
    # Call finalize directly, skipping iteration.
    result = stream.finalize()
    assert result.text  # non-empty — iterator was exhausted on our behalf


def test_stream_matches_generate_on_same_inputs() -> None:
    """Streamed output joined should equal what generate() produces."""
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    sources = _sources()
    prompt = "Compare streamed vs non-streamed"

    direct = cf.generate(prompt=prompt, sources=sources)
    streamed = cf.stream(prompt=prompt, sources=sources).finalize()

    assert direct.text == streamed.text
    assert [(c.source_id, c.span) for c in direct.citations] == [
        (c.source_id, c.span) for c in streamed.citations
    ]
    assert [r.rendered for r in direct.references] == [r.rendered for r in streamed.references]


def test_stream_with_policy_override() -> None:
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.stream(prompt="X", sources=_sources(), policy=Policy.REQUIRED)
    result = stream.finalize()
    assert result.text  # still works; MockBackend ignores policy


def test_default_backend_stream_falls_back_to_generate() -> None:
    """A backend that doesn't override `stream()` still works through the ABC default."""

    class EchoOnly(MockBackend):
        # Deliberately don't override stream().
        def stream(self, *args, **kwargs):  # type: ignore[no-untyped-def, override]
            # Call the ABC default via super of the *base* class.
            from citeformer.backends.base import Backend

            return Backend.stream(self, *args, **kwargs)

    cf = Citeformer(backend=EchoOnly(), citation_policy=Policy.AUTO)
    stream = cf.stream(prompt="X", sources=_sources())
    chunks = list(stream)
    # Default stream() yields the full text as one chunk.
    assert len(chunks) == 1
    assert chunks[0] == stream.text


@pytest.mark.parametrize("chunk_count", [1, 3, 10])
def test_stream_preserves_text_across_chunk_boundaries(chunk_count: int) -> None:
    """Splitting at arbitrary offsets should still round-trip."""

    class FixedChunks(MockBackend):
        def __init__(self, n_chunks: int) -> None:
            super().__init__()
            self.n_chunks = n_chunks

        def stream(self, *args, **kwargs):  # type: ignore[no-untyped-def, override]
            text = self.generate(*args, **kwargs)
            step = max(1, len(text) // self.n_chunks)
            for i in range(0, len(text), step):
                yield text[i : i + step]

    cf = Citeformer(backend=FixedChunks(chunk_count), citation_policy=Policy.AUTO)
    stream = cf.stream(prompt="Hello sources", sources=_sources())
    result = stream.finalize()
    direct = cf.generate(prompt="Hello sources", sources=_sources())
    assert result.text == direct.text
