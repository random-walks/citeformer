"""Integration tests for `HFBackend` — loads a real tiny model.

Marked ``integration`` so the default ``pytest`` run skips them. Run with:

    pytest -m integration
    # or
    make test-integration

These need the `hf` extra installed: ``uv sync --extra hf`` or
``pip install 'citeformer[hf]'``.

The point of these tests is to verify the §10.1 contract *structurally* on a
real model, not to measure generation quality. We use the smallest HF model
that sensibly works (``gpt2``, ~500 MB) so the test runs on a laptop CPU in
under 30 seconds.
"""

from __future__ import annotations

import re

import pytest

from citeformer import Citeformer, Policy, Source

# Pattern to extract [N] markers from generated text.
_CITE = re.compile(r"\[(\d+)\]")


def _sources(n: int) -> list[Source]:
    return [
        Source(
            metadata={"id": f"src-{i}", "type": "book", "title": f"Book {i}"},
            content=f"Content chunk {i}",
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture(scope="module")
def hf_backend():  # type: ignore[no-untyped-def]
    """Load a single HFBackend for all integration tests in this module."""
    from citeformer.backends.hf import HFBackend

    return HFBackend(model="gpt2")


@pytest.mark.integration
def test_hf_backend_grammar_compiles(hf_backend) -> None:  # type: ignore[no-untyped-def]
    """Smoke test: the §10.1 grammar compiles against a real tokenizer."""
    from citeformer.grammar import build_grammar

    grammar = build_grammar(n_sources=3, policy=Policy.REQUIRED)
    compiled = hf_backend._compiler.compile_grammar(grammar.gbnf, root_rule_name=grammar.root_rule)
    assert compiled is not None


@pytest.mark.integration
def test_hf_backend_cannot_fabricate_citations_required_policy(hf_backend) -> None:  # type: ignore[no-untyped-def]
    """The core P2 guarantee: with N=3 sources, no `[4]`, `[5]`, ... can appear.

    Structurally impossible at the logit level. If this ever fails, either
    XGrammar is broken, our grammar is wrong, or §10.1 has drifted.
    """
    sources = _sources(3)
    cf = Citeformer(backend=hf_backend, citation_policy=Policy.REQUIRED)
    result = cf.generate(
        prompt="Write one sentence referencing the books.",
        sources=sources,
        max_new_tokens=80,
        temperature=0.7,
    )
    emitted_ids = [int(m.group(1)) for m in _CITE.finditer(result.text)]
    # Empty is fine (generation might terminate before a cite fires) but any
    # emitted id MUST be in-range.
    for cid in emitted_ids:
        assert 1 <= cid <= 3, f"FABRICATED citation id {cid} in text: {result.text!r}"


@pytest.mark.integration
def test_hf_backend_cannot_fabricate_citations_auto_policy(hf_backend) -> None:  # type: ignore[no-untyped-def]
    """AUTO policy allows uncited prose, but any emitted cite must be valid."""
    sources = _sources(2)
    cf = Citeformer(backend=hf_backend, citation_policy=Policy.AUTO)
    result = cf.generate(
        prompt="Describe these books briefly.",
        sources=sources,
        max_new_tokens=60,
        temperature=0.7,
    )
    emitted_ids = [int(m.group(1)) for m in _CITE.finditer(result.text)]
    for cid in emitted_ids:
        assert 1 <= cid <= 2, f"FABRICATED citation id {cid} in text: {result.text!r}"


@pytest.mark.integration
def test_hf_backend_rejects_empty_sources(hf_backend) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="at least 1 source"):
        hf_backend.generate(
            prompt="anything",
            sources=[],
            policy=Policy.AUTO,
        )


@pytest.mark.integration
def test_hf_backend_required_with_tight_bound_emits_citations(hf_backend) -> None:  # type: ignore[no-untyped-def]
    """ADR-009: bounded `content` forces progression on small models.

    Pre-ADR-009, REQUIRED on a small model at modest `max_new_tokens` often
    produced zero citations because the unbounded `content ::= ... +` let the
    model stall in content state. With `max_content_chars=16` the grammar
    hard-masks everything except `[` after 16 non-terminating chars, so at
    least one citation must appear within the token budget.

    This test uses `gpt2` which lacks a chat template and happily runs on
    indefinitely — the exact stall-prone shape that exposed the old bug.
    """
    sources = _sources(3)
    cf = Citeformer(backend=hf_backend, citation_policy=Policy.REQUIRED)
    result = cf.generate(
        prompt="The books discuss:",
        sources=sources,
        max_new_tokens=120,
        temperature=0.7,
        max_content_chars=16,  # tight bound exercises the fix quickly
    )
    emitted_ids = [int(m.group(1)) for m in _CITE.finditer(result.text)]
    assert emitted_ids, (
        f"ADR-009 regression: expected ≥1 citation with max_content_chars=16, "
        f"got 0 in text: {result.text!r}"
    )
    for cid in emitted_ids:
        assert 1 <= cid <= 3, f"FABRICATED citation id {cid} in text: {result.text!r}"


@pytest.mark.integration
def test_hf_backend_stream_yields_multiple_chunks_and_matches_generate(hf_backend) -> None:  # type: ignore[no-untyped-def]
    """Streamed output joined should equal what generate() returns.

    The XGrammar LogitsProcessor is stateful per call; this asserts that the
    stream() path (which runs generate() in a background thread via
    TextIteratorStreamer) honors the same grammar constraints.
    """
    sources = _sources(3)
    cf = Citeformer(backend=hf_backend, citation_policy=Policy.AUTO)

    stream = cf.stream(
        prompt="Two short sentences:",
        sources=sources,
        max_new_tokens=40,
        temperature=0.0,  # deterministic
    )
    chunks = list(stream)
    assert len(chunks) > 1, f"expected >1 chunk, got {chunks!r}"
    result = stream.finalize()

    # Cross-check: the streamed text should match what generate() produces
    # when given the same seed-less deterministic setup (temperature=0).
    direct = cf.generate(
        prompt="Two short sentences:",
        sources=sources,
        max_new_tokens=40,
        temperature=0.0,
    )
    assert result.text == direct.text
    # And any emitted cite is still in-range — the structural guarantee applies
    # to streams exactly as it does to non-streamed generate().
    for cite in result.citations:
        assert 1 <= cite.source_id <= 3


@pytest.mark.integration
def test_hf_backend_compiler_caches_across_calls(hf_backend) -> None:  # type: ignore[no-untyped-def]
    """Two generate() calls with the same (n_sources, policy) should hit the compiler
    cache rather than recompile the grammar.

    Order-insensitive: we just assert that two calls with the same grammar
    leave the cache size unchanged between them. Doesn't matter whether the
    grammar was already cached from previous tests or freshly compiled.
    """
    sources = _sources(3)
    cf = Citeformer(backend=hf_backend, citation_policy=Policy.REQUIRED)

    cf.generate(prompt="short.", sources=sources, max_new_tokens=8)
    size_after_first = hf_backend._compiler.get_cache_size_bytes()
    cf.generate(prompt="another.", sources=sources, max_new_tokens=8)
    size_after_second = hf_backend._compiler.get_cache_size_bytes()
    # Second call with the same (n_sources, policy) → cache hit, no growth.
    assert size_after_second == size_after_first, (
        f"Expected cache hit; grew {size_after_first} → {size_after_second}"
    )
