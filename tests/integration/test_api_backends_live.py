"""Live integration tests for the OpenAI + Anthropic backends.

These hit real provider endpoints, cost real money (cents per run), and
are gated three ways:

1. ``@pytest.mark.integration`` — excluded from the default pytest run;
   you have to opt in via ``pytest -m integration`` / ``make test-integration``.
2. Env-var skip — if the relevant key isn't set the test is *skipped*,
   not failed, so CI without a key still passes cleanly.
3. Smallest reasonable models (``gpt-4o-mini``, ``claude-haiku-4-5``)
   and short max_tokens so a full pass is <$0.10.

What we're verifying end-to-end against the real APIs:

- The structural invariant — **every emitted cite id is in [1..N]**.
  This is the same guarantee the unit suite checks with fake clients,
  but here we're proving the live provider actually honours the JSON
  schema (OpenAI) / native Citations API (Anthropic).
- Policy semantics — ``REQUIRED`` produces at least one citation;
  ``AUTO`` may omit them on trivial questions.
- The ``Citeformer`` orchestrator wires the backend output through
  parsing, rendering, and reference construction end-to-end with no
  surprises at the boundary.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator

import pytest

from citeformer import Citeformer, Policy, Source

# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def sources() -> list[Source]:
    """Three classic-literature sources with full CSL metadata.

    Short abstracts so the provider can reason cheaply; known canonical
    authors so APA-7 rendering is trivially verifiable.
    """
    return [
        Source(
            metadata={
                "id": "poe",
                "type": "book",
                "title": "The Raven",
                "author": [{"family": "Poe", "given": "Edgar Allan"}],
                "issued": {"date-parts": [[1845]]},
            },
            content=(
                "Once upon a midnight dreary, while I pondered, weak and "
                "weary, over many a quaint and curious volume of forgotten lore."
            ),
        ),
        Source(
            metadata={
                "id": "mel",
                "type": "book",
                "title": "Moby-Dick",
                "author": [{"family": "Melville", "given": "Herman"}],
                "issued": {"date-parts": [[1851]]},
            },
            content=(
                "Call me Ishmael. Some years ago—never mind how long "
                "precisely—having little or no money in my purse, I thought "
                "I would sail about a little."
            ),
        ),
        Source(
            metadata={
                "id": "aus",
                "type": "book",
                "title": "Pride and Prejudice",
                "author": [{"family": "Austen", "given": "Jane"}],
                "issued": {"date-parts": [[1813]]},
            },
            content=(
                "It is a truth universally acknowledged, that a single man "
                "in possession of a good fortune, must be in want of a wife."
            ),
        ),
    ]


def _extract_bracketed_ids(text: str) -> set[int]:
    return {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", text)}


# --- OpenAI --------------------------------------------------------------


requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

# Cheapest model that supports strict JSON schema mode (Aug-2024+).
_OPENAI_MODEL = "gpt-4o-mini"


@pytest.fixture(scope="module")
def openai_cf() -> Iterator[Citeformer]:
    from citeformer.backends.openai import OpenAIBackend

    backend = OpenAIBackend(model=_OPENAI_MODEL)
    yield Citeformer(backend=backend, style="apa-7", citation_policy=Policy.REQUIRED)


@pytest.mark.integration
@requires_openai
def test_openai_required_honours_structural_invariant(
    openai_cf: Citeformer, sources: list[Source]
) -> None:
    """Live structural check — every cite id must be in [1..N]."""
    result = openai_cf.generate(
        prompt="In 2 sentences, compare the opening tone of any two of these classics.",
        sources=sources,
        max_tokens=256,
    )
    ids = _extract_bracketed_ids(result.text)
    assert ids, f"Expected at least one bracketed cite under REQUIRED policy; got {result.text!r}"
    assert ids.issubset({1, 2, 3}), f"Out-of-scope cite id(s): {ids - {1, 2, 3}}"
    assert all(1 <= c.source_id <= 3 for c in result.citations)
    # References must match the set of cited ids — the coupling rule.
    ref_ids = {r.source_id for r in result.references}
    assert ref_ids == ids


@pytest.mark.integration
@requires_openai
def test_openai_auto_policy_minimum_zero(openai_cf: Citeformer, sources: list[Source]) -> None:
    """AUTO policy — min_citations=0, but any emitted cite is still bounded."""
    from citeformer.backends.openai import OpenAIBackend

    cf = Citeformer(
        backend=OpenAIBackend(model=_OPENAI_MODEL),
        style="apa-7",
        citation_policy=Policy.AUTO,
    )
    result = cf.generate(
        prompt="In one sentence, summarise when each book was published.",
        sources=sources,
        max_tokens=200,
    )
    ids = _extract_bracketed_ids(result.text)
    assert ids.issubset({1, 2, 3})


@pytest.mark.integration
@requires_openai
def test_openai_n_equal_one_still_works(sources: list[Source]) -> None:
    """Edge case — N=1 means the enum has a single entry. Strict-mode schema
    must still validate the response."""
    from citeformer.backends.openai import OpenAIBackend

    cf = Citeformer(
        backend=OpenAIBackend(model=_OPENAI_MODEL),
        style="apa-7",
        citation_policy=Policy.REQUIRED,
    )
    result = cf.generate(
        prompt="In one sentence, describe this book's opening line.",
        sources=[sources[0]],
        max_tokens=120,
    )
    ids = _extract_bracketed_ids(result.text)
    assert ids == {1}, f"N=1 must produce only [1], got {ids}"


# --- Anthropic -----------------------------------------------------------


requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

# Cheapest Claude 4.x model with Citations support.
_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@pytest.fixture(scope="module")
def anthropic_cf() -> Iterator[Citeformer]:
    from citeformer.backends.anthropic import AnthropicBackend

    backend = AnthropicBackend(model=_ANTHROPIC_MODEL)
    yield Citeformer(backend=backend, style="apa-7", citation_policy=Policy.REQUIRED)


@pytest.mark.integration
@requires_anthropic
def test_anthropic_required_honours_structural_invariant(
    anthropic_cf: Citeformer, sources: list[Source]
) -> None:
    """Live check against Anthropic's native Citations API — cites must map to supplied docs."""
    result = anthropic_cf.generate(
        prompt="In two sentences, describe the opening scenes of any two of these books.",
        sources=sources,
        max_tokens=300,
    )
    ids = _extract_bracketed_ids(result.text)
    assert ids, f"Expected at least one cite; got {result.text!r}"
    assert ids.issubset({1, 2, 3}), f"Out-of-scope cite id(s): {ids - {1, 2, 3}}"
    for c in result.citations:
        assert 1 <= c.source_id <= 3


@pytest.mark.integration
@requires_anthropic
def test_anthropic_reference_list_couples_with_citations(
    anthropic_cf: Citeformer, sources: list[Source]
) -> None:
    """Rendered bibliography must exactly match the set of cited ids."""
    result = anthropic_cf.generate(
        prompt="In one sentence, describe the opening of The Raven.",
        sources=sources,
        max_tokens=180,
    )
    cited_ids = {c.source_id for c in result.citations}
    ref_ids = {r.source_id for r in result.references}
    assert cited_ids == ref_ids, f"Decoupled: cited={cited_ids}, references={ref_ids}"


@pytest.mark.integration
@requires_anthropic
def test_anthropic_empty_sources_rejected(anthropic_cf: Citeformer) -> None:
    """The backend must reject zero-source calls before hitting the API."""
    with pytest.raises(ValueError, match="at least 1 source"):
        anthropic_cf.generate(prompt="hi", sources=[])


# --- Gemini --------------------------------------------------------------


requires_gemini = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY / GOOGLE_API_KEY not set",
)

_GEMINI_MODEL = "gemini-2.0-flash"


@pytest.fixture(scope="module")
def gemini_cf() -> Iterator[Citeformer]:
    from citeformer.backends.gemini import GeminiBackend

    yield Citeformer(
        backend=GeminiBackend(model=_GEMINI_MODEL),
        style="apa-7",
        citation_policy=Policy.REQUIRED,
    )


@pytest.mark.integration
@requires_gemini
def test_gemini_required_honours_structural_invariant(
    gemini_cf: Citeformer, sources: list[Source]
) -> None:
    result = gemini_cf.generate(
        prompt="In 2 sentences, compare the opening tone of any two of these classics.",
        sources=sources,
        max_tokens=256,
    )
    ids = _extract_bracketed_ids(result.text)
    assert ids, f"Expected at least one cite; got {result.text!r}"
    assert ids.issubset({1, 2, 3}), f"Out-of-scope cite id(s): {ids - {1, 2, 3}}"


# --- Mistral -------------------------------------------------------------


requires_mistral = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set",
)

_MISTRAL_MODEL = "mistral-small-latest"


@pytest.fixture(scope="module")
def mistral_cf() -> Iterator[Citeformer]:
    from citeformer.backends.mistral import MistralBackend

    yield Citeformer(
        backend=MistralBackend(model=_MISTRAL_MODEL),
        style="apa-7",
        citation_policy=Policy.REQUIRED,
    )


@pytest.mark.integration
@requires_mistral
def test_mistral_required_honours_structural_invariant(
    mistral_cf: Citeformer, sources: list[Source]
) -> None:
    result = mistral_cf.generate(
        prompt="In 2 sentences, compare the opening tone of any two of these classics.",
        sources=sources,
        max_tokens=256,
    )
    ids = _extract_bracketed_ids(result.text)
    assert ids, f"Expected at least one cite; got {result.text!r}"
    assert ids.issubset({1, 2, 3}), f"Out-of-scope cite id(s): {ids - {1, 2, 3}}"
