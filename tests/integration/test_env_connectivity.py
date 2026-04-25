"""Env-var connectivity smoke for every API backend.

A single tiny call per backend, gated on the matching API key being
present in the env. Each test:

1. Skips cleanly if the key isn't set (CI without secrets stays green).
2. Issues the smallest possible request (1 source, ~80 max tokens).
3. Asserts the structural §10.1 invariant — every emitted cite id in
   ``[1..N]`` — end-to-end against the live provider.
4. Asserts ``last_usage`` populates with non-zero token counts so the
   ADR-012 token-accounting contract is verified live, not just under
   fakes.

Cost is roughly $0.01 total across all 5 API backends per full pass.

Run with ``make test-integration`` or
``uv run pytest -m integration tests/integration/test_env_connectivity.py``.

This file is the "did we wire any of the env vars wrong?" check —
useful before pushing a release tag, and cheap enough to run after any
backend or core change.
"""

from __future__ import annotations

import os
import re

import pytest

from citeformer import Policy, Source

_PROBE_SOURCE = Source(
    metadata={
        "id": "probe",
        "type": "book",
        "title": "Connectivity Probe",
        "author": [{"family": "Probe"}],
        "issued": {"date-parts": [[2026]]},
    },
    content=(
        "This is a one-source connectivity probe used by the citeformer "
        "test suite. The model should cite source [1] when summarising it."
    ),
)
_PROBE_PROMPT = "In one sentence, summarise the source content."

_CITE_RE = re.compile(r"\[(\d+)\]")


def _structural_check(text: str, n_sources: int) -> None:
    """Assert every cite id in ``text`` falls inside ``[1..n_sources]``."""
    ids = {int(m.group(1)) for m in _CITE_RE.finditer(text)}
    assert ids, f"expected at least one cite in {text!r}"
    assert ids.issubset(set(range(1, n_sources + 1))), (
        f"out-of-range cite id(s): {ids - set(range(1, n_sources + 1))}"
    )


def _usage_check(backend: object) -> None:
    """Assert the backend populated ``last_usage`` with non-zero counts."""
    usage = getattr(backend, "last_usage", None)
    assert usage is not None, "backend.last_usage was not populated"
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


# ---- per-backend smokes ----------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_connectivity_openai() -> None:
    from citeformer.backends.openai import OpenAIBackend

    backend = OpenAIBackend(model="gpt-4o-mini")
    text = backend.generate(
        prompt=_PROBE_PROMPT,
        sources=[_PROBE_SOURCE],
        policy=Policy.REQUIRED,
        max_tokens=80,
    )
    _structural_check(text, n_sources=1)
    _usage_check(backend)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_connectivity_anthropic() -> None:
    from citeformer.backends.anthropic import AnthropicBackend

    backend = AnthropicBackend(model="claude-haiku-4-5-20251001")
    text = backend.generate(
        prompt=_PROBE_PROMPT,
        sources=[_PROBE_SOURCE],
        policy=Policy.REQUIRED,
        max_tokens=80,
        temperature=0.0,
    )
    _structural_check(text, n_sources=1)
    _usage_check(backend)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_connectivity_anthropic_streaming_yields_chunks() -> None:
    """Real per-block streaming should emit at least one chunk and
    populate ``last_usage`` from ``get_final_message()``."""
    from citeformer.backends.anthropic import AnthropicBackend

    backend = AnthropicBackend(model="claude-haiku-4-5-20251001")
    chunks = list(
        backend.stream(
            prompt=_PROBE_PROMPT,
            sources=[_PROBE_SOURCE],
            policy=Policy.REQUIRED,
            max_tokens=80,
            temperature=0.0,
        )
    )
    assert chunks, "stream produced no chunks"
    text = "".join(chunks)
    _structural_check(text, n_sources=1)
    _usage_check(backend)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_connectivity_openrouter() -> None:
    from citeformer.backends.openrouter import OpenRouterBackend

    backend = OpenRouterBackend(
        model="openai/gpt-4o-mini",
        app_name="citeformer-connectivity-test",
        app_url="https://github.com/random-walks/citeformer",
    )
    text = backend.generate(
        prompt=_PROBE_PROMPT,
        sources=[_PROBE_SOURCE],
        policy=Policy.REQUIRED,
        max_tokens=80,
    )
    _structural_check(text, n_sources=1)
    _usage_check(backend)
    # OpenRouter reports per-call cost (in OR credits, not USD) on every
    # response since structured-output GA — assert it lands.
    assert backend.last_usage is not None
    assert backend.last_usage.cost_credits is not None
    assert backend.last_usage.cost_credits >= 0.0


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY / GOOGLE_API_KEY not set",
)
def test_connectivity_gemini() -> None:
    from citeformer.backends.gemini import GeminiBackend

    backend = GeminiBackend(model="gemini-2.0-flash")
    text = backend.generate(
        prompt=_PROBE_PROMPT,
        sources=[_PROBE_SOURCE],
        policy=Policy.REQUIRED,
        max_tokens=80,
    )
    _structural_check(text, n_sources=1)
    _usage_check(backend)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set",
)
def test_connectivity_mistral() -> None:
    from citeformer.backends.mistral import MistralBackend

    backend = MistralBackend(model="mistral-small-latest")
    text = backend.generate(
        prompt=_PROBE_PROMPT,
        sources=[_PROBE_SOURCE],
        policy=Policy.REQUIRED,
        max_tokens=80,
    )
    _structural_check(text, n_sources=1)
    _usage_check(backend)
