"""Cross-backend conformance tests.

The §10.1 / §10.3 contracts require that every backend produces output
with the same structural invariants — cite ids in [1..N], policies
honoured, marker styles propagated. The per-backend unit tests cover
each one in isolation; this file is the shared grid that proves they
agree.

Local backends (HF / vLLM / llama.cpp) load real models and live in
``tests/integration/``. The conformance grid here covers every backend
that can run with a fake / scripted client: ``MockBackend``,
``OpenAIBackend``, ``AnthropicBackend``, ``GeminiBackend``,
``MistralBackend``, ``OpenRouterBackend``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from citeformer import Citeformer, MarkerStyle, Policy, Source
from citeformer.backends import Backend, MockBackend
from citeformer.backends.anthropic import AnthropicBackend
from citeformer.backends.fireworks import FireworksBackend
from citeformer.backends.gemini import GeminiBackend
from citeformer.backends.mistral import MistralBackend
from citeformer.backends.openai import OpenAIBackend
from citeformer.backends.openrouter import OpenRouterBackend
from citeformer.backends.together import TogetherBackend

# ---- Fake-client factories -------------------------------------------------


def _fake_openai_client(payload: str, usage: Any = None) -> Any:
    """Build a fake ``openai.OpenAI`` client returning the given JSON body."""
    fake = SimpleNamespace()
    fake.chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=payload)),
                ],
                usage=usage,
            )
        )
    )
    return fake


def _fake_anthropic_client(text: str, n_sources: int) -> Any:
    """Build a fake Anthropic client emitting a single text block per source.

    We emit one block citing each source so REQUIRED-mode assertions ("at least
    one cite per scope") and structural assertions (every cite in 1..N) can
    both run against the same response shape.
    """
    blocks = [
        SimpleNamespace(
            type="text",
            text=text,
            citations=[SimpleNamespace(document_index=i) for i in range(n_sources)],
        ),
    ]
    fake = SimpleNamespace()
    fake.messages = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            content=blocks,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
    )
    return fake


def _fake_gemini_client(payload: str) -> Any:
    fake = SimpleNamespace()
    fake.models = SimpleNamespace(
        generate_content=lambda **kwargs: SimpleNamespace(
            text=payload,
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        )
    )
    return fake


def _fake_mistral_client(payload: str) -> Any:
    fake = SimpleNamespace()
    fake.chat = SimpleNamespace(
        complete=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
    )
    return fake


# ---- Per-backend factories that return a ready-to-use Backend instance ----

#: ``BackendFactory(n_sources)`` returns a ``Backend`` whose mock response
#: emits exactly one citation marker per source under the BRACKET style — so
#: every backend feeds the conformance asserts the same structural shape.
BackendFactory = Callable[[int], Backend]


def _segments_payload(n_sources: int) -> str:
    return json.dumps(
        {
            "segments": [
                {"text": f"Claim about source {i}.", "citations": [i]}
                for i in range(1, n_sources + 1)
            ]
        }
    )


def _make_mock(n_sources: int) -> Backend:
    """MockBackend's fallback-echo path emits one marker honouring ``marker_style``
    when at least one source is in scope, and nothing when ``sources=[]`` —
    perfect for the structural / marker-style / empty-source asserts.
    """
    del n_sources  # the mock fallback always emits a single [1]
    return MockBackend()


def _make_openai(n_sources: int) -> Backend:
    return OpenAIBackend(
        model="gpt-4o-mini",
        client=_fake_openai_client(
            _segments_payload(n_sources),
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=12),
        ),
    )


def _make_openrouter(n_sources: int) -> Backend:
    return OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=_fake_openai_client(
            _segments_payload(n_sources),
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=12, cost=0.001),
        ),
    )


def _make_anthropic(n_sources: int) -> Backend:
    return AnthropicBackend(
        model="claude-sonnet-4-6",
        client=_fake_anthropic_client("Claim about each source.", n_sources=n_sources),
    )


def _make_gemini(n_sources: int) -> Backend:
    return GeminiBackend(
        model="gemini-2.0-flash",
        client=_fake_gemini_client(_segments_payload(n_sources)),
    )


def _make_mistral(n_sources: int) -> Backend:
    return MistralBackend(
        model="mistral-large-latest",
        client=_fake_mistral_client(_segments_payload(n_sources)),
    )


def _make_together(n_sources: int) -> Backend:
    """Together is OpenAI-wire-compatible — same segments payload works."""
    return TogetherBackend(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        client=_fake_openai_client(
            _segments_payload(n_sources),
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=12),
        ),
    )


def _make_fireworks(n_sources: int) -> Backend:
    """Fireworks uses native GBNF — its response is plain text with markers,
    not a segments JSON. The fake client *inspects the grammar payload* and
    emits text with matching delimiters, simulating provider-side
    grammar-constrained sampling. Lets the cross-backend conformance grid
    exercise marker-style propagation just like the other backends."""

    def _create(**kwargs: Any) -> Any:
        grammar = (kwargs.get("response_format") or {}).get("grammar", "")
        open_d, close_d = "[", "]"
        if '"("' in grammar and '")"' in grammar:
            open_d, close_d = "(", ")"
        elif '"{"' in grammar and '"}"' in grammar:
            open_d, close_d = "{", "}"
        elif '"^"' in grammar:
            open_d, close_d = "^", ""
        text = " ".join(f"Claim {i} {open_d}{i}{close_d}." for i in range(1, n_sources + 1))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=12),
        )

    fake = SimpleNamespace()
    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))
    return FireworksBackend(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        client=fake,
    )


ALL_BACKENDS: list[tuple[str, BackendFactory]] = [
    ("mock", _make_mock),
    ("openai", _make_openai),
    ("openrouter", _make_openrouter),
    ("anthropic", _make_anthropic),
    ("gemini", _make_gemini),
    ("mistral", _make_mistral),
    ("together", _make_together),
    ("fireworks", _make_fireworks),
]

#: Subset of ``ALL_BACKENDS`` that hit a remote API and therefore populate
#: ``last_usage`` (the contract from ADR-012). MockBackend doesn't.
API_BACKENDS: list[tuple[str, BackendFactory]] = [
    (name, factory) for name, factory in ALL_BACKENDS if name != "mock"
]

_CITE_RE = re.compile(r"\[(\d+)\]")


def _all_cite_ids(text: str) -> list[int]:
    return [int(m.group(1)) for m in _CITE_RE.finditer(text)]


# ---- Conformance grid ------------------------------------------------------


@pytest.fixture
def sources() -> list[Source]:
    return [
        Source(metadata={"id": f"s{i}", "type": "book"}, content=f"Body {i}.") for i in range(1, 4)
    ]


@pytest.mark.parametrize("name,factory", ALL_BACKENDS)
def test_backend_emits_only_in_range_cite_ids(
    name: str,
    factory: BackendFactory,
    sources: list[Source],
) -> None:
    """§10.1 — every cite id must be in [1..N]. Universal across backends."""
    del name
    backend = factory(len(sources))
    text = backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    ids = _all_cite_ids(text)
    assert ids, f"Expected at least one cite in {text!r}"
    assert all(1 <= i <= len(sources) for i in ids), (
        f"Out-of-range cite ids: {[i for i in ids if i < 1 or i > len(sources)]}"
    )


@pytest.mark.parametrize("name,factory", ALL_BACKENDS)
def test_backend_rejects_empty_sources(
    name: str,
    factory: BackendFactory,
) -> None:
    """Every backend except MockBackend MUST reject empty sources before
    hitting the network. MockBackend echoes harmlessly when there's nothing
    to cite, so it gets a free pass.
    """
    backend = factory(1)
    if name == "mock":
        # MockBackend is by design lenient — exercising the no-cite echo is
        # the equivalent assert.
        text = backend.generate(prompt="hi", sources=[], policy=Policy.AUTO)
        assert "[" not in text
        return
    with pytest.raises(ValueError, match="at least 1 source"):
        backend.generate(prompt="hi", sources=[], policy=Policy.AUTO)


@pytest.mark.parametrize("name,factory", API_BACKENDS)
def test_api_backend_populates_last_usage(
    name: str,
    factory: BackendFactory,
    sources: list[Source],
) -> None:
    """ADR-012 — every API backend sets ``last_usage`` so the orchestrator can
    surface it on ``GenerationResult.usage``."""
    del name
    backend = factory(len(sources))
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is not None  # type: ignore[attr-defined]
    usage = backend.last_usage  # type: ignore[attr-defined]
    assert usage.input_tokens >= 0
    assert usage.output_tokens >= 0


@pytest.mark.parametrize("name,factory", ALL_BACKENDS)
def test_orchestrator_threads_usage_onto_generation_result(
    name: str,
    factory: BackendFactory,
    sources: list[Source],
) -> None:
    """End-to-end — through the orchestrator, ``result.usage`` is set on
    API backends and ``None`` on local/mock backends. Same orchestrator
    code path either way."""
    backend = factory(len(sources))
    cf = Citeformer(backend=backend, citation_policy=Policy.AUTO)
    result = cf.generate(prompt="hi", sources=sources)
    if name == "mock":
        assert result.usage is None
    else:
        assert result.usage is not None
        assert result.usage.input_tokens >= 0


@pytest.mark.parametrize("name,factory", ALL_BACKENDS)
@pytest.mark.parametrize(
    "marker_style,open_char,close_char",
    [
        (MarkerStyle.BRACKET, "[", "]"),
        (MarkerStyle.PAREN, "(", ")"),
        (MarkerStyle.CURLY, "{", "}"),
        (MarkerStyle.CARET, "^", ""),
    ],
)
def test_backend_marker_style_propagates(
    name: str,
    factory: BackendFactory,
    sources: list[Source],
    marker_style: MarkerStyle,
    open_char: str,
    close_char: str,
) -> None:
    """§10.1 invariant — marker style must round-trip through every backend.

    For backends whose mock fixtures emit bracket-shaped citations natively
    (the segments-based payload uses bracket-style integers in a JSON
    array, then the flattener wraps them in the requested marker style),
    swapping ``marker_style`` should change the wrapper characters in the
    final output.
    """
    del name
    backend = factory(len(sources))
    text = backend.generate(
        prompt="hi",
        sources=sources,
        policy=Policy.AUTO,
        marker_style=marker_style,
    )
    # Must contain at least the first source wrapped in the requested style.
    assert f"{open_char}1{close_char}" in text


@pytest.mark.parametrize("name,factory", ALL_BACKENDS)
def test_backend_streaming_finalizes_to_same_text(
    name: str,
    factory: BackendFactory,
    sources: list[Source],
) -> None:
    """``Citeformer.stream().finalize()`` must produce a result whose text
    contains the same cite ids as the non-streaming ``generate()`` path."""
    # Build two backend instances — the mocks have client-side state we don't
    # want to cross-contaminate between calls.
    backend_gen = factory(len(sources))
    backend_stream = factory(len(sources))

    cf_gen = Citeformer(backend=backend_gen, citation_policy=Policy.AUTO)
    cf_stream = Citeformer(backend=backend_stream, citation_policy=Policy.AUTO)

    gen_result = cf_gen.generate(prompt="hi", sources=sources)
    stream_result = cf_stream.stream(prompt="hi", sources=sources).finalize()

    # Cite ids must match — exact text may not (stream/non-stream wrap whitespace
    # slightly differently across backends).
    assert set(c.source_id for c in gen_result.citations) == set(
        c.source_id for c in stream_result.citations
    ), f"{name}: gen ids != stream ids"
