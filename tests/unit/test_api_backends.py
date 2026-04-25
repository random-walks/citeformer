"""Unit tests for OpenAIBackend + AnthropicBackend with mocked clients.

Real API calls happen in integration tests (marked ``integration``) and only
run when the caller sets ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``. These
unit tests drive the clients through shaped stand-ins so we can verify the
schema / message construction and the response-parsing path without a
network call or an API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from citeformer import MarkerStyle, Policy, Source
from citeformer.backends.anthropic import AnthropicBackend
from citeformer.backends.openai import (
    OpenAIBackend,
    _build_citation_schema,
    _flatten_segments,
)

# ----- Shared fixtures -------------------------------------------------------


@pytest.fixture
def sources() -> list[Source]:
    return [
        Source(
            metadata={"id": f"s{i}", "type": "book", "title": f"Source {i}"},
            content=f"Body text for source {i}.",
        )
        for i in range(1, 4)
    ]


# ----- OpenAI schema construction -------------------------------------------


def test_openai_schema_enum_bounded_by_n_sources() -> None:
    schema = _build_citation_schema(n_sources=3, policy=Policy.AUTO)
    props = schema["properties"]["segments"]["items"]["properties"]
    assert props["citations"]["items"]["enum"] == [1, 2, 3]


def test_openai_schema_min_items_depends_on_policy() -> None:
    strict = _build_citation_schema(n_sources=3, policy=Policy.REQUIRED)
    loose = _build_citation_schema(n_sources=3, policy=Policy.AUTO)
    segs_strict = strict["properties"]["segments"]["items"]
    segs_loose = loose["properties"]["segments"]["items"]
    assert segs_strict["properties"]["citations"]["minItems"] == 1
    assert segs_loose["properties"]["citations"]["minItems"] == 0


def test_openai_schema_additional_properties_is_false() -> None:
    """Strict mode requires additionalProperties: false at every level."""
    schema = _build_citation_schema(n_sources=2, policy=Policy.AUTO)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["segments"]["items"]["additionalProperties"] is False


# ----- OpenAI output flattening ---------------------------------------------


def test_flatten_segments_bracket_style_default() -> None:
    raw = json.dumps(
        {
            "segments": [
                {"text": "First claim.", "citations": [1]},
                {"text": "Second claim.", "citations": [2, 3]},
            ]
        }
    )
    out = _flatten_segments(raw, marker_style=MarkerStyle.BRACKET)
    assert "First claim [1]." in out
    assert "Second claim [2][3]." in out


@pytest.mark.parametrize(
    "marker_style,expected_marker",
    [
        (MarkerStyle.BRACKET, "[1]"),
        (MarkerStyle.PAREN, "(1)"),
        (MarkerStyle.CURLY, "{1}"),
        (MarkerStyle.CARET, "^1"),
    ],
)
def test_flatten_segments_honours_marker_style(
    marker_style: MarkerStyle, expected_marker: str
) -> None:
    raw = json.dumps({"segments": [{"text": "Claim.", "citations": [1]}]})
    out = _flatten_segments(raw, marker_style=marker_style)
    assert expected_marker in out


def test_flatten_segments_returns_text_when_payload_invalid() -> None:
    """A non-JSON payload returns verbatim rather than blowing up."""
    out = _flatten_segments("not-json-at-all", marker_style=MarkerStyle.BRACKET)
    assert out == "not-json-at-all"


def test_flatten_segments_tolerates_empty_citations() -> None:
    raw = json.dumps({"segments": [{"text": "Uncited claim.", "citations": []}]})
    out = _flatten_segments(raw, marker_style=MarkerStyle.BRACKET)
    assert out == "Uncited claim."


# ----- OpenAI end-to-end with mocked client ---------------------------------


class _FakeOpenAI:
    """Minimal stand-in for the ``openai.OpenAI`` client."""

    def __init__(self, response_content: str) -> None:
        self._response = response_content
        self.last_payload: dict[str, Any] | None = None

        def _create(**kwargs: Any) -> Any:
            self.last_payload = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response_content))]
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


def test_openai_backend_end_to_end_maps_segments_to_markers(
    sources: list[Source],
) -> None:
    fake_payload = json.dumps(
        {
            "segments": [
                {"text": "Alpha claim.", "citations": [1]},
                {"text": "Beta claim.", "citations": [2]},
            ]
        }
    )
    fake = _FakeOpenAI(fake_payload)
    backend = OpenAIBackend(model="gpt-4o-mini", client=fake)
    text = backend.generate(prompt="Describe the sources.", sources=sources, policy=Policy.AUTO)
    assert "[1]" in text
    assert "[2]" in text
    # Grammar-equivalent enforcement: schema's enum matches the sources.
    payload = fake.last_payload
    assert payload is not None
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["segments"]["items"]["properties"]["citations"]["items"][
        "enum"
    ] == [1, 2, 3]


def test_openai_backend_sends_strict_true(sources: list[Source]) -> None:
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = OpenAIBackend(model="gpt-4o-mini", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    payload = fake.last_payload
    assert payload is not None
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_openai_backend_embeds_sources_in_system_prompt(
    sources: list[Source],
) -> None:
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = OpenAIBackend(model="gpt-4o-mini", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    payload = fake.last_payload
    assert payload is not None
    system = payload["messages"][0]["content"]
    # Each source should appear in the system text with its 1-indexed marker.
    for i in range(1, len(sources) + 1):
        assert f"[{i}]" in system


def test_openai_backend_rejects_empty_sources() -> None:
    fake = _FakeOpenAI("{}")
    backend = OpenAIBackend(model="gpt-4o-mini", client=fake)
    with pytest.raises(ValueError, match="at least 1 source"):
        backend.generate(prompt="hi", sources=[], policy=Policy.AUTO)


def test_openai_stream_yields_multiple_chunks(sources: list[Source]) -> None:
    fake = _FakeOpenAI(
        json.dumps(
            {
                "segments": [
                    {"text": "First claim.", "citations": [1]},
                    {"text": "Second claim.", "citations": [2]},
                ]
            }
        )
    )
    backend = OpenAIBackend(model="gpt-4o-mini", client=fake)
    chunks = list(backend.stream(prompt="hi", sources=sources, policy=Policy.AUTO))
    assert len(chunks) >= 2


# ----- Anthropic adapter -----------------------------------------------------


class _FakeAnthropic:
    """Stand-in for the ``anthropic.Anthropic`` client."""

    def __init__(self, content_blocks: list[Any], usage: Any = None) -> None:
        self._blocks = content_blocks
        self._usage = usage
        self.last_kwargs: dict[str, Any] | None = None

        def _create(**kwargs: Any) -> Any:
            self.last_kwargs = kwargs
            return SimpleNamespace(content=self._blocks, usage=self._usage)

        self.messages = SimpleNamespace(create=_create)


def test_anthropic_backend_maps_document_index_to_cite_id(
    sources: list[Source],
) -> None:
    # Two text blocks, each citing one document (0-indexed → 1-indexed).
    blocks = [
        SimpleNamespace(
            type="text",
            text="Alpha claim.",
            citations=[SimpleNamespace(document_index=0)],
        ),
        SimpleNamespace(
            type="text",
            text="Beta claim.",
            citations=[SimpleNamespace(document_index=2)],
        ),
    ]
    fake = _FakeAnthropic(blocks)
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    text = backend.generate(prompt="Describe.", sources=sources, policy=Policy.AUTO)
    assert "[1]" in text
    assert "[3]" in text
    # Sources must have been threaded through as documents with citations enabled.
    kwargs = fake.last_kwargs
    assert kwargs is not None
    docs = [
        item
        for item in kwargs["messages"][0]["content"]
        if isinstance(item, dict) and item.get("type") == "document"
    ]
    assert len(docs) == 3
    assert all(d["citations"] == {"enabled": True} for d in docs)


def test_anthropic_backend_accepts_dict_shaped_blocks(sources: list[Source]) -> None:
    """Some SDKs / serialisations return dicts instead of model objects."""
    blocks = [
        {
            "type": "text",
            "text": "Claim one.",
            "citations": [{"document_index": 0}, {"document_index": 1}],
        },
    ]
    fake = _FakeAnthropic(blocks)
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    text = backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "[1]" in text
    assert "[2]" in text


def test_anthropic_backend_skips_text_blocks_without_citations(
    sources: list[Source],
) -> None:
    blocks = [
        SimpleNamespace(type="text", text="Uncited prose.", citations=None),
    ]
    fake = _FakeAnthropic(blocks)
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    text = backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "[" not in text
    assert "Uncited prose." in text


@pytest.mark.parametrize(
    "marker_style,open_char,close_char",
    [
        (MarkerStyle.BRACKET, "[", "]"),
        (MarkerStyle.PAREN, "(", ")"),
        (MarkerStyle.CURLY, "{", "}"),
        (MarkerStyle.CARET, "^", ""),
    ],
)
def test_anthropic_backend_honours_marker_style(
    sources: list[Source],
    marker_style: MarkerStyle,
    open_char: str,
    close_char: str,
) -> None:
    blocks = [
        SimpleNamespace(
            type="text",
            text="Claim.",
            citations=[SimpleNamespace(document_index=0)],
        ),
    ]
    fake = _FakeAnthropic(blocks)
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    text = backend.generate(
        prompt="hi",
        sources=sources,
        policy=Policy.AUTO,
        marker_style=marker_style,
    )
    assert f"{open_char}1{close_char}" in text


def test_anthropic_backend_rejects_empty_sources() -> None:
    fake = _FakeAnthropic([])
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    with pytest.raises(ValueError, match="at least 1 source"):
        backend.generate(prompt="hi", sources=[], policy=Policy.AUTO)


def test_anthropic_backend_dedupes_repeated_document_index(
    sources: list[Source],
) -> None:
    """A block citing the same doc twice → single marker emitted."""
    blocks = [
        SimpleNamespace(
            type="text",
            text="Claim.",
            citations=[
                SimpleNamespace(document_index=1),
                SimpleNamespace(document_index=1),
            ],
        ),
    ]
    fake = _FakeAnthropic(blocks)
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    text = backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert text.count("[2]") == 1


# ----- Anthropic prompt caching ---------------------------------------------


def test_anthropic_backend_sets_cache_control_on_documents_by_default(
    sources: list[Source],
) -> None:
    """Document blocks carry ``cache_control: ephemeral`` so repeat-source
    RAG bills cache-read prices on subsequent calls."""
    fake = _FakeAnthropic([])
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    docs = [
        item
        for item in fake.last_kwargs["messages"][0]["content"]
        if isinstance(item, dict) and item.get("type") == "document"
    ]
    assert len(docs) == 3
    assert all(d["cache_control"] == {"type": "ephemeral"} for d in docs)


def test_anthropic_backend_omits_cache_control_when_opted_out(
    sources: list[Source],
) -> None:
    """``use_prompt_cache=False`` sends the document blocks without
    cache_control — useful for truly one-shot calls where caching is overhead."""
    fake = _FakeAnthropic([])
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    backend.generate(
        prompt="hi", sources=sources, policy=Policy.AUTO, use_prompt_cache=False
    )
    docs = [
        item
        for item in fake.last_kwargs["messages"][0]["content"]
        if isinstance(item, dict) and item.get("type") == "document"
    ]
    assert all("cache_control" not in d for d in docs)


# ----- Anthropic temperature ------------------------------------------------


def test_anthropic_backend_threads_temperature_when_supplied(
    sources: list[Source],
) -> None:
    """Pre-revamp the option was silently dropped — this asserts it's threaded
    through to the SDK call so users can actually control sampling temperature."""
    fake = _FakeAnthropic([])
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    backend.generate(
        prompt="hi", sources=sources, policy=Policy.AUTO, temperature=0.0
    )
    assert fake.last_kwargs["temperature"] == 0.0


def test_anthropic_backend_omits_temperature_when_not_supplied(
    sources: list[Source],
) -> None:
    """When the caller doesn't pass temperature, we don't put one on the
    request — Anthropic's own default applies."""
    fake = _FakeAnthropic([])
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "temperature" not in fake.last_kwargs


# ----- Anthropic usage extraction -------------------------------------------


def test_anthropic_backend_populates_last_usage_from_response(
    sources: list[Source],
) -> None:
    """``last_usage`` must be set after ``generate()`` so the orchestrator
    can thread it onto ``GenerationResult.usage``."""
    fake = _FakeAnthropic(
        [],
        usage=SimpleNamespace(
            input_tokens=512,
            output_tokens=128,
            cache_creation_input_tokens=400,
            cache_read_input_tokens=100,
        ),
    )
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 512
    assert backend.last_usage.output_tokens == 128
    assert backend.last_usage.cache_creation_input_tokens == 400
    assert backend.last_usage.cache_read_input_tokens == 100


def test_anthropic_backend_handles_usage_dict_shape(sources: list[Source]) -> None:
    """Some serialisations surface usage as a dict; both shapes must work."""
    fake = _FakeAnthropic(
        [],
        usage={"input_tokens": 256, "output_tokens": 64},
    )
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 256
    assert backend.last_usage.output_tokens == 64
    # Cache fields default to None when the provider didn't supply them.
    assert backend.last_usage.cache_creation_input_tokens is None
    assert backend.last_usage.cache_read_input_tokens is None


def test_anthropic_backend_last_usage_is_none_when_response_omits_usage(
    sources: list[Source],
) -> None:
    """A fake response without a usage attribute leaves ``last_usage = None``
    rather than crashing — defensive against test stubs and SDK drift."""
    fake = _FakeAnthropic([])
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is None


# ----- Anthropic real streaming ---------------------------------------------


class _FakeAnthropicStream:
    """Stand-in for the ``messages.stream(...)`` context manager.

    The real SDK returns an object that's iterable (yields typed events) and
    supports ``get_final_message()``. This stand-in replays a scripted
    sequence of ``content_block_stop`` events so our streaming path can be
    exercised without an SDK or a network call.
    """

    def __init__(self, events: list[Any], final_message: Any) -> None:
        self._events = events
        self._final_message = final_message
        self.last_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> _FakeAnthropicStream:
        self.last_kwargs = kwargs
        return self

    def __enter__(self) -> _FakeAnthropicStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self) -> Any:
        return iter(self._events)

    def get_final_message(self) -> Any:
        return self._final_message


class _FakeAnthropicWithStream:
    """Stand-in client exposing both ``create`` and ``stream``."""

    def __init__(self, *, stream: _FakeAnthropicStream) -> None:
        self._stream = stream

        def _create(**kwargs: Any) -> Any:  # not used by stream tests
            return SimpleNamespace(content=[], usage=None)

        self.messages = SimpleNamespace(create=_create, stream=stream)


def test_anthropic_backend_real_streaming_yields_per_block(
    sources: list[Source],
) -> None:
    """Each completed block emits a chunk carrying its accumulated cite markers.

    The real SDK only attaches citations at ``content_block_stop``, so per-block
    is the natural granularity for the Citations API.
    """
    block_one = SimpleNamespace(
        type="text",
        text="Alpha claim.",
        citations=[SimpleNamespace(document_index=0)],
    )
    block_two = SimpleNamespace(
        type="text",
        text="Beta claim.",
        citations=[SimpleNamespace(document_index=2)],
    )
    events = [
        SimpleNamespace(type="content_block_stop", content_block=block_one),
        SimpleNamespace(type="content_block_stop", content_block=block_two),
    ]
    final = SimpleNamespace(
        content=[block_one, block_two],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )
    stream = _FakeAnthropicStream(events, final)
    fake = _FakeAnthropicWithStream(stream=stream)
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    chunks = list(backend.stream(prompt="hi", sources=sources, policy=Policy.AUTO))
    text = "".join(chunks)
    assert "[1]" in text
    assert "[3]" in text
    assert len(chunks) == 2
    # ``last_usage`` is populated from ``get_final_message()``.
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 100


def test_anthropic_backend_streaming_ignores_non_content_block_stop_events(
    sources: list[Source],
) -> None:
    """``message_start`` / ``content_block_delta`` / ``message_stop`` events
    should pass through silently — only completed blocks matter."""
    block = SimpleNamespace(
        type="text",
        text="Only block.",
        citations=[SimpleNamespace(document_index=0)],
    )
    events = [
        SimpleNamespace(type="message_start"),
        SimpleNamespace(type="content_block_start"),
        SimpleNamespace(type="content_block_delta"),
        SimpleNamespace(type="content_block_stop", content_block=block),
        SimpleNamespace(type="message_stop"),
    ]
    final = SimpleNamespace(content=[block], usage=None)
    stream = _FakeAnthropicStream(events, final)
    fake = _FakeAnthropicWithStream(stream=stream)
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    chunks = list(backend.stream(prompt="hi", sources=sources, policy=Policy.AUTO))
    assert len(chunks) == 1
    assert "[1]" in "".join(chunks)


def test_anthropic_backend_streaming_falls_back_when_sdk_lacks_stream(
    sources: list[Source],
) -> None:
    """An older client (or a fake that only mocks ``create``) should fall
    back to the non-streaming path rather than crashing."""
    blocks = [
        SimpleNamespace(
            type="text",
            text="Claim.",
            citations=[SimpleNamespace(document_index=0)],
        ),
    ]
    fake = _FakeAnthropic(blocks)  # only mocks `create`, no `stream`
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake)
    chunks = list(backend.stream(prompt="hi", sources=sources, policy=Policy.AUTO))
    assert any("[1]" in c for c in chunks)
