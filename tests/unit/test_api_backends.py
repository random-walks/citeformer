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
    assert (
        schema["properties"]["segments"]["items"]["additionalProperties"] is False
    )


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
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=response_content))
                ]
            )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=_create)
        )


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
    text = backend.generate(
        prompt="Describe the sources.", sources=sources, policy=Policy.AUTO
    )
    assert "[1]" in text
    assert "[2]" in text
    # Grammar-equivalent enforcement: schema's enum matches the sources.
    payload = fake.last_payload
    assert payload is not None
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["segments"]["items"]["properties"]["citations"][
        "items"
    ]["enum"] == [1, 2, 3]


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
    chunks = list(
        backend.stream(prompt="hi", sources=sources, policy=Policy.AUTO)
    )
    assert len(chunks) >= 2


# ----- Anthropic adapter -----------------------------------------------------


class _FakeAnthropic:
    """Stand-in for the ``anthropic.Anthropic`` client."""

    def __init__(self, content_blocks: list[Any]) -> None:
        self._blocks = content_blocks
        self.last_kwargs: dict[str, Any] | None = None

        def _create(**kwargs: Any) -> Any:
            self.last_kwargs = kwargs
            return SimpleNamespace(content=self._blocks)

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
    text = backend.generate(
        prompt="Describe.", sources=sources, policy=Policy.AUTO
    )
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
