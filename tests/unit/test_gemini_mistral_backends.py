"""Unit tests for GeminiBackend + MistralBackend with mocked clients.

Live-API coverage (requires ``GEMINI_API_KEY`` / ``MISTRAL_API_KEY``)
lives in ``tests/integration/test_api_backends_live.py`` as
``@pytest.mark.integration``. These unit tests use ``SimpleNamespace``
stand-ins so we can assert schema + message-construction invariants
without network calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from citeformer import MarkerStyle, Policy, Source
from citeformer.backends.gemini import GeminiBackend, _build_citation_schema
from citeformer.backends.mistral import MistralBackend

# --- Shared fixtures ---------------------------------------------------------


@pytest.fixture
def sources() -> list[Source]:
    return [
        Source(
            metadata={"id": f"s{i}", "type": "book", "title": f"Source {i}"},
            content=f"Body text for source {i}.",
        )
        for i in range(1, 4)
    ]


# --- Gemini schema construction ---------------------------------------------


def test_gemini_schema_enum_bounded_by_n_sources() -> None:
    schema = _build_citation_schema(n_sources=4, policy=Policy.AUTO)
    items = schema["properties"]["segments"]["items"]
    assert items["properties"]["citations"]["items"]["enum"] == [1, 2, 3, 4]


def test_gemini_schema_uses_snake_case_min_items() -> None:
    """Gemini's schema dialect expects ``min_items`` (not camelCase)."""
    schema = _build_citation_schema(n_sources=3, policy=Policy.REQUIRED)
    citations = schema["properties"]["segments"]["items"]["properties"]["citations"]
    assert citations["min_items"] == 1


def test_gemini_schema_omits_additional_properties() -> None:
    """We deliberately DON'T set additionalProperties:false — Gemini's
    validator 400s on some variants. The enum + required fields are the
    constraint."""
    schema = _build_citation_schema(n_sources=2, policy=Policy.AUTO)
    assert "additionalProperties" not in schema
    assert "additionalProperties" not in schema["properties"]["segments"]["items"]


# --- Gemini end-to-end -------------------------------------------------------


class _FakeGemini:
    """Minimal stand-in for the ``google.genai.Client``."""

    def __init__(self, response_text: str) -> None:
        self.last_kwargs: dict[str, Any] | None = None

        def _generate(**kwargs: Any) -> Any:
            self.last_kwargs = kwargs
            return SimpleNamespace(text=response_text)

        self.models = SimpleNamespace(generate_content=_generate)


def test_gemini_backend_end_to_end_maps_segments_to_markers(
    sources: list[Source],
) -> None:
    fake_payload = json.dumps(
        {
            "segments": [
                {"text": "Alpha claim.", "citations": [1]},
                {"text": "Beta claim.", "citations": [3]},
            ]
        }
    )
    fake = _FakeGemini(fake_payload)
    backend = GeminiBackend(model="gemini-2.0-flash", client=fake)
    text = backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "[1]" in text
    assert "[3]" in text


def test_gemini_backend_threads_schema_into_config(sources: list[Source]) -> None:
    fake = _FakeGemini(json.dumps({"segments": []}))
    backend = GeminiBackend(model="gemini-2.0-flash", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.REQUIRED)
    kwargs = fake.last_kwargs
    assert kwargs is not None
    schema = kwargs["config"]["response_schema"]
    assert schema["properties"]["segments"]["items"]["properties"]["citations"]["items"][
        "enum"
    ] == [1, 2, 3]
    assert kwargs["config"]["response_mime_type"] == "application/json"


def test_gemini_backend_embeds_sources_in_system_instruction(
    sources: list[Source],
) -> None:
    fake = _FakeGemini(json.dumps({"segments": []}))
    backend = GeminiBackend(model="gemini-2.0-flash", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    system = fake.last_kwargs["config"]["system_instruction"]
    for i in range(1, len(sources) + 1):
        assert f"[{i}]" in system


def test_gemini_backend_rejects_empty_sources() -> None:
    fake = _FakeGemini("{}")
    backend = GeminiBackend(model="gemini-2.0-flash", client=fake)
    with pytest.raises(ValueError, match="at least 1 source"):
        backend.generate(prompt="hi", sources=[], policy=Policy.AUTO)


@pytest.mark.parametrize(
    "marker_style,expected_marker",
    [
        (MarkerStyle.BRACKET, "[1]"),
        (MarkerStyle.PAREN, "(1)"),
        (MarkerStyle.CURLY, "{1}"),
        (MarkerStyle.CARET, "^1"),
    ],
)
def test_gemini_backend_honours_marker_style(
    sources: list[Source], marker_style: MarkerStyle, expected_marker: str
) -> None:
    fake_payload = json.dumps({"segments": [{"text": "Claim.", "citations": [1]}]})
    backend = GeminiBackend(model="gemini-2.0-flash", client=_FakeGemini(fake_payload))
    text = backend.generate(
        prompt="hi", sources=sources, policy=Policy.AUTO, marker_style=marker_style
    )
    assert expected_marker in text


def test_gemini_stream_yields_multiple_chunks(sources: list[Source]) -> None:
    payload = json.dumps(
        {
            "segments": [
                {"text": "One.", "citations": [1]},
                {"text": "Two.", "citations": [2]},
            ]
        }
    )
    backend = GeminiBackend(model="gemini-2.0-flash", client=_FakeGemini(payload))
    chunks = list(backend.stream(prompt="hi", sources=sources, policy=Policy.AUTO))
    assert len(chunks) >= 2


# --- Mistral end-to-end ------------------------------------------------------


class _FakeMistral:
    """Stand-in for the ``mistralai.Mistral`` client."""

    def __init__(self, response_content: str) -> None:
        self.last_kwargs: dict[str, Any] | None = None

        def _complete(**kwargs: Any) -> Any:
            self.last_kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response_content))]
            )

        self.chat = SimpleNamespace(complete=_complete)


def test_mistral_backend_end_to_end_maps_segments_to_markers(
    sources: list[Source],
) -> None:
    payload = json.dumps(
        {
            "segments": [
                {"text": "Alpha claim.", "citations": [1]},
                {"text": "Beta claim.", "citations": [2, 3]},
            ]
        }
    )
    backend = MistralBackend(model="mistral-large-latest", client=_FakeMistral(payload))
    text = backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "[1]" in text
    assert "[2][3]" in text


def test_mistral_backend_sends_strict_true(sources: list[Source]) -> None:
    fake = _FakeMistral(json.dumps({"segments": []}))
    backend = MistralBackend(model="mistral-large-latest", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert fake.last_kwargs["response_format"]["json_schema"]["strict"] is True


def test_mistral_backend_schema_enum_bounds_cite_ids(sources: list[Source]) -> None:
    fake = _FakeMistral(json.dumps({"segments": []}))
    backend = MistralBackend(model="mistral-large-latest", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.REQUIRED)
    schema = fake.last_kwargs["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["segments"]["items"]["properties"]["citations"]["items"][
        "enum"
    ] == [1, 2, 3]


def test_mistral_backend_embeds_sources_in_system_prompt(sources: list[Source]) -> None:
    fake = _FakeMistral(json.dumps({"segments": []}))
    backend = MistralBackend(model="mistral-large-latest", client=fake)
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    system = fake.last_kwargs["messages"][0]["content"]
    for i in range(1, len(sources) + 1):
        assert f"[{i}]" in system


def test_mistral_backend_rejects_empty_sources() -> None:
    fake = _FakeMistral("{}")
    backend = MistralBackend(model="mistral-large-latest", client=fake)
    with pytest.raises(ValueError, match="at least 1 source"):
        backend.generate(prompt="hi", sources=[], policy=Policy.AUTO)


def test_mistral_stream_yields_multiple_chunks(sources: list[Source]) -> None:
    payload = json.dumps(
        {
            "segments": [
                {"text": "One.", "citations": [1]},
                {"text": "Two.", "citations": [2]},
            ]
        }
    )
    backend = MistralBackend(model="mistral-large-latest", client=_FakeMistral(payload))
    chunks = list(backend.stream(prompt="hi", sources=sources, policy=Policy.AUTO))
    assert len(chunks) >= 2
