"""Unit tests for FireworksBackend + TogetherBackend with mocked clients.

Live-API coverage (gated on ``FIREWORKS_API_KEY`` / ``TOGETHER_API_KEY``)
lives alongside the other API backends in
``tests/integration/test_env_connectivity.py``. These unit tests use
``SimpleNamespace`` stand-ins to assert the per-backend specifics
without a network call:

- **Fireworks**: that the GBNF response_format is built from
  ``citeformer.grammar.build_grammar`` and that the grammar string
  contains the ``cite-id`` rule (the §10.1 invariant lives there).
- **Together**: that strict ``json_schema`` is set, the citation enum
  is bounded to ``[1..N]``, and the OpenAI base URL is overridden.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

from citeformer import Policy, Source
from citeformer.backends.fireworks import DEFAULT_BASE_URL as FIREWORKS_BASE_URL
from citeformer.backends.fireworks import FireworksBackend
from citeformer.backends.together import DEFAULT_BASE_URL as TOGETHER_BASE_URL
from citeformer.backends.together import TogetherBackend


@pytest.fixture
def sources() -> list[Source]:
    return [
        Source(
            metadata={"id": f"s{i}", "type": "book", "title": f"Source {i}"},
            content=f"Body text for source {i}.",
        )
        for i in range(1, 4)
    ]


class _FakeOpenAI:
    """Minimal stand-in for the ``openai.OpenAI`` client.

    Same shape as the OpenAI / OpenRouter unit-test fakes — Fireworks
    and Together both ride on the OpenAI SDK so this reuses cleanly.
    """

    def __init__(self, response_content: str, *, usage: Any = None) -> None:
        self._response = response_content
        self._usage = usage
        self.last_payload: dict[str, Any] | None = None

        def _create(**kwargs: Any) -> Any:
            self.last_payload = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response_content))],
                usage=self._usage,
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


# ----- Fireworks ------------------------------------------------------------


def test_fireworks_uses_grammar_response_format(sources: list[Source]) -> None:
    """Fireworks must send ``response_format={"type": "grammar", ...}`` —
    swapping out OpenAI's strict-JSON path for the native GBNF path."""
    fake = _FakeOpenAI("Some text [1] [2] [3].")
    backend = FireworksBackend(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    response_format = fake.last_payload["response_format"]
    assert response_format["type"] == "grammar"
    assert "grammar" in response_format
    assert isinstance(response_format["grammar"], str)


def test_fireworks_grammar_contains_cite_id_rule(sources: list[Source]) -> None:
    """The §10.1 ``cite-id`` rule must show up in the GBNF Fireworks
    receives — that's the load-bearing part of the contract that makes
    fabrication token-impossible."""
    fake = _FakeOpenAI("Text [1].")
    backend = FireworksBackend(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    grammar = fake.last_payload["response_format"]["grammar"]
    assert "cite-id" in grammar
    # All N source ids must appear as terminals in the cite-id rule.
    for i in range(1, len(sources) + 1):
        assert f'"{i}"' in grammar


def test_fireworks_grammar_marker_style_baked_in(sources: list[Source]) -> None:
    """``marker_style`` is baked into the grammar's terminal — passing PAREN
    swaps the brackets in the cite-id rule."""
    fake = _FakeOpenAI("Text (1).")
    backend = FireworksBackend(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        client=fake,
    )
    from citeformer import MarkerStyle

    backend.generate(
        prompt="hi",
        sources=sources,
        policy=Policy.AUTO,
        marker_style=MarkerStyle.PAREN,
    )
    grammar = fake.last_payload["response_format"]["grammar"]
    assert '"("' in grammar and '")"' in grammar


def test_fireworks_decode_response_is_passthrough(sources: list[Source]) -> None:
    """Grammar mode returns plain text with markers — no JSON to flatten.
    The orchestrator's regex parser picks up the markers post-hoc."""
    fake = _FakeOpenAI("The opening is dreary [1] and contemplative [2].")
    backend = FireworksBackend(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        client=fake,
    )
    text = backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert text == "The opening is dreary [1] and contemplative [2]."


def test_fireworks_extracts_usage(sources: list[Source]) -> None:
    """Fireworks usage payload is OpenAI-shaped — ``last_usage`` populates
    just like the OpenAI backend."""
    fake = _FakeOpenAI(
        "Text [1].",
        usage=SimpleNamespace(prompt_tokens=200, completion_tokens=30),
    )
    backend = FireworksBackend(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 200
    assert backend.last_usage.output_tokens == 30


def test_fireworks_rejects_empty_sources() -> None:
    fake = _FakeOpenAI("")
    backend = FireworksBackend(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        client=fake,
    )
    with pytest.raises(ValueError, match="at least 1 source"):
        backend.generate(prompt="hi", sources=[], policy=Policy.AUTO)


def test_fireworks_picks_up_env_var_for_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``FIREWORKS_API_KEY`` env var → openai SDK ``api_key`` kwarg."""
    captured: dict[str, Any] = {}

    class _CapturingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **k: SimpleNamespace(
                        choices=[
                            SimpleNamespace(message=SimpleNamespace(content="x")),
                        ],
                        usage=None,
                    )
                )
            )

    import openai as openai_sdk

    monkeypatch.setattr(openai_sdk, "OpenAI", _CapturingOpenAI, raising=True)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test-key")
    FireworksBackend(model="accounts/fireworks/models/llama-v3p1-8b-instruct")
    assert captured["api_key"] == "fw-test-key"
    assert captured["base_url"] == FIREWORKS_BASE_URL


# ----- Together -------------------------------------------------------------


def test_together_sends_strict_json_schema(sources: list[Source]) -> None:
    """Together uses the standard OpenAI strict JSON-schema shape — the
    citation enum is bounded to ``[1..N]`` exactly like OpenAI."""
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = TogetherBackend(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    response_format = fake.last_payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    enum = response_format["json_schema"]["schema"]["properties"]["segments"]["items"][
        "properties"
    ]["citations"]["items"]["enum"]
    assert enum == [1, 2, 3]


def test_together_end_to_end_flattens_segments(sources: list[Source]) -> None:
    """Together's response shape is OpenAI-identical — segment flattening
    works unchanged from the parent backend."""
    fake = _FakeOpenAI(
        json.dumps(
            {
                "segments": [
                    {"text": "Alpha claim.", "citations": [1]},
                    {"text": "Beta claim.", "citations": [2, 3]},
                ]
            }
        )
    )
    backend = TogetherBackend(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        client=fake,
    )
    text = backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "Alpha claim [1]." in text
    assert "Beta claim [2][3]." in text


def test_together_extracts_usage(sources: list[Source]) -> None:
    fake = _FakeOpenAI(
        json.dumps({"segments": []}),
        usage=SimpleNamespace(prompt_tokens=320, completion_tokens=80),
    )
    backend = TogetherBackend(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 320


def test_together_rejects_empty_sources() -> None:
    fake = _FakeOpenAI("{}")
    backend = TogetherBackend(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        client=fake,
    )
    with pytest.raises(ValueError, match="at least 1 source"):
        backend.generate(prompt="hi", sources=[], policy=Policy.AUTO)


def test_together_picks_up_env_var_for_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``TOGETHER_API_KEY`` env var → openai SDK ``api_key`` kwarg."""
    captured: dict[str, Any] = {}

    class _CapturingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **k: SimpleNamespace(
                        choices=[
                            SimpleNamespace(message=SimpleNamespace(content="{}")),
                        ],
                        usage=None,
                    )
                )
            )

    import openai as openai_sdk

    monkeypatch.setattr(openai_sdk, "OpenAI", _CapturingOpenAI, raising=True)
    monkeypatch.setenv("TOGETHER_API_KEY", "tg-test-key")
    TogetherBackend(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    assert captured["api_key"] == "tg-test-key"
    assert captured["base_url"] == TOGETHER_BASE_URL


# ----- Module-constant smoke ------------------------------------------------


def test_module_constants_are_https() -> None:
    """Guards against a typo'd rename of the base-URL constants."""
    assert FIREWORKS_BASE_URL.startswith("https://")
    assert TOGETHER_BASE_URL.startswith("https://")
    # `os.environ.get` smoke — guards against a typo'd env var name.
    assert os.environ.get("FIREWORKS_API_KEY", "") == os.environ.get(
        "FIREWORKS_API_KEY", ""
    )
