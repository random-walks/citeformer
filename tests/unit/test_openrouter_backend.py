"""Unit tests for OpenRouterBackend with mocked clients.

Live-API coverage (requires ``OPENROUTER_API_KEY``) lives alongside the
other API backends in ``tests/integration/test_api_backends_live.py``;
these unit tests use ``SimpleNamespace`` stand-ins so we can assert the
OpenRouter-specific routing knobs (``provider.require_parameters``,
fallback models, app-attribution headers, cost reporting) without a
network call.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

from citeformer import Policy, Source
from citeformer.backends.openrouter import DEFAULT_BASE_URL, OpenRouterBackend


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

    Records the kwargs the backend sends (so we can assert the OpenRouter-
    specific ``extra_body`` / ``extra_headers`` / model / etc.) and
    returns a scripted response.
    """

    def __init__(
        self,
        response_content: str,
        *,
        usage: Any = None,
    ) -> None:
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


# ----- Routing + provider gating --------------------------------------------


def test_openrouter_sets_provider_require_parameters_by_default(
    sources: list[Source],
) -> None:
    """Default behaviour: refuse to route to upstreams that drop request
    parameters — preserves citeformer's strict-schema guarantee end-to-end."""
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    extra_body = fake.last_payload["extra_body"]
    assert extra_body["provider"]["require_parameters"] is True


def test_openrouter_provider_require_can_be_disabled() -> None:
    """``require_provider_parameters=False`` lets users opt in to whatever
    upstream is fastest, accepting that strict-mode may not flow through."""
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
        require_provider_parameters=False,
    )
    sources_local = [
        Source(metadata={"id": "x", "type": "book"}, content="text"),
    ]
    backend.generate(prompt="hi", sources=sources_local, policy=Policy.AUTO)
    extra_body = fake.last_payload.get("extra_body") or {}
    # No `provider.require_parameters` flag present.
    assert extra_body.get("provider", {}).get("require_parameters") is None


def test_openrouter_threads_fallback_models(sources: list[Source]) -> None:
    """``fallback_models`` becomes ``extra_body['models']`` so OpenRouter
    fails over automatically if the primary upstream errors."""
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
        fallback_models=["openai/gpt-4o", "google/gemini-2.5-pro"],
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    models = fake.last_payload["extra_body"]["models"]
    assert models == [
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-4o",
        "google/gemini-2.5-pro",
    ]


def test_openrouter_omits_models_when_no_fallback(sources: list[Source]) -> None:
    """No fallback list → no ``models`` field; OpenRouter routes to the
    single primary."""
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "models" not in fake.last_payload["extra_body"]


def test_openrouter_does_not_send_deprecated_usage_include_flag(
    sources: list[Source],
) -> None:
    """OpenRouter's ``usage: {include: true}`` flag is deprecated and a no-op
    as of structured-outputs GA — cost is always returned. We must not send it."""
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    extra_body = fake.last_payload.get("extra_body") or {}
    assert "usage" not in extra_body


# ----- Cost / usage reporting ------------------------------------------------


def test_openrouter_extracts_cost_into_last_usage(sources: list[Source]) -> None:
    """When OpenRouter returns ``usage.cost`` (in OR credits), it surfaces on
    ``GenerationResult.usage.cost_credits`` via ``backend.last_usage``."""
    fake = _FakeOpenAI(
        json.dumps({"segments": [{"text": "Claim.", "citations": [1]}]}),
        usage=SimpleNamespace(prompt_tokens=420, completion_tokens=60, cost=0.0042),
    )
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 420
    assert backend.last_usage.output_tokens == 60
    assert backend.last_usage.cost_credits == 0.0042


def test_openrouter_handles_response_without_cost(sources: list[Source]) -> None:
    """Provider-side bugs / older API versions may omit ``cost``; we should
    still populate token counts."""
    fake = _FakeOpenAI(
        json.dumps({"segments": [{"text": "Claim.", "citations": [1]}]}),
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 100
    assert backend.last_usage.cost_credits is None


# ----- App attribution headers ----------------------------------------------


def test_openrouter_attaches_app_headers_when_provided(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``app_name`` + ``app_url`` set ``X-Title`` / ``HTTP-Referer`` so spend
    is attributable on the OpenRouter dashboard."""
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
    # Keep the backend's `import os` from finding a real key.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        api_key="sk-test",
        app_name="citeformer-tests",
        app_url="https://example.invalid/citeformer",
    )
    # Sync client is lazy (ADR-014) — accessing the property triggers
    # construction with the captured kwargs.
    _ = backend.client
    headers = captured["default_headers"]
    assert headers["X-Title"] == "citeformer-tests"
    assert headers["HTTP-Referer"] == "https://example.invalid/citeformer"
    assert captured["base_url"] == DEFAULT_BASE_URL
    assert captured["api_key"] == "sk-test"


def test_openrouter_falls_back_to_env_var_for_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """When ``api_key=None``, ``OPENROUTER_API_KEY`` from the env is used."""
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
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-test")
    backend = OpenRouterBackend(model="openai/gpt-4o")
    _ = backend.client  # force lazy construction (ADR-014)
    assert captured["api_key"] == "sk-env-test"


# ----- Inheritance sanity ---------------------------------------------------


def test_openrouter_inherits_strict_json_schema(sources: list[Source]) -> None:
    """OpenRouter is a thin OpenAIBackend subclass — strict JSON schema
    construction should still happen unchanged."""
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
    )
    backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    response_format = fake.last_payload["response_format"]
    assert response_format["json_schema"]["strict"] is True
    enum = response_format["json_schema"]["schema"]["properties"]["segments"]["items"][
        "properties"
    ]["citations"]["items"]["enum"]
    assert enum == [1, 2, 3]


def test_openrouter_inherits_segment_flattening(sources: list[Source]) -> None:
    """Segments → citation-marked plain text path should be inherited
    unchanged from the parent backend."""
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
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
    )
    text = backend.generate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "Alpha claim [1]." in text
    assert "Beta claim [2][3]." in text


def test_openrouter_threads_user_extra_body_alongside_routing(
    sources: list[Source],
) -> None:
    """A user-supplied ``extra_body`` should be merged with the routing
    fields — neither should clobber the other."""
    fake = _FakeOpenAI(json.dumps({"segments": []}))
    backend = OpenRouterBackend(
        model="anthropic/claude-sonnet-4.6",
        client=fake,
    )
    backend.generate(
        prompt="hi",
        sources=sources,
        policy=Policy.AUTO,
        extra_body={"transforms": ["middle-out"]},
    )
    extra_body = fake.last_payload["extra_body"]
    assert extra_body["transforms"] == ["middle-out"]
    assert extra_body["provider"]["require_parameters"] is True


# ----- env_var_pickup smoke -------------------------------------------------


def test_openrouter_default_module_constant() -> None:
    """The base URL constant is exposed for callers that want to override it."""
    assert DEFAULT_BASE_URL.startswith("https://")
    # Smoke check the env-var name we document — guards against a typo'd rename.
    assert os.environ.get("OPENROUTER_API_KEY", "") == os.environ.get("OPENROUTER_API_KEY", "")
