"""OpenRouter backend — multi-provider routing on the OpenAI wire format.

OpenRouter (https://openrouter.ai) is a thin proxy in front of every
production LLM provider — OpenAI, Anthropic, Google, Mistral, Meta,
DeepSeek, and so on. The wire format is identical to OpenAI's chat
completions API, so this backend is implemented as a thin
:class:`OpenAIBackend` subclass: schema construction, segment flattening,
streaming, and usage extraction are all inherited unchanged. We only add
the OpenRouter-specific knobs:

- **Per-model routing.** Models are namespaced by provider, e.g.
  ``"anthropic/claude-sonnet-4.6"``, ``"openai/gpt-4o"``,
  ``"google/gemini-2.5-pro"``. The default is configurable via
  ``model=...`` at construction time; see
  https://openrouter.ai/models for the live catalogue.
- **Provider gating.** OpenRouter's ``provider.require_parameters: true``
  knob refuses to route to any upstream that doesn't accept every
  parameter you sent. Combined with the strict ``response_format``
  citeformer always sets, this means: if the upstream provider can't do
  logit-tier strict schema enforcement, OpenRouter returns a 4xx instead
  of silently falling back to a non-strict provider. That preserves the
  citeformer guarantee end-to-end. Disable with ``require_provider_parameters=False``
  if you'd rather take whatever upstream answers first.
- **App attribution.** OpenRouter encourages clients to identify
  themselves with ``HTTP-Referer`` + ``X-Title`` headers. Pass
  ``app_name`` and/or ``app_url`` at construction time and they're
  threaded onto every request.
- **Cost reporting.** When ``include_cost=True`` (default), OpenRouter
  attaches a per-call ``cost`` field to the ``usage`` payload — actual
  dollars charged for that completion. The ``OpenAIBackend`` usage
  extractor already understands this field and surfaces it on
  :attr:`GenerationResult.usage.cost_usd`.
- **Provider fallback.** Pass ``fallback_models=[...]`` to enable
  OpenRouter's automatic failover when the primary upstream is down or
  rate-limited. Citeformer's strict ``response_format`` requirement
  flows to fallback providers too.

Tier honesty: when ``require_provider_parameters=True`` (the default),
this backend is **logit-tier on every modern upstream** that supports
strict structured outputs (OpenAI ``gpt-4o-2024-08-06+``, Anthropic
Sonnet 4.5+, Gemini 2.x, Mistral 2024-11+, Groq, Fireworks, Together).
With ``require_provider_parameters=False`` it falls back to schema-tier
post-validation on whichever provider answers — fabrication is still
structurally impossible in the returned payload, but enforcement isn't
guaranteed at sample time anymore.

Requires the ``openrouter`` extra: ``pip install citeformer[openrouter]``
(re-uses the ``openai`` SDK, since OpenRouter is OpenAI wire-compatible).
"""

from __future__ import annotations

import logging
from typing import Any

from citeformer.backends.openai import OpenAIBackend

_LOG = logging.getLogger(__name__)

#: OpenRouter's OpenAI-compatible base URL.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: A reasonable default — Anthropic's flagship Sonnet routed via OpenRouter.
#: Picked because it supports strict structured outputs (logit-tier
#: enforcement on the upstream) and is broadly available across regions.
_DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"


class OpenRouterBackend(OpenAIBackend):
    """OpenRouter multi-provider backend with per-model routing.

    Inherits OpenAI's schema construction, segment flattening, streaming
    fallback, and token-usage extraction — only the request-augmentation
    hook and a handful of init knobs differ. See the module docstring for
    the OpenRouter-specific knobs (provider gating, app attribution,
    cost reporting, fallback models).

    Attributes:
        model: OpenRouter model id including the provider prefix
            (``"anthropic/claude-sonnet-4.6"``, ``"openai/gpt-4o"``, …).
        client: The underlying ``openai.OpenAI`` client, configured with
            OpenRouter's base URL + the user's app-attribution headers.
        require_provider_parameters: If ``True`` (default), only routes
            to upstreams that accept every request parameter — preserves
            the strict-schema enforcement guarantee end-to-end.
        fallback_models: Ordered list of secondary models OpenRouter may
            failover to. ``None`` disables fallback.
        include_cost: If ``True`` (default), asks OpenRouter to attach a
            per-call USD cost to the usage payload.
    """

    require_provider_parameters: bool
    fallback_models: list[str] | None
    include_cost: bool

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        app_name: str | None = None,
        app_url: str | None = None,
        require_provider_parameters: bool = True,
        fallback_models: list[str] | None = None,
        include_cost: bool = True,
        **client_kwargs: Any,
    ) -> None:
        """Construct an OpenRouter backend.

        Args:
            model: OpenRouter model id (``"<provider>/<model>"``). See
                https://openrouter.ai/models for the live catalogue.
            client: Pre-built ``openai.OpenAI`` client (already pointed at
                OpenRouter). When ``None``, one is constructed using the
                other arguments.
            api_key: OpenRouter API key. When ``None``, falls back to
                ``OPENROUTER_API_KEY`` from the environment, then to
                ``client_kwargs["api_key"]`` if supplied.
            base_url: OpenRouter API base URL. Override only for
                staging / proxy testing.
            app_name: Sets the ``X-Title`` header — shown on the
                OpenRouter dashboard so you can attribute spend to your
                app. Optional but encouraged.
            app_url: Sets the ``HTTP-Referer`` header — same purpose.
                Optional.
            require_provider_parameters: When ``True`` (default), refuses
                to route to upstreams that drop request parameters.
                Preserves citeformer's strict-schema enforcement.
            fallback_models: Optional ordered list of secondary models for
                automatic failover if the primary upstream errors or rate
                limits.
            include_cost: When ``True`` (default), asks OpenRouter to
                attach the per-call USD cost to ``usage.cost`` so it
                surfaces on :attr:`GenerationResult.usage.cost_usd`.
            **client_kwargs: Forwarded to ``openai.OpenAI(**kwargs)`` when
                ``client`` is ``None`` (``timeout``, ``max_retries``, …).
        """
        if client is None:
            import os

            resolved_api_key = (
                api_key
                or os.environ.get("OPENROUTER_API_KEY")
                or client_kwargs.pop("api_key", None)
            )
            default_headers: dict[str, str] = {}
            if app_url:
                default_headers["HTTP-Referer"] = app_url
            if app_name:
                default_headers["X-Title"] = app_name
            existing_headers = client_kwargs.pop("default_headers", None) or {}
            merged_headers = {**default_headers, **existing_headers}
            client_kwargs.setdefault("base_url", base_url)
            if resolved_api_key is not None:
                client_kwargs["api_key"] = resolved_api_key
            if merged_headers:
                client_kwargs["default_headers"] = merged_headers

        super().__init__(model=model, client=client, **client_kwargs)
        self.require_provider_parameters = require_provider_parameters
        self.fallback_models = list(fallback_models) if fallback_models else None
        self.include_cost = include_cost

    def _augment_create_kwargs(
        self,
        kwargs: dict[str, Any],
        *,
        options: dict[str, Any],
    ) -> None:
        """Attach OpenRouter-specific routing fields to the completion call.

        Threads ``provider``, ``models`` (fallback list), and ``usage`` flags
        through ``extra_body`` — the OpenAI SDK's escape hatch for
        non-standard request fields. The base ``OpenAIBackend.generate()``
        calls this hook just before posting the request.
        """
        extra_body: dict[str, Any] = dict(options.get("extra_body") or {})

        provider_block: dict[str, Any] = dict(extra_body.get("provider") or {})
        if self.require_provider_parameters and "require_parameters" not in provider_block:
            provider_block["require_parameters"] = True
        if provider_block:
            extra_body["provider"] = provider_block

        if self.fallback_models and "models" not in extra_body:
            extra_body["models"] = [self.model, *self.fallback_models]

        if self.include_cost and "usage" not in extra_body:
            extra_body["usage"] = {"include": True}

        if extra_body:
            kwargs["extra_body"] = extra_body
