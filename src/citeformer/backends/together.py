"""Together AI backend — provider-runtime enforcement on the OpenAI wire format.

Together AI's chat-completions endpoint is OpenAI-API-compatible and
exposes a ``response_format={"type": "json_schema", "json_schema": {...}}``
mode (`docs <https://docs.together.ai/docs/json-mode>`_) that runs
constrained decoding inside the Together runtime — the same mechanism
OpenAI / Mistral use, just on Together-hosted open-weight models. They
also support a ``response_format={"type": "regex", "pattern": "..."}``
mode which would be a natural fit for marker-only output, but the
json_schema path is what every existing API backend uses; reusing it
keeps the conformance contract uniform.

Tier honesty: this is **provider-runtime constrained sampling** — a
fabricated cite id is token-impossible to sample inside the Together
runtime. Same guarantee as OpenAI's strict mode, just with open-weight
upstream models (Llama, Qwen, DeepSeek, …) instead of closed ones.

Implementation note: like OpenRouter, Together is a thin
:class:`OpenAIBackend` subclass — schema construction, segment
flattening, streaming pseudo-stream, and ``last_usage`` extraction all
inherited unchanged. We only override ``__init__`` to point the SDK at
Together's base URL and pick up ``TOGETHER_API_KEY``.

Requires the ``together`` extra: ``pip install citeformer[together]``
(re-uses the ``openai`` SDK; no Together-specific client needed).
"""

from __future__ import annotations

import logging
from typing import Any

from citeformer.backends.openai import OpenAIBackend

_LOG = logging.getLogger(__name__)

#: Together's OpenAI-compatible base URL.
DEFAULT_BASE_URL = "https://api.together.xyz/v1"

#: Cheap default — small Llama 3.1 model that supports json_schema.
_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


class TogetherBackend(OpenAIBackend):
    """Together AI backend with strict json_schema cite enforcement.

    Attributes:
        model: Together model id (``meta-llama/...``, ``Qwen/...``, …).
        client: The underlying ``openai.OpenAI`` client, configured with
            Together's base URL.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        **client_kwargs: Any,
    ) -> None:
        """Construct a Together backend.

        Args:
            model: Together model id. See https://api.together.xyz/models
                for the live catalogue. Default is
                ``Meta-Llama-3.1-8B-Instruct-Turbo`` — small + cheap +
                supports json_schema constrained decoding.
            client: Pre-built ``openai.OpenAI`` client (already pointed at
                Together). When ``None``, one is constructed using the
                other arguments.
            api_key: Together API key. When ``None``, falls back to
                ``TOGETHER_API_KEY`` from the environment, then to
                ``client_kwargs["api_key"]`` if supplied.
            base_url: Together API base URL. Override only for staging /
                proxy testing.
            **client_kwargs: Forwarded to ``openai.OpenAI(**kwargs)`` when
                ``client`` is ``None`` (``timeout``, ``max_retries``, …).
        """
        if client is None:
            import os

            resolved_api_key = (
                api_key or os.environ.get("TOGETHER_API_KEY") or client_kwargs.pop("api_key", None)
            )
            client_kwargs.setdefault("base_url", base_url)
            if resolved_api_key is not None:
                client_kwargs["api_key"] = resolved_api_key

        super().__init__(model=model, client=client, **client_kwargs)
