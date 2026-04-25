"""Fireworks AI backend — true logit-tier enforcement on a hosted API via GBNF.

Fireworks AI exposes a native ``response_format={"type": "grammar",
"grammar": "<GBNF>"}`` mode (`docs <https://docs.fireworks.ai/structured-responses/structured-output-grammar-based>`_)
that compiles the supplied GBNF and uses it to mask logits at every
decode step inside the Fireworks runtime — the same mechanism XGrammar
uses inside the local HF backend, just running on Fireworks's servers
instead of yours. This is the cleanest "true logit-tier on a hosted
API" backend possible: we drop the existing :func:`citeformer.grammar.
build_grammar` output in **unchanged** and the same `cite-id` rule that
constrains local sampling now constrains Fireworks's sampling.

Tier honesty: this is **logit-tier**, not schema-tier. A fabricated
cite id is token-impossible to sample inside the Fireworks runtime —
exactly the same guarantee as ``HFBackend`` / ``VLLMBackend`` /
``LlamaCppBackend``, just with the masking running on a managed
GPU instead of yours.

Implementation note: Fireworks is wire-compatible with OpenAI's chat
completions API, so this backend subclasses :class:`OpenAIBackend` and
overrides only two hooks — :meth:`_build_response_format` (swap
strict-JSON for grammar mode) and :meth:`_decode_response_text` (the
grammar response is plain text with inline markers, not a segments
JSON object, so flattening is a no-op). All the messaging assembly,
streaming pseudo-stream, and ``last_usage`` extraction are inherited
unchanged.

Requires the ``fireworks`` extra: ``pip install citeformer[fireworks]``
(re-uses the ``openai`` SDK; no Fireworks-specific client needed).
"""

from __future__ import annotations

import logging
from typing import Any

from citeformer.backends.openai import OpenAIBackend
from citeformer.core import MarkerStyle, Policy

_LOG = logging.getLogger(__name__)

#: Fireworks' OpenAI-compatible base URL.
DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

#: Cheap, broadly-available model that supports grammar mode.
_DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"


class FireworksBackend(OpenAIBackend):
    """Fireworks AI backend with native GBNF logit-tier enforcement.

    Attributes:
        model: Fireworks model id (``accounts/fireworks/models/<name>``).
        client: The underlying ``openai.OpenAI`` client, configured with
            Fireworks's base URL.
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
        """Construct a Fireworks backend.

        Args:
            model: Fireworks model id (``accounts/fireworks/models/...``).
                Default is ``llama-v3p1-8b-instruct`` — small, serverless,
                supports grammar mode. See https://fireworks.ai/models for
                the live catalogue.
            client: Pre-built ``openai.OpenAI`` client (already pointed at
                Fireworks). When ``None``, one is constructed using the
                other arguments.
            api_key: Fireworks API key. When ``None``, falls back to
                ``FIREWORKS_API_KEY`` from the environment, then to
                ``client_kwargs["api_key"]`` if supplied.
            base_url: Fireworks API base URL. Override only for staging /
                proxy testing.
            **client_kwargs: Forwarded to ``openai.OpenAI(**kwargs)`` when
                ``client`` is ``None`` (``timeout``, ``max_retries``, …).
        """
        if client is None:
            import os

            resolved_api_key = (
                api_key or os.environ.get("FIREWORKS_API_KEY") or client_kwargs.pop("api_key", None)
            )
            client_kwargs.setdefault("base_url", base_url)
            if resolved_api_key is not None:
                client_kwargs["api_key"] = resolved_api_key

        super().__init__(model=model, client=client, **client_kwargs)

    def _build_response_format(
        self,
        *,
        n_sources: int,
        policy: Policy,
        marker_style: MarkerStyle,
    ) -> dict[str, Any]:
        """Build a Fireworks GBNF response_format from citeformer's grammar.

        Calls :func:`citeformer.grammar.build_grammar` with the same args
        the local backends use, then ships the resulting GBNF string as
        Fireworks's ``{"type": "grammar", ...}`` payload. The Fireworks
        runtime compiles it once (cached server-side per grammar string)
        and applies token-level masking at every decode step.
        """
        from citeformer.grammar import build_grammar

        grammar = build_grammar(
            n_sources=n_sources,
            policy=policy,
            marker_style=marker_style,
        )
        return {"type": "grammar", "grammar": grammar.gbnf}

    def _decode_response_text(self, raw: str, *, marker_style: MarkerStyle) -> str:
        """Passthrough — grammar mode emits plain text with markers, no JSON.

        The orchestrator's regex parser picks up the markers in the same
        post-hoc pass it uses for local backends. ``marker_style`` is
        already baked into the grammar at the terminal level via
        ``build_grammar(..., marker_style=...)``, so no per-style
        reformatting is needed here.
        """
        del marker_style  # already baked into the grammar
        return raw or ""
