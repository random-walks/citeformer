"""Mistral backend — schema-level cite-id enforcement via ``response_format``.

Mistral's ``chat.complete`` supports a ``response_format={"type":
"json_schema", "json_schema": {...}}`` parameter that constrains the
assistant to produce a JSON object validating against the supplied
schema. When ``strict: true`` is set, the Mistral server rejects any
response whose citation integers fall outside the supplied enum —
structurally equivalent to what XGrammar does at the logit layer for
local backends.

Tier honesty (same story as OpenAI):

- **Local backends** enforce at the logit layer — fabrication is
  token-impossible to sample.
- **This backend** enforces at the **schema** layer — fabrication is
  structurally impossible in the returned payload.

Requires the ``mistral`` extra: ``pip install citeformer[mistral]``.

Model requirements: the ``strict: true`` JSON-schema mode is supported
on ``mistral-large-2411`` (Nov 2024) and every Mistral model released
after, including ``mistral-small-latest`` and ``mistral-large-latest``.

SDK version: pins ``mistralai>=2.0``. The 2.x line switched to a
namespace-package layout (``from mistralai.client import Mistral``);
1.x used a different entry-point name and isn't supported here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from citeformer.backends.base import Backend
from citeformer.backends.openai import (
    _build_citation_schema as _shared_schema_builder,
)
from citeformer.backends.openai import (
    _extract_openai_usage,
    _flatten_segments,
)
from citeformer.core import MarkerStyle, Policy, Source, TokenUsage

_LOG = logging.getLogger(__name__)

_DEFAULT_MODEL = "mistral-large-latest"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.7


class MistralBackend(Backend):
    """Mistral Chat Completions backend with schema-level cite enforcement.

    Request shape mirrors :class:`OpenAIBackend` — segments + citations
    with enum-bounded integers — so downstream code is identical.

    Attributes:
        model: Mistral model id (``mistral-large-latest`` by default).
        client: The authenticated ``mistralai.Mistral`` client.
        last_usage: Token-usage payload from the most recent ``generate()``
            call. ``None`` before the first call.
    """

    model: str
    client: Any
    last_usage: TokenUsage | None

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Construct a Mistral backend.

        Args:
            model: Mistral model id supporting strict JSON schema
                (``mistral-large-2411`` or later, or ``*-latest`` aliases).
            client: Pre-built ``mistralai.Mistral`` client. If ``None``,
                one is constructed from env (picks up ``MISTRAL_API_KEY``).
            **client_kwargs: Forwarded to ``Mistral(**kwargs)`` when
                ``client`` is ``None``.
        """
        try:
            # mistralai 2.x is a namespace package — the concrete client lives
            # under `mistralai.client`. The `mistral` extra pins `>=2.0`, so
            # this is the only supported shape.
            from mistralai.client import Mistral
        except ImportError as e:
            raise ImportError(
                "MistralBackend requires the `mistral` extra. "
                "Install with `pip install citeformer[mistral]`."
            ) from e

        self.model = model
        self.client = client if client is not None else Mistral(**client_kwargs)
        self.last_usage = None

    def generate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> str:
        """Generate text with schema-level citation constraint.

        Args:
            prompt: User prompt.
            sources: Sources in scope. Position (1-indexed) becomes the
                enum entry.
            policy: Citation policy — shapes the system prompt and the
                schema's ``minItems`` (REQUIRED → 1; AUTO / QUOTES_ONLY → 0).
            **options: ``max_tokens`` (default 1024), ``temperature``
                (default 0.7), ``marker_style`` (default BRACKET),
                ``system_prompt`` (extra system content).

        Returns:
            Flattened text carrying ``marker_style`` markers for every
            cited source, in document order.

        Raises:
            ValueError: If ``sources`` is empty.
        """
        if len(sources) < 1:
            raise ValueError("MistralBackend requires at least 1 source")

        max_tokens = int(options.get("max_tokens", _DEFAULT_MAX_TOKENS))
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))
        marker_style = options.get("marker_style", MarkerStyle.BRACKET)
        schema = _shared_schema_builder(n_sources=len(sources), policy=policy)

        messages = self._build_messages(
            prompt=prompt,
            sources=sources,
            policy=policy,
            system_prompt=options.get("system_prompt"),
        )

        response: Any = self.client.chat.complete(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "CitedSegments",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        self.last_usage = _extract_openai_usage(getattr(response, "usage", None))
        raw = response.choices[0].message.content
        return _flatten_segments(raw, marker_style=marker_style)

    def stream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> Iterator[str]:
        """Yield sentence-level chunks by slicing :meth:`generate`'s output.

        Same rationale as OpenAI / Gemini: Mistral's streaming surface
        emits partial JSON which isn't safe to flatten before the full
        response validates.
        """
        text = self.generate(prompt=prompt, sources=sources, policy=policy, **options)
        buf: list[str] = []
        for char in text:
            buf.append(char)
            if char in ".!?" and buf:
                yield "".join(buf) + " "
                buf = []
        if buf:
            yield "".join(buf)

    @staticmethod
    def _build_messages(
        *,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        system_prompt: str | None,
    ) -> list[dict[str, str]]:
        """Identical in shape to the OpenAI helper. Sources go in the system role.

        Kept independent rather than shared because the two providers
        have diverged on subtle message-format details in the past
        (system role naming, role ordering constraints) and this gives
        each backend room to grow without cross-coupling.
        """
        parts: list[str] = []
        if system_prompt:
            parts.append(system_prompt)
        parts.append(
            "You are answering with citations. You MUST return a JSON object of "
            "shape {'segments': [{'text': '...', 'citations': [int, ...]}]}. "
            "Each citation integer must refer to one of the numbered sources "
            "below by its 1-indexed position."
        )
        if policy is Policy.REQUIRED:
            parts.append("Every segment MUST cite at least one source (required policy).")
        elif policy is Policy.QUOTES_ONLY:
            parts.append(
                "Segments containing direct quotations MUST cite their source; "
                "paraphrased segments may omit citations (quotes_only policy)."
            )
        else:
            parts.append(
                "Citations are optional; only cite when a segment's claim depends "
                "on a specific source (auto policy)."
            )
        parts.append("Sources:")
        for i, src in enumerate(sources, start=1):
            title = src.metadata.get("title", f"Source {i}")
            parts.append(f"[{i}] {title}: {src.content[:400]}")
        return [
            {"role": "system", "content": "\n\n".join(parts)},
            {"role": "user", "content": prompt},
        ]
