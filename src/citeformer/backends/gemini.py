"""Gemini backend — schema-level cite-id enforcement via ``response_schema``.

Gemini's ``generate_content`` accepts a ``response_mime_type="application/json"``
+ ``response_schema=<schema>`` pair; the model is constrained to emit a
JSON object that validates against the schema. The schema subset supported
is OpenAPI-ish — not full JSON Schema — but ``type``, ``enum``, ``items``,
``properties``, and ``required`` are all honoured, which is everything we
need to express ``citations[*] ∈ {1..N}``.

Tier honesty (same story as OpenAI):

- **Local backends** enforce at the logit layer — a fabricated cite id
  is token-impossible to sample.
- **This backend** enforces at the **schema** layer. Gemini validates
  the assistant's response against the schema server-side; fabrication
  is structurally impossible in the returned payload.

Requires the ``gemini`` extra: ``pip install citeformer[gemini]``. The
extra pulls in ``google-genai`` (the unified SDK that replaced
``google-generativeai`` in 2025). Pass a live client via ``client=…`` to
override the default ``genai.Client()`` construction (which reads
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` from the environment).

Model requirements: any ``gemini-1.5`` or ``gemini-2.x`` family model
that supports structured output. ``gemini-2.0-flash`` is a good default.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from citeformer.backends.base import Backend
from citeformer.backends.openai import _flatten_segments
from citeformer.core import MarkerStyle, Policy, Source

_LOG = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.0-flash"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.7


class GeminiBackend(Backend):
    """Gemini backend with schema-level cite enforcement.

    Requests the same structured payload shape as :class:`OpenAIBackend`::

        {
          "segments": [
            {"text": "A sentence.", "citations": [1, 2]},
            {"text": "Another one.", "citations": [3]}
          ]
        }

    where ``citations[*]`` integers are enum-constrained to 1..N. After
    validation, segments are flattened into citation-marked plain text
    so downstream consumers (Citeformer / verify / render) see the same
    shape as local-backend output.

    Attributes:
        model: Gemini model identifier (``gemini-2.0-flash`` default).
        client: The authenticated ``google.genai.Client``.
    """

    model: str
    client: Any

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Construct a Gemini backend.

        Args:
            model: Gemini model id (``gemini-2.0-flash``, ``gemini-1.5-pro`` …).
            client: Pre-built ``genai.Client``. If ``None``, one is built
                from env (picks up ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``).
            **client_kwargs: Forwarded to ``genai.Client(**kwargs)`` when
                ``client`` is ``None`` (``api_key``, ``vertexai``, …).
        """
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "GeminiBackend requires the `gemini` extra. "
                "Install with `pip install citeformer[gemini]`."
            ) from e

        self.model = model
        self.client = client if client is not None else genai.Client(**client_kwargs)

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
            sources: Sources in scope. Position (1-indexed) becomes the enum entry.
            policy: Citation policy. Shapes the system instruction and the
                schema's ``minItems`` on the citations array (REQUIRED → 1,
                AUTO/QUOTES_ONLY → 0).
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
            raise ValueError("GeminiBackend requires at least 1 source")

        max_tokens = int(options.get("max_tokens", _DEFAULT_MAX_TOKENS))
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))
        marker_style = options.get("marker_style", MarkerStyle.BRACKET)
        schema = _build_citation_schema(n_sources=len(sources), policy=policy)
        system_instruction = _build_system_instruction(
            sources=sources,
            policy=policy,
            extra=options.get("system_prompt"),
        )

        response: Any = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
                "system_instruction": system_instruction,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        raw = getattr(response, "text", None) or ""
        return _flatten_segments(raw, marker_style=marker_style)

    def stream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> Iterator[str]:
        """Yield sentence-level chunks. Wraps :meth:`generate` + naive splitting.

        Gemini's true streaming surface emits partial JSON tokens which
        aren't safe to flatten per-chunk — we'd risk yielding a citation
        fragment. Slicing the validated response on sentence boundaries
        gives callers progressive output without violating the schema
        contract.
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


def _build_citation_schema(*, n_sources: int, policy: Policy) -> dict[str, Any]:
    """Build the Gemini-compatible schema.

    Gemini's OpenAPI subset supports ``type``, ``enum``, ``items``,
    ``properties``, ``required``, ``min_items`` (note the snake_case —
    that's the key the Python SDK expects), and ``description``.

    We **do not** set ``additionalProperties: false`` here — Gemini's
    validator ignores it silently on at least 2.0-flash, and including
    it on some variants triggers a 400. The schema is still constrained
    by the required-fields + enum combination.
    """
    cite_ids = list(range(1, n_sources + 1))
    min_citations = 1 if policy is Policy.REQUIRED else 0
    return {
        "type": "object",
        "required": ["segments"],
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "citations"],
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "A single sentence or clause of the response.",
                        },
                        "citations": {
                            "type": "array",
                            "description": (
                                f"Source indices (1..{n_sources}). "
                                f"Under {policy.value} policy, segment citations "
                                f"must be at least {min_citations}."
                            ),
                            "items": {
                                "type": "integer",
                                "enum": cite_ids,
                            },
                            "min_items": min_citations,
                        },
                    },
                },
            }
        },
    }


def _build_system_instruction(
    *,
    sources: list[Source],
    policy: Policy,
    extra: str | None,
) -> str:
    """Compose the system_instruction — same shape as OpenAI backend's.

    Gemini supports a dedicated ``system_instruction`` field rather than
    a ``system`` role in the messages list, so we thread the same
    information through there.
    """
    parts: list[str] = []
    if extra:
        parts.append(extra)
    parts.append(
        "You are answering with citations. Return a JSON object of shape "
        "{'segments': [{'text': '...', 'citations': [int, ...]}]}. "
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
    return "\n\n".join(parts)
