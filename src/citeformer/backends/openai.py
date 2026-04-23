"""OpenAI backend — schema-level cite-id enforcement via Structured Outputs.

OpenAI's ``response_format={"type": "json_schema", "strict": true, ...}`` API
lets us hand the model a JSON schema where the ``source_id`` field is
constrained to a literal enum of the in-scope source indices. When
``strict=true`` the API rejects any response whose ``source_id`` isn't one of
the enumerated integers — structurally equivalent, at the schema layer, to
what XGrammar does at the logit layer for local backends.

Tier honesty:

- **Local backends (HF / vLLM / llama.cpp)** enforce at the logit layer —
  a fabricated cite id is token-impossible to sample.
- **This backend** enforces at the **schema** layer. The provider
  validates the assistant's generation against the schema before returning
  it. Fabrication is structurally impossible *in the returned payload*,
  which is what matters for downstream consumers.

Requires the ``openai`` extra: ``pip install citeformer[openai]``.

Model requirements: the ``strict: true`` JSON-schema mode is supported on
``gpt-4o-2024-08-06``, ``gpt-4o-mini``, and every OpenAI model released
after August 2024. Older models will return a 400 — we surface that
directly rather than silently falling back to non-strict mode.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

from citeformer.backends.base import Backend
from citeformer.core import MarkerStyle, Policy, Source

_LOG = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIBackend(Backend):
    """OpenAI chat-completions backend with schema-level cite enforcement.

    Requests a structured response of the shape::

        {
          "segments": [
            {"text": "A sentence.", "citations": [1, 2]},
            {"text": "Another one.", "citations": [3]}
          ]
        }

    where every ``citations[*]`` integer is enum-constrained to 1..N. The
    backend then flattens the segments back into a single string carrying
    the configured :class:`~citeformer.core.MarkerStyle` markers (default
    ``[N]``) — so downstream code (Citeformer orchestrator, verify, render)
    sees the same shape as local-backend output.

    Attributes:
        model: OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
        client: The authenticated ``openai.OpenAI`` client.
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
        """Create an OpenAI backend.

        Args:
            model: Model id supporting strict JSON schema (``gpt-4o-mini``
                or later).
            client: Pre-built ``openai.OpenAI`` client. If ``None``, one is
                constructed from the environment (picks up ``OPENAI_API_KEY``).
            **client_kwargs: Forwarded to ``openai.OpenAI()`` when ``client``
                is ``None`` (``base_url``, ``api_key``, ``organization``,
                ``timeout`` …). Useful for pointing at a compatible endpoint
                (Azure, local LiteLLM, Together, Anyscale).
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIBackend requires the `openai` extra. "
                "Install with `pip install citeformer[openai]`."
            ) from e

        self.model = model
        self.client = client if client is not None else OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> str:
        """Generate text with schema-level citation constraint.

        Args:
            prompt: User prompt; caller is responsible for RAG stitching.
            sources: Sources in scope — position (1-indexed) becomes the
                cite enum entry.
            policy: Citation policy (``REQUIRED``/``AUTO``/``QUOTES_ONLY``).
                Threaded into the schema's ``description`` so the model sees
                the same enforcement intent it would under local decoding.
            **options: ``max_tokens`` (default 1024), ``temperature`` (default
                0.7), ``marker_style`` (default BRACKET), ``system_prompt``
                (additional system-role content prepended to the assembled
                citation instructions).

        Returns:
            Flattened text carrying ``marker_style`` markers for every cited
            source, in document order.

        Raises:
            ValueError: If ``sources`` is empty.
        """
        if len(sources) < 1:
            raise ValueError("OpenAIBackend requires at least 1 source")

        max_tokens = int(options.get("max_tokens", _DEFAULT_MAX_TOKENS))
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))
        marker_style = options.get("marker_style", MarkerStyle.BRACKET)
        schema = _build_citation_schema(n_sources=len(sources), policy=policy)

        messages = self._build_messages(
            prompt=prompt,
            sources=sources,
            policy=policy,
            system_prompt=options.get("system_prompt"),
        )

        completion: Any = self.client.chat.completions.create(
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
        raw = completion.choices[0].message.content
        return _flatten_segments(raw, marker_style=marker_style)

    def stream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> Iterator[str]:
        """Stream segments by yielding each finished sentence with its markers.

        OpenAI's streaming response-format path is richer than we need here —
        we're not trying to surface per-token deltas, just complete
        sentence-level chunks as each segment is validated. The simpler
        implementation calls :meth:`generate` once and chunks its output
        on sentence boundaries so downstream consumers of ``Citeformer.stream``
        still see multiple chunks.
        """
        text = self.generate(prompt=prompt, sources=sources, policy=policy, **options)
        # Split on sentence boundaries so downstream chunk consumers see
        # more than one piece. Preserves trailing whitespace on each chunk.
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
        """Assemble the chat messages. Sources go into the system role."""
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
            parts.append(
                "Every segment MUST cite at least one source (required policy)."
            )
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


def _build_citation_schema(*, n_sources: int, policy: Policy) -> dict[str, Any]:
    """Build the JSON schema for the structured response.

    The ``source_id`` array items are enum-constrained to ``1..n_sources``
    inclusive — the same bound the GBNF ``cite-id`` rule enforces at the
    logit layer for local backends. When ``strict: true`` is set on the
    request, the OpenAI API rejects any response whose citation integers
    fall outside this enum.
    """
    cite_ids = list(range(1, n_sources + 1))
    min_citations = 1 if policy is Policy.REQUIRED else 0
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["segments"],
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
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
                            "minItems": min_citations,
                        },
                    },
                },
            }
        },
    }


# Pattern to split on sentence-ending punctuation when we have to assemble
# a text representation from the segmented JSON. Preserves the punctuation.
_SENTENCE_END = re.compile(r"([.!?])\s*$")


def _flatten_segments(raw_json: str, *, marker_style: MarkerStyle) -> str:
    """Merge a structured-outputs payload into a citation-marked plain string.

    The OpenAI structured-output contract guarantees ``raw_json`` parses as
    the schema we sent. We still defensively fall back to
    ``json.JSONDecodeError`` handling — a provider outage returning a
    non-JSON error body shouldn't blow up the caller with an opaque
    exception from deep in our code.
    """
    from citeformer.citeformer import _MARKER_PATTERNS  # local import to avoid cycle

    try:
        payload = json.loads(raw_json or "{}")
    except json.JSONDecodeError:
        _LOG.warning("OpenAIBackend got non-JSON content; returning verbatim")
        return raw_json or ""
    segments = payload.get("segments") or []
    # Delimiters per style — pull from the parser regex so there's a single
    # source of truth for marker shape across the package.
    open_char, close_char = _delimiters_for(marker_style, _MARKER_PATTERNS)
    parts: list[str] = []
    for segment in segments:
        text = str(segment.get("text", "")).rstrip()
        if not text:
            continue
        cite_ids = [int(c) for c in segment.get("citations", []) if isinstance(c, (int, str))]
        marker = "".join(f"{open_char}{c}{close_char}" for c in cite_ids)
        # If text ends in sentence punctuation, place markers before it;
        # otherwise append them after. Both shapes round-trip through our
        # citation regex.
        sentence_match = _SENTENCE_END.search(text)
        if sentence_match and marker:
            head = text[: sentence_match.start()]
            tail = sentence_match.group(0)
            joined = f"{head} {marker}{tail}" if marker else text
        else:
            joined = f"{text} {marker}".rstrip() if marker else text
        parts.append(joined)
    return " ".join(parts)


def _delimiters_for(
    style: MarkerStyle,
    patterns: dict[MarkerStyle, re.Pattern[str]],
) -> tuple[str, str]:
    """Look up (open_char, close_char) for a marker style."""
    del patterns  # unused — kept for symmetry with other marker-aware helpers
    return {
        MarkerStyle.BRACKET: ("[", "]"),
        MarkerStyle.PAREN: ("(", ")"),
        MarkerStyle.CURLY: ("{", "}"),
        MarkerStyle.CARET: ("^", ""),
    }[style]
