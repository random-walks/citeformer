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
from collections.abc import AsyncIterator, Iterator
from typing import Any

from citeformer.backends.base import Backend
from citeformer.core import MarkerStyle, Policy, Source, TokenUsage

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
        last_usage: Token-usage payload from the most recent ``generate()``
            call. ``None`` before the first call. The orchestrator threads
            this onto :attr:`GenerationResult.usage`.
    """

    model: str
    last_usage: TokenUsage | None

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        client: Any | None = None,
        async_client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Create an OpenAI backend.

        Args:
            model: Model id supporting strict JSON schema (``gpt-4o-mini``
                or later).
            client: Pre-built ``openai.OpenAI`` client used by :meth:`generate`
                / :meth:`stream`. If ``None``, one is built lazily from
                ``client_kwargs`` on the first sync call (picks up
                ``OPENAI_API_KEY`` from env then).
            async_client: Pre-built ``openai.AsyncOpenAI`` client used by
                :meth:`agenerate` / :meth:`astream` (ADR-014). If ``None``,
                one is built lazily from ``client_kwargs`` on the first async
                call. Sync-only callers don't pay the async construction
                cost; async-only callers don't pay the sync one.
            **client_kwargs: Forwarded to ``openai.OpenAI()`` /
                ``openai.AsyncOpenAI()`` when the respective client is ``None``
                (``base_url``, ``api_key``, ``organization``, ``timeout``, …).
                Useful for pointing at a compatible endpoint (Azure, local
                LiteLLM, Together, Anyscale).
        """
        try:
            from openai import OpenAI  # noqa: F401  — verifies the extra is installed
        except ImportError as e:
            raise ImportError(
                "OpenAIBackend requires the `openai` extra. "
                "Install with `pip install citeformer[openai]`."
            ) from e

        self.model = model
        self._client_kwargs = dict(client_kwargs)
        self._sync_client_override: Any | None = client
        self._sync_client_cache: Any | None = None
        self._async_client_override: Any | None = async_client
        self._async_client_cache: Any | None = None
        self.last_usage = None

    @property
    def client(self) -> Any:
        """Lazy ``openai.OpenAI`` client used by the sync surface.

        Built on first access from ``client_kwargs`` (or returns the
        constructor-supplied override). Async-only callers never trigger
        construction — important for tests that inject only an
        ``async_client`` without setting ``OPENAI_API_KEY``.
        """
        if self._sync_client_override is not None:
            return self._sync_client_override
        if self._sync_client_cache is None:
            from openai import OpenAI

            self._sync_client_cache = OpenAI(**self._client_kwargs)
        return self._sync_client_cache

    @property
    def async_client(self) -> Any:
        """Lazy ``openai.AsyncOpenAI`` client used by the async surface.

        Built on first access from the same ``client_kwargs`` the sync client
        uses (so a backend pointing at ``base_url=...`` for OpenRouter /
        Fireworks / Together cascades correctly to the async client too).
        Sync-only callers never trigger construction.
        """
        if self._async_client_override is not None:
            return self._async_client_override
        if self._async_client_cache is None:
            from openai import AsyncOpenAI

            self._async_client_cache = AsyncOpenAI(**self._client_kwargs)
        return self._async_client_cache

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

        messages = self._build_messages(
            prompt=prompt,
            sources=sources,
            policy=policy,
            system_prompt=options.get("system_prompt"),
        )
        response_format = self._build_response_format(
            n_sources=len(sources),
            policy=policy,
            marker_style=marker_style,
        )

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
        }
        # Subclasses (OpenRouter) inject extra fields via _augment_create_kwargs.
        self._augment_create_kwargs(create_kwargs, options=options)
        completion: Any = self.client.chat.completions.create(**create_kwargs)
        self.last_usage = _extract_openai_usage(getattr(completion, "usage", None))
        raw = completion.choices[0].message.content
        return self._decode_response_text(raw, marker_style=marker_style)

    async def agenerate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> str:
        """Native-async counterpart of :meth:`generate` (ADR-014).

        Uses ``self.async_client`` (the lazy ``AsyncOpenAI``) so concurrent
        callers don't tie up executor threads on the SDK's HTTP wait. The
        request shape, schema construction, segment flattening, and
        ``last_usage`` extraction are identical to the sync path — only the
        client call is awaited. Subclasses (OpenRouter / Fireworks /
        Together) inherit this unchanged; their ``_build_response_format``
        / ``_augment_create_kwargs`` hooks fire from here just like in
        the sync path.
        """
        if len(sources) < 1:
            raise ValueError("OpenAIBackend requires at least 1 source")

        max_tokens = int(options.get("max_tokens", _DEFAULT_MAX_TOKENS))
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))
        marker_style = options.get("marker_style", MarkerStyle.BRACKET)

        messages = self._build_messages(
            prompt=prompt,
            sources=sources,
            policy=policy,
            system_prompt=options.get("system_prompt"),
        )
        response_format = self._build_response_format(
            n_sources=len(sources),
            policy=policy,
            marker_style=marker_style,
        )

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
        }
        self._augment_create_kwargs(create_kwargs, options=options)
        completion: Any = await self.async_client.chat.completions.create(**create_kwargs)
        self.last_usage = _extract_openai_usage(getattr(completion, "usage", None))
        raw = completion.choices[0].message.content
        return self._decode_response_text(raw, marker_style=marker_style)

    def _build_response_format(
        self,
        *,
        n_sources: int,
        policy: Policy,
        marker_style: MarkerStyle,
    ) -> dict[str, Any]:
        """Construct the ``response_format`` payload for the completion call.

        OpenAI's strict-mode JSON schema with enum-bounded citation ids is
        the default. Fireworks overrides this to return a native GBNF
        grammar instead (``{"type": "grammar", "grammar": ...}``) since
        their runtime accepts a raw grammar string. ``marker_style`` is
        ignored at this layer for OpenAI (the segments shape doesn't carry
        marker delimiters; flattening picks them up separately) but is
        threaded through so subclasses with grammar-shaped response
        formats can inline the right delimiter terminals.
        """
        del marker_style  # OpenAI's segment flattener picks marker_style up separately
        schema = _build_citation_schema(n_sources=n_sources, policy=policy)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "CitedSegments",
                "strict": True,
                "schema": schema,
            },
        }

    def _decode_response_text(self, raw: str, *, marker_style: MarkerStyle) -> str:
        """Decode the model's response into citation-marker plain text.

        OpenAI's strict-mode response is a JSON segments object; we flatten
        it to text with inline markers. Fireworks (and any other backend
        whose response_format yields plain text directly, like a grammar)
        overrides this to a passthrough.
        """
        return _flatten_segments(raw, marker_style=marker_style)

    def _augment_create_kwargs(
        self,
        kwargs: dict[str, Any],
        *,
        options: dict[str, Any],
    ) -> None:
        """Hook for subclasses to inject provider-specific request fields.

        The base OpenAI backend is a no-op; OpenRouter overrides this to
        thread ``extra_body`` (provider routing) and ``extra_headers`` (app
        attribution) onto the completion call without duplicating any of the
        schema or message-assembly logic.
        """
        del kwargs, options  # intentionally unused on the base backend

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
        yield from _chunk_on_sentences(text)

    async def astream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> AsyncIterator[str]:
        """Native-async counterpart of :meth:`stream` (ADR-014).

        Awaits :meth:`agenerate` (uses the async client) and then yields the
        same sentence-chunked output the sync :meth:`stream` produces.
        Cascades to OpenRouter / Fireworks / Together since they don't
        override ``stream`` either.
        """
        text = await self.agenerate(prompt=prompt, sources=sources, policy=policy, **options)
        for chunk in _chunk_on_sentences(text):
            yield chunk

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


def _chunk_on_sentences(text: str) -> Iterator[str]:
    """Yield ``text`` in sentence-boundary chunks for streaming UX.

    Shared between :meth:`OpenAIBackend.stream` and
    :meth:`OpenAIBackend.astream` so the two paths produce byte-for-byte
    identical chunk sequences.
    """
    buf: list[str] = []
    for char in text:
        buf.append(char)
        if char in ".!?" and buf:
            yield "".join(buf) + " "
            buf = []
    if buf:
        yield "".join(buf)


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


def _extract_openai_usage(raw: Any) -> TokenUsage | None:
    """Pull token counts off an OpenAI-style ``usage`` payload.

    Shared by ``OpenAIBackend``, ``MistralBackend`` (the SDK's response
    shape mirrors OpenAI's), and ``OpenRouterBackend``. Handles both
    object and dict shapes — fake clients in unit tests use
    SimpleNamespace, real SDKs use typed objects, OpenRouter occasionally
    surfaces extra fields like ``cost`` and ``prompt_tokens_details``.
    """
    if raw is None:
        return None

    def _get(name: str) -> Any:
        if isinstance(raw, dict):
            return raw.get(name)
        return getattr(raw, name, None)

    prompt = _get("prompt_tokens")
    completion = _get("completion_tokens")
    if prompt is None and completion is None:
        return None

    cached = None
    details = _get("prompt_tokens_details")
    if details is not None:
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
        else:
            cached = getattr(details, "cached_tokens", None)

    cost = _get("cost")
    return TokenUsage(
        input_tokens=int(prompt or 0),
        output_tokens=int(completion or 0),
        cache_read_input_tokens=int(cached) if cached is not None else None,
        cost_credits=float(cost) if cost is not None else None,
    )
