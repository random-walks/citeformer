"""Anthropic backend — adapter over Anthropic's native Citations API.

Anthropic's Messages API has first-class Citations support (launched
Jan 2025): pass documents as ``{"type": "document", ..., "citations":
{"enabled": true}}`` and every assistant-side text block is decorated
with an optional ``citations`` array referencing the document index +
character span.

This backend is an **adapter**, not an enforcement layer. Claude's own
system ensures the returned citation references point at a document that
was actually provided — fabricating a reference is provider-side
impossible. We translate Anthropic's native shape back into citeformer's
:class:`~citeformer.core.Citation` / :class:`~citeformer.core.Reference`
types so downstream code can mix Anthropic output with local-backend
output in the same pipeline.

Because the enforcement is native, ``marker_style`` is advisory on this
backend — we render Claude's citations in the chosen shape for
consistency with the rest of citeformer, but the provider itself doesn't
know about marker styles; it emits a structured citation block per
assertion.

Prompt caching (``cache_control``) is on by default for the document
blocks. Claude prices cache-read tokens at ~10% of fresh input tokens,
so for any RAG pipeline that reuses the same source list across calls
the saving is substantial. Disable with ``use_prompt_cache=False`` if
the documents are one-shot.

True per-block streaming via :meth:`stream` is wired to the SDK's
``messages.stream()`` context manager — text deltas are batched per
block so the citation markers attach to the right block when the block
finishes (the per-token delta path doesn't carry citation info on the
wire; you only see citations at ``content_block_stop``).

Requires the ``anthropic`` extra: ``pip install citeformer[anthropic]``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from citeformer.backends.base import Backend
from citeformer.core import MarkerStyle, Policy, Source, TokenUsage

_LOG = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 1.0  # Anthropic's API default; honoured if caller passes one
_DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicBackend(Backend):
    """Anthropic Messages API backend with native citation support.

    Attributes:
        model: Anthropic model id (e.g. ``"claude-sonnet-4-6"``).
        client: The ``anthropic.Anthropic`` client.
        last_usage: Token-usage payload from the most recent ``generate()``
            / ``stream()`` call. ``None`` before the first call. The
            orchestrator threads this onto :attr:`GenerationResult.usage`.
        last_rich_citations: One dict per marker emitted in the most
            recent call, in left-to-right output order. Each carries the
            ``source_id``, ``cited_text`` (the exact span Claude cited
            from), ``source_span`` (offsets into the source content), and
            ``document_title`` returned by the Citations API. The
            orchestrator zips this with the parsed marker list and
            populates :attr:`Citation.cited_text` / ``source_span`` /
            ``document_title``. Empty list when the call emitted no
            citations.
    """

    model: str
    client: Any
    last_usage: TokenUsage | None
    last_rich_citations: list[dict[str, Any]]

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Construct an Anthropic backend.

        Args:
            model: Anthropic model id supporting Citations (any 3.5+ or
                Claude 4 family).
            client: Pre-built ``anthropic.Anthropic`` client. If ``None``,
                one is constructed from the environment (picks up
                ``ANTHROPIC_API_KEY``).
            **client_kwargs: Forwarded to ``Anthropic()`` when ``client`` is
                ``None``.
        """
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicBackend requires the `anthropic` extra. "
                "Install with `pip install citeformer[anthropic]`."
            ) from e

        self.model = model
        self.client = client if client is not None else Anthropic(**client_kwargs)
        self.last_usage = None
        self.last_rich_citations = []

    def generate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> str:
        """Call Messages API with citations enabled; flatten to marker-decorated text.

        Args:
            prompt: User prompt.
            sources: Sources in scope. Each becomes one document block.
            policy: Citation policy — threaded into the system prompt so
                Claude sees the caller's enforcement intent. The provider
                itself doesn't have a typed policy, so we rely on the
                system prompt to shape behaviour.
            **options: ``max_tokens`` (default 1024), ``temperature``
                (default Anthropic's own default — passed through only
                when explicitly supplied), ``system_prompt`` (extra
                system content), ``marker_style`` (default BRACKET —
                advisory, used to render citation markers),
                ``use_prompt_cache`` (default ``True``; sets
                ``cache_control: ephemeral`` on every document block so
                repeat-source RAG pays cache-read prices on subsequent
                calls), ``extra_headers`` (forwarded to the SDK).

        Returns:
            Flattened text carrying the configured marker style for every
            assertion Claude cited.

        Raises:
            ValueError: If ``sources`` is empty.
        """
        if len(sources) < 1:
            raise ValueError("AnthropicBackend requires at least 1 source")

        request_kwargs = self._build_request(prompt, sources, policy, options)
        marker_style = options.get("marker_style", MarkerStyle.BRACKET)

        message: Any = self.client.messages.create(**request_kwargs)
        self.last_usage = _extract_usage(getattr(message, "usage", None))
        record: list[dict[str, Any]] = []
        text = _flatten_blocks(message.content, marker_style=marker_style, record=record)
        self.last_rich_citations = record
        return text

    def stream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> Iterator[str]:
        """Stream block-sized chunks via Anthropic's native ``messages.stream()``.

        Each yielded chunk corresponds to one finished text block from
        Claude — text + the marker(s) for any citations attached to that
        block. Yielding per-block (rather than per-token) is the natural
        granularity for the Citations API: citation events only arrive at
        ``content_block_stop``, so per-token text deltas would have to be
        rewritten in-place when the citations land. The per-block path is
        honest and produces clean output.

        Falls back to the non-streaming path on SDKs that don't expose
        ``messages.stream`` (very old client versions or test stand-ins
        that mock only ``messages.create``).

        Args:
            prompt: See :meth:`generate`.
            sources: See :meth:`generate`.
            policy: See :meth:`generate`.
            **options: Same options as :meth:`generate`.

        Yields:
            Per-block text chunks (each terminated by a single space)
            carrying any citation markers that landed on the block.
        """
        if len(sources) < 1:
            raise ValueError("AnthropicBackend requires at least 1 source")

        marker_style = options.get("marker_style", MarkerStyle.BRACKET)
        request_kwargs = self._build_request(prompt, sources, policy, options)

        stream_method = getattr(self.client.messages, "stream", None)
        if stream_method is None:
            # Old SDK or a fake client that only mocked `create` — fall back
            # to the non-streaming path so callers still get a usable result.
            yield self.generate(prompt=prompt, sources=sources, policy=policy, **options)
            return

        record: list[dict[str, Any]] = []
        with stream_method(**request_kwargs) as stream:
            for block in _iter_completed_blocks(stream):
                rendered = _render_block(block, marker_style=marker_style, record=record)
                if rendered:
                    yield rendered + " "
            final_message = stream.get_final_message()
        self.last_usage = _extract_usage(getattr(final_message, "usage", None))
        self.last_rich_citations = record

    def _build_request(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the kwargs dict shared by ``generate()`` and ``stream()``.

        Centralised so caching, system-prompt assembly, and document-block
        construction stay consistent across the two entry points — and so
        the unit tests only need to verify one shape.
        """
        max_tokens = int(options.get("max_tokens", _DEFAULT_MAX_TOKENS))
        system_prompt = _build_system_prompt(policy, options.get("system_prompt"))
        use_cache = bool(options.get("use_prompt_cache", True))
        documents = _build_documents(sources, use_cache=use_cache)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *documents,
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if "temperature" in options:
            request_kwargs["temperature"] = float(options["temperature"])
        elif options.get("_force_default_temperature"):
            request_kwargs["temperature"] = _DEFAULT_TEMPERATURE
        if "extra_headers" in options:
            request_kwargs["extra_headers"] = options["extra_headers"]
        return request_kwargs


def _build_documents(sources: list[Source], *, use_cache: bool) -> list[dict[str, Any]]:
    """Build the document content blocks Claude consumes for citations.

    Setting ``cache_control: {"type": "ephemeral"}`` on each document
    block opts the prefix into Anthropic's prompt-caching path — repeat
    calls with the same source list bill cache-read tokens (~10% of
    input) instead of full input tokens. Set ``use_cache=False`` for
    truly one-shot calls where caching is overhead.
    """
    documents: list[dict[str, Any]] = []
    for i, src in enumerate(sources, start=1):
        block: dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": src.content or str(src.metadata.get("title", f"Source {i}")),
            },
            "title": str(src.metadata.get("title", f"Source {i}")),
            "citations": {"enabled": True},
        }
        if use_cache:
            block["cache_control"] = {"type": "ephemeral"}
        documents.append(block)
    return documents


def _build_system_prompt(policy: Policy, extra: str | None) -> str:
    """Assemble the system prompt that nudges Claude to the right citation density."""
    parts: list[str] = []
    if extra:
        parts.append(extra)
    if policy is Policy.REQUIRED:
        parts.append(
            "Cite EVERY assertion in your response. Every sentence must "
            "reference at least one provided document."
        )
    elif policy is Policy.QUOTES_ONLY:
        parts.append(
            "Cite any direct quotation from the provided documents. Paraphrases may be uncited."
        )
    else:
        parts.append(
            "Cite assertions that depend on specific information from the "
            "provided documents. Avoid citing restatements of the user's "
            "question."
        )
    return "\n\n".join(parts) if parts else "Cite your sources."


def _extract_usage(raw: Any) -> TokenUsage | None:
    """Pull ``input_tokens`` / ``output_tokens`` (and cache fields) off a usage object.

    Handles both the SDK's typed ``Usage`` object (attribute access) and
    a plain dict (the unit tests' fake clients return SimpleNamespace,
    real SDKs may evolve, and dicts arrive when consumers unpickle a
    response). Missing fields collapse to zero / ``None``.
    """
    if raw is None:
        return None

    def _get(name: str) -> Any:
        if isinstance(raw, dict):
            return raw.get(name)
        return getattr(raw, name, None)

    input_tokens = _get("input_tokens")
    output_tokens = _get("output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    cache_creation = _get("cache_creation_input_tokens")
    cache_read = _get("cache_read_input_tokens")
    return TokenUsage(
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
        cache_creation_input_tokens=(int(cache_creation) if cache_creation is not None else None),
        cache_read_input_tokens=(int(cache_read) if cache_read is not None else None),
    )


def _iter_completed_blocks(stream: Any) -> Iterator[Any]:
    """Yield each block the stream finishes, in order.

    Anthropic's streaming surface emits a sequence of typed events; we
    only care about ``content_block_stop`` (the moment a block is
    complete with all its citations attached). Different SDK versions
    surface the block payload at slightly different attribute names —
    we try the common ones in order.
    """
    for event in stream:
        event_type = getattr(event, "type", None) or (
            event.get("type") if isinstance(event, dict) else None
        )
        if event_type != "content_block_stop":
            continue
        block: Any = (
            getattr(event, "content_block", None)
            or getattr(event, "block", None)
            or (event.get("content_block") if isinstance(event, dict) else None)
            or (event.get("block") if isinstance(event, dict) else None)
        )
        if block is not None:
            yield block


def _render_block(
    block: Any,
    *,
    marker_style: MarkerStyle,
    record: list[dict[str, Any]] | None = None,
) -> str:
    """Render one content block to text + trailing markers (or just text)."""
    return _flatten_blocks([block], marker_style=marker_style, record=record)


def _flatten_blocks(
    content: Any,
    *,
    marker_style: MarkerStyle,
    record: list[dict[str, Any]] | None = None,
) -> str:
    """Fold Anthropic's block list back into plain text with inline markers.

    The Messages API returns ``content`` as a list of blocks. Text blocks
    (``block.type == 'text'``) may carry a ``citations`` list; each entry
    has ``document_index`` (0-indexed into the order we supplied
    documents). We emit a marker (``[N]`` / ``(N)`` / ``{N}`` / ``^N`` per
    marker_style) for each citation at the *end* of the referenced text
    block, remapping document_index → 1-indexed cite id.

    When ``record`` is supplied, one dict is appended for every marker
    actually emitted (left-to-right, matching the order the regex parser
    will see them) carrying ``source_id``, ``cited_text``,
    ``source_span``, and ``document_title`` from the citation event. The
    Anthropic backend's ``last_rich_citations`` is wired through this
    side-channel; the orchestrator zips it with the parsed marker list
    to populate the rich :class:`Citation` fields.
    """
    open_char, close_char = _delimiters_for(marker_style)
    parts: list[str] = []
    for block in content or []:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "text":
            continue
        text = str(
            getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
        )
        citations = (
            getattr(block, "citations", None)
            or (block.get("citations") if isinstance(block, dict) else None)
            or []
        )
        seen: set[int] = set()
        marker_suffix: list[str] = []
        for cite in citations:
            doc_index = _attr_or_key(cite, "document_index")
            if doc_index is None:
                continue
            cid = int(doc_index) + 1  # Anthropic is 0-indexed; we're 1-indexed
            if cid in seen:
                continue
            seen.add(cid)
            marker_suffix.append(f"{open_char}{cid}{close_char}")
            if record is not None:
                start = _attr_or_key(cite, "start_char_index")
                end = _attr_or_key(cite, "end_char_index")
                source_span = (
                    (int(start), int(end)) if start is not None and end is not None else None
                )
                record.append(
                    {
                        "source_id": cid,
                        "cited_text": _attr_or_key(cite, "cited_text"),
                        "source_span": source_span,
                        "document_title": _attr_or_key(cite, "document_title"),
                    }
                )
        joined = text.rstrip()
        if marker_suffix:
            suffix = " ".join(marker_suffix)
            # If the text ends in sentence punctuation, put markers before it.
            if joined and joined[-1] in ".!?":
                parts.append(f"{joined[:-1]} {suffix}{joined[-1]}")
            else:
                parts.append(f"{joined} {suffix}")
        else:
            parts.append(joined)
    return " ".join(parts).strip()


def _attr_or_key(obj: Any, name: str) -> Any:
    """Read ``name`` off either an object (attribute) or a dict (key).

    Anthropic's SDK returns typed objects in production; some
    serialisations and the unit-test fakes use plain dicts. Both shapes
    must work transparently across the citation-attribute reads.
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _delimiters_for(style: MarkerStyle) -> tuple[str, str]:
    return {
        MarkerStyle.BRACKET: ("[", "]"),
        MarkerStyle.PAREN: ("(", ")"),
        MarkerStyle.CURLY: ("{", "}"),
        MarkerStyle.CARET: ("^", ""),
    }[style]
