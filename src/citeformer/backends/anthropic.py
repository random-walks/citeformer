"""Anthropic backend — adapter over Anthropic's native Citations API.

Anthropic's Messages API has first-class Citations support (launched
Jan 2025): pass documents as `{"type": "document", ..., "citations": {"enabled": true}}`
and every assistant-side text block is decorated with an optional
``citations`` array referencing the document index + character span.

This backend is an **adapter**, not an enforcement layer. Claude's own
system ensures the returned citation references point at a document that
was actually provided — so fabricating a reference is a provider-side
impossibility. We translate Anthropic's native shape back into citeformer's
:class:`~citeformer.core.Citation` / :class:`~citeformer.core.Reference`
types so downstream code can mix Anthropic output with local-backend output
in the same pipeline.

Because the enforcement is native, ``marker_style`` is advisory on this
backend — we render Claude's citations in the chosen shape for consistency
with the rest of citeformer, but the provider itself doesn't know about
marker styles; it emits a structured citation block per assertion.

Requires the ``anthropic`` extra: ``pip install citeformer[anthropic]``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from citeformer.backends.base import Backend
from citeformer.core import MarkerStyle, Policy, Source

_LOG = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicBackend(Backend):
    """Anthropic Messages API backend with native citation support.

    Attributes:
        model: Anthropic model id (e.g. ``"claude-sonnet-4-6"``).
        client: The ``anthropic.Anthropic`` client.
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
            **options: ``max_tokens`` (default 1024), ``system_prompt``
                (optional extra system content), ``marker_style`` (default
                BRACKET — advisory, used to render citation markers).

        Returns:
            Flattened text carrying the configured marker style for every
            assertion Claude cited.

        Raises:
            ValueError: If ``sources`` is empty.
        """
        if len(sources) < 1:
            raise ValueError("AnthropicBackend requires at least 1 source")

        max_tokens = int(options.get("max_tokens", _DEFAULT_MAX_TOKENS))
        marker_style = options.get("marker_style", MarkerStyle.BRACKET)
        system_prompt = _build_system_prompt(policy, options.get("system_prompt"))

        documents = [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": src.content or str(src.metadata.get("title", f"Source {i}")),
                },
                "title": str(src.metadata.get("title", f"Source {i}")),
                "citations": {"enabled": True},
            }
            for i, src in enumerate(sources, start=1)
        ]

        message: Any = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        *documents,
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return _flatten_blocks(message.content, marker_style=marker_style)

    def stream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> Iterator[str]:
        """Yield the flattened response in sentence-sized chunks.

        Anthropic's streaming surface is block-oriented; we take the simpler
        path of calling :meth:`generate` once and slicing the result on
        punctuation so downstream chunk consumers still see progression.
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


def _flatten_blocks(content: Any, *, marker_style: MarkerStyle) -> str:
    """Fold Anthropic's block list back into plain text with inline markers.

    The Messages API returns ``content`` as a list of blocks. Text blocks
    (``block.type == 'text'``) may carry a ``citations`` list; each entry
    has ``document_index`` (0-indexed into the order we supplied
    documents). We emit a marker (``[N]`` / ``(N)`` / ``{N}`` / ``^N`` per
    marker_style) for each citation at the *end* of the referenced text
    block, remapping document_index → 1-indexed cite id.
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
            doc_index = (
                getattr(cite, "document_index", None)
                if not isinstance(cite, dict)
                else cite.get("document_index")
            )
            if doc_index is None:
                continue
            cid = int(doc_index) + 1  # Anthropic is 0-indexed; we're 1-indexed
            if cid not in seen:
                seen.add(cid)
                marker_suffix.append(f"{open_char}{cid}{close_char}")
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


def _delimiters_for(style: MarkerStyle) -> tuple[str, str]:
    return {
        MarkerStyle.BRACKET: ("[", "]"),
        MarkerStyle.PAREN: ("(", ")"),
        MarkerStyle.CURLY: ("{", "}"),
        MarkerStyle.CARET: ("^", ""),
    }[style]
