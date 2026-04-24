"""Prompt assembly helpers for RAG-style generation with citations.

Most users of citeformer stitch their own prompt before calling
``Citeformer.generate()`` — they add a system message, list the sources, drop
in a few citation-density hints, and append the task. That boilerplate is
easy to get subtly wrong (misnumber the sources, forget to show the ``[N]``
example, bury the task under too much preamble), so we ship a canonical
builder here.

`build_rag_prompt` is intentionally string-in / string-out — the returned
string is the exact prompt to feed to `Citeformer.generate()`. No chat
templates applied: that's the caller's job (model-specific). If you need
to format with a specific model's chat template, wrap the string returned
here as a user message and apply the template yourself.

The design goal is "helpful default, trivially overridable". Every section
is optional; pass ``None`` / empty to skip.
"""

from __future__ import annotations

from citeformer.core import Source

_DEFAULT_CITE_HINT = (
    "Cite every claim with a ``[N]`` marker, where N is the number of the "
    "source. Do not invent citations."
)


def build_rag_prompt(
    *,
    query: str,
    sources: list[Source],
    system: str | None = None,
    cite_hint: str | None = _DEFAULT_CITE_HINT,
    example: str | None = None,
    answer_prefix: str | None = "Answer:",
) -> str:
    """Assemble a RAG-style prompt with numbered source context.

    Args:
        query: The user-facing task or question. Required.
        sources: Sources in scope. Their 1-indexed position in the list
            becomes the citation id the model is allowed to emit. Must be
            non-empty.
        system: Optional top-of-prompt system message (role framing, style
            guidance). Rendered verbatim at the top of the prompt.
        cite_hint: Short instruction telling the model how to cite.
            Defaults to a reasonable generic hint; pass ``None`` to omit
            (useful if your ``system`` already explains citation conventions).
        example: Optional one-line example sentence showing the ``[N]``
            pattern in context. Helps small models imitate the shape.
        answer_prefix: Trailing token(s) that invite the model to start the
            answer (e.g. ``"Answer:"``, ``"Survey:"``, ``"Summary:"``).
            Set to ``None`` to omit — useful when your chat template adds
            its own assistant-turn prefix.

    Returns:
        A string suitable for passing to ``Citeformer.generate(prompt=…)``.

    Raises:
        ValueError: If ``query`` is empty or ``sources`` is empty.

    Example:
        >>> prompt = build_rag_prompt(
        ...     query="Explain self-attention.",
        ...     sources=[Source(metadata={"id": "vaswani", "type": "article-journal",
        ...                               "title": "Attention Is All You Need",
        ...                               "author": [{"family": "Vaswani"}]},
        ...                     content="...")],
        ... )
        >>> "[1]" in prompt
        True
    """
    if not query.strip():
        raise ValueError("build_rag_prompt requires a non-empty `query`.")
    if not sources:
        raise ValueError("build_rag_prompt requires at least 1 source.")

    parts: list[str] = []
    if system:
        parts.append(system.strip())

    parts.append("Sources:\n" + _format_source_list(sources))

    if cite_hint:
        parts.append(cite_hint.strip())

    if example:
        parts.append("Example: " + example.strip())

    parts.append("Task: " + query.strip())

    if answer_prefix:
        parts.append(answer_prefix.strip())

    # Blank line separators between sections keep it readable for humans
    # debugging prompt behavior.
    return "\n\n".join(parts)


def _format_source_list(sources: list[Source]) -> str:
    """Render sources as a numbered ``[N] Author: Title`` list."""
    lines = []
    for i, source in enumerate(sources, start=1):
        metadata = source.metadata
        title = str(metadata.get("title", "Untitled")).strip()
        author_str = _format_author_tag(metadata.get("author") or [])
        if author_str:
            lines.append(f"[{i}] {author_str}: {title}")
        else:
            lines.append(f"[{i}] {title}")
    return "\n".join(lines)


def _format_author_tag(authors_raw: list[object] | object) -> str:
    """Compact author tag: ``Smith``, ``Smith & Jones``, ``Smith et al.``.

    Pulls family names (or ``literal`` for organizational authors). Caps at
    three displayed names before switching to ``et al.``.
    """
    if not isinstance(authors_raw, list) or not authors_raw:
        return ""
    names: list[str] = []
    for a in authors_raw[:3]:
        if not isinstance(a, dict):
            continue
        family = a.get("family") or a.get("literal") or ""
        family = str(family).strip()
        if family:
            names.append(family)
    if not names:
        return ""
    if len(authors_raw) > 3:
        return ", ".join(names) + " et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    return ", ".join(names)


__all__ = ["build_rag_prompt"]
