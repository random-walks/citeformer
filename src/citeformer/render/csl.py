"""Deterministic bibliography rendering via home-grown formatters.

Given a source list + the citation list emitted by generation, we produce
one `Reference` per unique cited `source_id` using a `CitationFormatter`
from `citeformer.render.formatters`. Fabricated citations (out of range)
are silently dropped — `verify()` surfaces them via the coverage check.

We stopped consuming citeproc-py in the home-grown rewrite (see
``docs/decisions/004-citeproc-rewrite.md``) because of accumulated quirks
(Chicago page-range crash, APA double-period, noisy CSL-JSON warnings)
and because Vancouver has no canonical bundled style in the upstream
CSL repo. citeproc-py is still available behind the optional
``citeproc-compat`` extra for users who want the "any of 10,000 CSL
files" escape hatch.

``GenerationResult.text`` continues to carry the model-emitted ``[N]``
markers verbatim (see ``docs/decisions/002-inline-markers-stay-numeric.md``).
Only ``Reference.inline_marker`` is rendered in the style's native form.
"""

from __future__ import annotations

from citeformer.core import Citation, Reference, Source
from citeformer.render.formatters import CitationFormatter, get_formatter


def render_references(
    sources: list[Source],
    citations: list[Citation],
    style_name: str,
) -> list[Reference]:
    """Render a deterministic bibliography for the cited sources.

    Args:
        sources: All sources in scope (1-indexed positions).
        citations: Citations parsed from generated text.
        style_name: Bundled style (``"apa-7"``, ``"ieee"``, …). Aliases
            (``"apa"``, ``"mla"``, ``"chicago"``) also work.

    Returns:
        One `Reference` per unique in-range cited `source_id`, in ascending
        id order. Each has the style's native inline marker and the full
        bibliography entry.

    Raises:
        ValueError: If ``style_name`` isn't a known built-in formatter.
    """
    cited_ids = sorted({c.source_id for c in citations if 1 <= c.source_id <= len(sources)})
    if not cited_ids:
        return []

    formatter = get_formatter(style_name)
    references: list[Reference] = []
    for position, cid in enumerate(cited_ids, start=1):
        item = sources[cid - 1].metadata
        references.append(
            Reference(
                source_id=cid,
                inline_marker=formatter.inline(item, position),
                rendered=formatter.bibliography(item, position),
            )
        )
    return references


def render_single_reference(
    source: Source,
    *,
    style_name: str,
    number: int = 1,
) -> Reference:
    """Render a `Reference` for a single `Source` without a full `Citation` list.

    Handy for previewing what a source will look like before you feed it to
    the model — e.g. in docs, example notebooks, or user interfaces.

    Args:
        source: The source to format.
        style_name: Bundled style or alias.
        number: Bibliography position to use for numeric styles; ignored by
            author-date styles.

    Returns:
        A `Reference` with the style's inline marker + full entry.
    """
    formatter = get_formatter(style_name)
    return Reference(
        source_id=number,
        inline_marker=formatter.inline(source.metadata, number),
        rendered=formatter.bibliography(source.metadata, number),
    )


# Re-export for convenience.
__all__ = ["CitationFormatter", "render_references", "render_single_reference"]
