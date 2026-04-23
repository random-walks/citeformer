"""CSL bibliography rendering via citeproc-py.

The "model never touches the reference list" promise lives here. Given the
`Source` list + the `Citation` list that came out of generation, we render:

- One `Reference` per unique cited `source_id`.
- `Reference.inline_marker`: the style's native inline form (e.g. ``"(Smith,
  2023)"`` for APA, ``"[1]"`` for IEEE).
- `Reference.rendered`: the full bibliography entry in the requested CSL
  style (e.g. ``"Smith, J. (2023). Title. Journal, 12(3), 45-67."``).

The inline generated text (`GenerationResult.text`) keeps the model-emitted
``[N]`` markers verbatim — we don't post-process them to match the style's
inline form. The grammar guarantees numeric markers; the reference list shows
the style-appropriate form. Users who want a fully-rendered inline output can
do their own substitution via the `Reference.inline_marker` values.
"""

from __future__ import annotations

from typing import Any

from citeformer.core import Citation, Reference, Source
from citeformer.render.styles import load_style


def render_references(
    sources: list[Source],
    citations: list[Citation],
    style_name: str,
) -> list[Reference]:
    """Render a deterministic bibliography for the cited sources.

    Args:
        sources: All sources in scope (1-indexed positions).
        citations: Citations parsed from generated text.
        style_name: Bundled style (``"apa-7"``, ``"ieee"``, …) or a
            filesystem path to a ``.csl`` file.

    Returns:
        One `Reference` per unique in-range cited `source_id`, in ascending id
        order. Out-of-range citations (which should never happen under
        grammar-level enforcement) are silently skipped — `verify()` surfaces
        them later.
    """
    from citeproc import Citation as CSLCitation
    from citeproc import CitationItem
    from citeproc.formatter import plain
    from citeproc.frontend import CitationStylesBibliography
    from citeproc.source.json import CiteProcJSON

    # Resolve the cited ids, de-duplicate, keep the positional-1 invariant.
    cited_ids = sorted({c.source_id for c in citations if 1 <= c.source_id <= len(sources)})
    if not cited_ids:
        return []

    # Build the CSL-JSON array. Override each item's `id` with a predictable
    # ``src-{N}`` so we can map citeproc's output back to our source_id after
    # the bibliography sorts them.
    csl_items: list[dict[str, Any]] = []
    for cid in cited_ids:
        src = sources[cid - 1]
        item = dict(src.metadata)
        item["id"] = f"src-{cid}"
        csl_items.append(item)

    style = load_style(style_name)
    csl_source = CiteProcJSON(csl_items)
    bibliography = CitationStylesBibliography(style, csl_source, formatter=plain)

    # Register one CSLCitation per item so citeproc assigns positions / labels.
    registrations: dict[str, CSLCitation] = {}
    for item in csl_items:
        cit = CSLCitation([CitationItem(item["id"])])
        bibliography.register(cit)
        registrations[item["id"]] = cit

    # Render the inline marker per citation before sorting (sort may renumber).
    inline_by_id: dict[str, str] = {}
    for item_id, cit in registrations.items():
        rendered = bibliography.cite(cit, _ignore_warnings)
        inline_by_id[item_id] = str(rendered)

    # Sort and render the full bibliography; bibliography.items is the sorted
    # CitationItem list in the same order as the returned list-of-lists.
    bibliography.sort()
    rendered_entries = bibliography.bibliography()
    rendered_by_id: dict[str, str] = {}
    for item, entry_parts in zip(bibliography.items, rendered_entries, strict=False):
        key = getattr(item, "key", None)
        if key is None:
            # Defensive: citeproc-py sometimes yields plain strings in the
            # rendered list when an item was pruned; skip.
            continue
        rendered_by_id[key] = "".join(str(part) for part in entry_parts)

    # Assemble in original source_id order for stable downstream output.
    references: list[Reference] = []
    for cid in cited_ids:
        item_id = f"src-{cid}"
        references.append(
            Reference(
                source_id=cid,
                inline_marker=inline_by_id.get(item_id, f"[{cid}]"),
                rendered=rendered_by_id.get(item_id, ""),
            )
        )
    return references


def _ignore_warnings(warning: Any) -> None:
    """Swallow citeproc-py warning callbacks.

    citeproc-py's `bibliography.cite(citation, callback)` expects a callable
    that receives warnings (e.g. missing fields). We drop them silently here;
    callers who care can inspect `Reference.rendered` for empty strings that
    indicate a rendering failure.
    """
    del warning
