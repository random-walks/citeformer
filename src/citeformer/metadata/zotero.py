"""Zotero CSL-JSON library loader.

Zotero's "Export → CSL JSON" option produces an array of CSL-JSON items —
already the shape :class:`~citeformer.core.Source` consumes. This module
provides a thin loader plus a couple of ergonomic niceties:

- De-duplication of items whose ``id`` collides on export (Zotero sometimes
  emits colliding ``itemKey`` values when the same record appears in multiple
  collections).
- Optional filtering by a user-supplied predicate (e.g. "only papers from
  2020 onward", "only AI/ML tags").
- Graceful handling of minor Zotero quirks (``issued.date-parts`` with
  stringified year, stray null fields).

The Better BibTeX plugin's CSL-JSON export is also supported — it's
substantially identical to stock Zotero output. The same loader handles both.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

__all__ = ["load_zotero_csl"]


def load_zotero_csl(
    source: str | Path | Iterable[dict[str, Any]],
    *,
    filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    dedupe: bool = True,
) -> list[dict[str, Any]]:
    """Load a Zotero CSL-JSON export → list of normalised CSL-JSON items.

    Args:
        source: Path to a ``.json`` CSL-JSON export, a raw CSL-JSON string,
            or an iterable of items (lets you compose with in-memory data).
        filter_fn: Optional predicate; items returning ``False`` are dropped.
            Passed each item *after* normalisation so it sees the final
            shape downstream code will see.
        dedupe: If ``True`` (default), items with duplicate ``id`` values
            are merged keeping the first occurrence. Zotero's CSL export
            sometimes emits colliding keys when the same record lives in
            multiple collections.

    Returns:
        Normalised CSL-JSON items in document order.
    """
    raw = _read_raw(source)
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalised = _normalise_zotero_item(item)
        if filter_fn is not None and not filter_fn(normalised):
            continue
        item_id = str(normalised.get("id", ""))
        if dedupe and item_id and item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        out.append(normalised)
    return out


def _read_raw(source: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve the ``source`` argument into a list of raw dicts."""
    if isinstance(source, (str, Path)):
        path = Path(source) if not isinstance(source, Path) else source
        text = path.read_text(encoding="utf-8") if path.is_file() else str(source)
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(
                "Zotero CSL-JSON must be a top-level list; "
                f"got {type(data).__name__}"
            )
        return data
    return list(source)


def _normalise_zotero_item(item: dict[str, Any]) -> dict[str, Any]:
    """Smooth over minor Zotero CSL-JSON quirks."""
    cleaned: dict[str, Any] = {}
    for key, value in item.items():
        if value is None:
            continue
        if key == "issued":
            normalised_date = _normalise_date(value)
            if normalised_date is not None:
                cleaned[key] = normalised_date
            continue
        if key in ("author", "editor", "translator"):
            if isinstance(value, list):
                cleaned[key] = [v for v in value if isinstance(v, dict)]
            continue
        cleaned[key] = value
    return cleaned


def _normalise_date(raw: Any) -> dict[str, Any] | None:
    """Turn assorted date shapes into ``{date-parts: [[year, month?, day?]]}``."""
    if not isinstance(raw, dict):
        return None
    parts = raw.get("date-parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], list):
        first = parts[0]
        try:
            ints = [int(str(p)) for p in first if p not in (None, "")]
        except ValueError:
            return None
        if not ints:
            return None
        return {"date-parts": [ints]}
    # Some exports give "issued": {"literal": "forthcoming"} — pass through.
    if "literal" in raw:
        return {"literal": str(raw["literal"])}
    return None
