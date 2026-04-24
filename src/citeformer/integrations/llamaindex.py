"""LlamaIndex ↔ citeformer adapter.

LlamaIndex retrievers return ``List[NodeWithScore]`` (each wraps a
``TextNode`` with ``text`` + ``metadata`` attributes and a relevance
``score``). To feed those into ``Citeformer.generate`` we convert each
to a ``Source`` with CSL-JSON-shaped metadata.

Duck-typed: we don't import LlamaIndex at module load. Any object with a
``text: str`` attribute and a ``metadata: dict`` attribute works —
whether it's ``llama_index.core.schema.TextNode``, a ``NodeWithScore``
(the adapter unwraps ``.node`` transparently), a pydantic model, or a
plain namespace.

Typical usage::

    from citeformer import Citeformer
    from citeformer.integrations.llamaindex import sources_from_nodes

    nodes = index.as_retriever().retrieve(query)
    sources = sources_from_nodes(nodes)

    cf = Citeformer(backend=...)
    result = cf.generate(prompt=query, sources=sources)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from citeformer.core import Source

MetadataConverter = Callable[[dict[str, Any]], dict[str, Any]]


class _TextNodeLike(Protocol):
    """Duck-typed shape — matches LlamaIndex `TextNode`, `Document`, mocks."""

    text: str
    metadata: dict[str, Any]


class _NodeWithScoreLike(Protocol):
    """Duck-typed shape for LlamaIndex `NodeWithScore`."""

    node: _TextNodeLike
    score: float | None


def default_metadata_converter(metadata: dict[str, Any]) -> dict[str, Any]:
    """Fallback conversion from LlamaIndex-style metadata to CSL-JSON.

    Pulls common keys LlamaIndex loaders use (``title``, ``file_name``,
    ``url``, ``page_label``, ``document_title``) and packages them as a
    CSL-JSON ``{id, type: 'webpage', title}`` item. Loaders that set
    richer structured metadata (e.g. the SimpleDirectoryReader's
    ``file_path``) get stashed under ``_llamaindex_metadata`` so
    callers keep visibility.
    """
    title = str(
        metadata.get("title")
        or metadata.get("document_title")
        or metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("url")
        or "Untitled"
    )
    ident = str(
        metadata.get("id")
        or metadata.get("doc_id")
        or metadata.get("file_path")
        or metadata.get("source")
        or metadata.get("url")
        or title
    )

    csl: dict[str, Any] = {"id": ident, "type": "webpage", "title": title}

    url = metadata.get("url") or metadata.get("source")
    if isinstance(url, str) and url.lower().startswith(("http://", "https://")):
        csl["URL"] = url

    author = metadata.get("author") or metadata.get("authors")
    if isinstance(author, list):
        csl["author"] = _normalize_authors(author)
    elif isinstance(author, str) and author.strip():
        csl["author"] = [{"literal": author.strip()}]

    year = metadata.get("year") or metadata.get("date")
    if isinstance(year, int):
        csl["issued"] = {"date-parts": [[year]]}
    elif isinstance(year, str) and year[:4].isdigit():
        csl["issued"] = {"date-parts": [[int(year[:4])]]}

    extras = {
        k: v
        for k, v in metadata.items()
        if k
        not in {
            "title",
            "document_title",
            "file_name",
            "source",
            "url",
            "id",
            "doc_id",
            "file_path",
            "author",
            "authors",
            "year",
            "date",
        }
    }
    if extras:
        csl["_llamaindex_metadata"] = extras

    return csl


def source_from_node(
    node: _TextNodeLike | _NodeWithScoreLike,
    *,
    metadata_converter: MetadataConverter | None = None,
) -> Source:
    """Convert one LlamaIndex-shaped node into a citeformer ``Source``.

    Accepts either a bare ``TextNode``-like object (has ``text`` +
    ``metadata``) or a ``NodeWithScore``-like wrapper (has ``.node`` with
    the above attributes). The adapter unwraps the latter automatically,
    so callers don't have to reach into ``.node`` themselves.

    Args:
        node: The LlamaIndex node or node-with-score to convert.
        metadata_converter: Optional override for CSL-JSON conversion.

    Returns:
        A `Source` with content from ``node.text`` and CSL-JSON metadata.

    Raises:
        TypeError: If the object doesn't have the expected attributes.
    """
    inner: Any = node
    if hasattr(node, "node") and not hasattr(node, "text"):
        # `NodeWithScore` style — unwrap.
        inner = node.node

    try:
        content = str(inner.text)
        raw_meta = dict(inner.metadata or {})
    except AttributeError as e:
        raise TypeError(
            "Expected an object with `text: str` + `metadata: dict` attributes "
            "(or a `NodeWithScore`-like wrapper with `.node` exposing them); "
            f"got {type(node).__name__!r}."
        ) from e

    converter = metadata_converter or default_metadata_converter
    return Source(metadata=converter(raw_meta), content=content)


def sources_from_nodes(
    nodes: Iterable[_TextNodeLike | _NodeWithScoreLike],
    *,
    metadata_converter: MetadataConverter | None = None,
) -> list[Source]:
    """Convert an iterable of LlamaIndex nodes to citeformer sources.

    Preserves order; LlamaIndex retrievers return relevance-sorted
    results, and the citation-id assigned by citeformer (1-indexed
    position) mirrors that order.
    """
    return [source_from_node(n, metadata_converter=metadata_converter) for n in nodes]


def _normalize_authors(raw: list[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            if "family" in item or "literal" in item:
                out.append({k: str(v) for k, v in item.items()})
                continue
            if "last" in item and "first" in item:
                out.append({"family": str(item["last"]), "given": str(item["first"])})
                continue
            if "name" in item:
                out.append({"literal": str(item["name"])})
                continue
        if isinstance(item, str) and item.strip():
            out.append({"literal": item.strip()})
    return out


__all__ = [
    "MetadataConverter",
    "default_metadata_converter",
    "source_from_node",
    "sources_from_nodes",
]
