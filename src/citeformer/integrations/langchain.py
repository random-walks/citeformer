"""LangChain ↔ citeformer adapter.

LangChain's retrieval story produces ``List[Document]`` — each with
``page_content`` (the chunk text) and ``metadata`` (free-form dict). To
feed those into ``Citeformer.generate`` we need to convert each
``Document`` to a ``Source`` with CSL-JSON-shaped metadata.

Duck-typed: we don't import LangChain at module load, so you can use
these functions with anything that has ``page_content: str`` +
``metadata: dict`` attributes — LangChain's `Document`, a mock, a
pydantic model, whatever.

Typical usage::

    from citeformer import Citeformer
    from citeformer.integrations.langchain import sources_from_documents
    from citeformer.backends.hf import HFBackend

    docs = retriever.get_relevant_documents(query)   # LangChain retriever
    sources = sources_from_documents(docs)

    cf = Citeformer(backend=HFBackend("gpt2"))
    result = cf.generate(prompt=query, sources=sources)

If your retrieved documents have rich metadata (a Zotero library, a
Crossref-backed vectorstore), pass ``metadata_converter=`` to map from
your custom shape to CSL-JSON. The default converter produces a
minimal-but-valid CSL item (``{id, type: 'webpage', title}``) from
whatever is in ``Document.metadata``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from citeformer.core import Source

MetadataConverter = Callable[[dict[str, Any]], dict[str, Any]]


class _DocumentLike(Protocol):
    """Duck-typed shape — LangChain `Document` matches, so do mocks and
    pydantic models that expose the same two attributes.
    """

    page_content: str
    metadata: dict[str, Any]


def default_metadata_converter(metadata: dict[str, Any]) -> dict[str, Any]:
    """Fallback conversion from LangChain-style metadata to CSL-JSON.

    Pulls common keys the LangChain ecosystem uses (`title`, `source`,
    `url`, `author`) and packages them as a minimal CSL-JSON
    ``{id, type: 'webpage', title, URL?}`` item. Unknown keys are kept
    under `_langchain_metadata` so downstream code can still access them
    if needed.
    """
    title = str(
        metadata.get("title")
        or metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("url")
        or "Untitled"
    )
    ident = str(metadata.get("id") or metadata.get("source") or metadata.get("url") or title)

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

    # Stash anything extra under a private key so callers can still see it.
    extras = {
        k: v
        for k, v in metadata.items()
        if k
        not in {"title", "source", "file_name", "url", "id", "author", "authors", "year", "date"}
    }
    if extras:
        csl["_langchain_metadata"] = extras

    return csl


def source_from_document(
    document: _DocumentLike,
    *,
    metadata_converter: MetadataConverter | None = None,
) -> Source:
    """Convert one LangChain-shaped ``Document`` into a citeformer ``Source``.

    Args:
        document: Object with ``page_content: str`` + ``metadata: dict``
            attributes. LangChain's ``langchain_core.documents.Document``
            is the canonical shape; any duck-typed equivalent works.
        metadata_converter: Optional override for the default CSL-JSON
            conversion. Signature: ``(dict) -> dict``. Useful when your
            retrieved documents come from a rich source (Zotero,
            Crossref-backed vectorstore) and you want to preserve that.

    Returns:
        A `Source` with ``content = document.page_content`` and
        ``metadata`` shaped as CSL-JSON.

    Raises:
        TypeError: If ``document`` doesn't have the expected attributes.
    """
    try:
        content = str(document.page_content)
        raw_meta = dict(document.metadata or {})
    except AttributeError as e:
        raise TypeError(
            "Expected an object with `page_content: str` and `metadata: dict` attributes "
            f"(got {type(document).__name__!r})."
        ) from e

    converter = metadata_converter or default_metadata_converter
    return Source(metadata=converter(raw_meta), content=content)


def sources_from_documents(
    documents: Iterable[_DocumentLike],
    *,
    metadata_converter: MetadataConverter | None = None,
) -> list[Source]:
    """Convert an iterable of LangChain documents to citeformer sources.

    Preserves order; downstream citation ids correspond 1:1 with list
    position, which matches how LangChain retrievers return their
    relevance-ordered results.
    """
    return [source_from_document(doc, metadata_converter=metadata_converter) for doc in documents]


def _normalize_authors(raw: list[Any]) -> list[dict[str, str]]:
    """Make a best-effort list of CSL-JSON author records from varied input."""
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            # Pass through if it already looks CSL-JSON.
            if "family" in item or "literal" in item:
                out.append({k: str(v) for k, v in item.items()})
                continue
            # Heuristic: {"name": "..."} or {"first": "...", "last": "..."}
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
    "source_from_document",
    "sources_from_documents",
]
