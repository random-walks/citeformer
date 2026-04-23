"""Tests for `citeformer.integrations.langchain` + `.llamaindex`.

Duck-typed: we build simple namespace objects with the same attribute
shape as LangChain's `Document` and LlamaIndex's `TextNode` / `NodeWithScore`.
Running these tests doesn't require LangChain or LlamaIndex installed —
the adapters never import the real libraries at module load.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from citeformer.integrations.langchain import (
    default_metadata_converter as lc_default,
)
from citeformer.integrations.langchain import (
    source_from_document,
    sources_from_documents,
)
from citeformer.integrations.llamaindex import (
    default_metadata_converter as li_default,
)
from citeformer.integrations.llamaindex import (
    source_from_node,
    sources_from_nodes,
)

# --- LangChain ----------------------------------------------------------------


def _doc(content: str, **metadata) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    return SimpleNamespace(page_content=content, metadata=dict(metadata))


def test_source_from_document_populates_content_and_metadata() -> None:
    doc = _doc("The Transformer introduced self-attention.", title="Attention Is All You Need")
    source = source_from_document(doc)
    assert source.content == "The Transformer introduced self-attention."
    assert source.metadata["title"] == "Attention Is All You Need"
    assert source.metadata["type"] == "webpage"
    assert "id" in source.metadata


def test_source_from_document_picks_url_as_fallback_title() -> None:
    doc = _doc("body", url="https://arxiv.org/abs/1706.03762")
    source = source_from_document(doc)
    assert source.metadata["URL"] == "https://arxiv.org/abs/1706.03762"
    assert source.metadata["title"] == "https://arxiv.org/abs/1706.03762"


def test_source_from_document_handles_string_author() -> None:
    doc = _doc("body", title="x", author="Vaswani et al.")
    source = source_from_document(doc)
    assert source.metadata["author"] == [{"literal": "Vaswani et al."}]


def test_source_from_document_handles_structured_author_list() -> None:
    doc = _doc(
        "body",
        title="x",
        author=[
            {"family": "Vaswani", "given": "Ashish"},
            {"last": "Shazeer", "first": "Noam"},
            {"name": "OpenAI"},
            "Alan Turing",
        ],
    )
    source = source_from_document(doc)
    authors = source.metadata["author"]
    assert authors == [
        {"family": "Vaswani", "given": "Ashish"},
        {"family": "Shazeer", "given": "Noam"},
        {"literal": "OpenAI"},
        {"literal": "Alan Turing"},
    ]


def test_source_from_document_parses_year_string() -> None:
    doc = _doc("body", title="x", date="2017-06-12")
    source = source_from_document(doc)
    assert source.metadata["issued"] == {"date-parts": [[2017]]}


def test_source_from_document_extras_preserved_under_private_key() -> None:
    doc = _doc("body", title="x", chunk_index=5, relevance=0.82)
    source = source_from_document(doc)
    assert source.metadata["_langchain_metadata"] == {
        "chunk_index": 5,
        "relevance": 0.82,
    }


def test_sources_from_documents_preserves_order() -> None:
    docs = [_doc(f"body {i}", title=f"T{i}") for i in range(5)]
    sources = sources_from_documents(docs)
    assert len(sources) == 5
    for i, source in enumerate(sources):
        assert source.content == f"body {i}"
        assert source.metadata["title"] == f"T{i}"


def test_source_from_document_custom_converter() -> None:
    doc = _doc("body", title="x", extra="custom")

    def converter(meta: dict) -> dict:  # type: ignore[type-arg]
        return {"id": "manual", "type": "book", "title": meta["title"].upper()}

    source = source_from_document(doc, metadata_converter=converter)
    assert source.metadata == {"id": "manual", "type": "book", "title": "X"}


def test_source_from_document_raises_on_missing_attrs() -> None:
    with pytest.raises(TypeError, match="page_content"):
        source_from_document(SimpleNamespace(content="wrong", meta={}))  # type: ignore[arg-type]


def test_lc_default_converter_falls_back_to_untitled_when_empty() -> None:
    csl = lc_default({})
    assert csl["title"] == "Untitled"
    assert csl["type"] == "webpage"


# --- LlamaIndex ---------------------------------------------------------------


def _text_node(text: str, **metadata) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    return SimpleNamespace(text=text, metadata=dict(metadata))


def _node_with_score(text: str, score: float, **metadata) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    return SimpleNamespace(node=_text_node(text, **metadata), score=score)


def test_source_from_text_node_basic() -> None:
    node = _text_node("chunk text", title="My Doc", file_name="my-doc.txt")
    source = source_from_node(node)
    assert source.content == "chunk text"
    assert source.metadata["title"] == "My Doc"


def test_source_from_node_with_score_unwraps_inner_node() -> None:
    nws = _node_with_score("chunk text", 0.91, document_title="Wrapped Doc")
    source = source_from_node(nws)
    assert source.content == "chunk text"
    assert source.metadata["title"] == "Wrapped Doc"


def test_llamaindex_preserves_file_path_as_id() -> None:
    node = _text_node("body", file_path="/home/u/paper.pdf", title="Paper")
    source = source_from_node(node)
    assert source.metadata["id"] == "/home/u/paper.pdf"


def test_llamaindex_extras_preserved_under_private_key() -> None:
    node = _text_node("body", title="x", page_label="12", start_char_idx=240)
    source = source_from_node(node)
    assert source.metadata["_llamaindex_metadata"] == {
        "page_label": "12",
        "start_char_idx": 240,
    }


def test_sources_from_nodes_mixed_types() -> None:
    nodes = [
        _text_node("a", title="A"),
        _node_with_score("b", 0.9, title="B"),
        _text_node("c", title="C"),
    ]
    sources = sources_from_nodes(nodes)
    assert [s.content for s in sources] == ["a", "b", "c"]
    assert [s.metadata["title"] for s in sources] == ["A", "B", "C"]


def test_source_from_node_raises_on_missing_attrs() -> None:
    with pytest.raises(TypeError, match="text"):
        source_from_node(SimpleNamespace(body="wrong"))  # type: ignore[arg-type]


def test_li_default_converter_recognises_url_type() -> None:
    csl = li_default({"url": "https://arxiv.org/abs/1234.5678"})
    assert csl["URL"] == "https://arxiv.org/abs/1234.5678"


def test_li_handles_integer_year() -> None:
    node = _text_node("body", title="x", year=2017)
    source = source_from_node(node)
    assert source.metadata["issued"] == {"date-parts": [[2017]]}


# --- End-to-end: convert → generate -------------------------------------------


def test_langchain_to_citeformer_end_to_end() -> None:
    """Round-trip: LangChain docs → citeformer sources → MockBackend generate."""
    from citeformer import Citeformer, Policy
    from citeformer.backends import MockBackend

    docs = [
        _doc("transformer content", title="Attention", author="Vaswani et al."),
        _doc("bert content", title="BERT", author=[{"family": "Devlin"}]),
    ]
    sources = sources_from_documents(docs)

    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    result = cf.generate(prompt="summarize", sources=sources)
    # MockBackend emits [1], so we should get one citation mapping to source 0.
    assert result.citations
    assert 1 <= result.citations[0].source_id <= 2


def test_llamaindex_to_citeformer_end_to_end() -> None:
    """Round-trip: LlamaIndex nodes → citeformer sources → MockBackend generate."""
    from citeformer import Citeformer, Policy
    from citeformer.backends import MockBackend

    nodes = [
        _node_with_score("body", 0.95, title="A", author="Smith"),
        _node_with_score("body", 0.7, title="B"),
    ]
    sources = sources_from_nodes(nodes)

    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    result = cf.generate(prompt="q", sources=sources)
    assert len(result.references) >= 1
