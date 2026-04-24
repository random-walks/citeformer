"""Integration tests that exercise the real LangChain / LlamaIndex types.

The duck-typed unit tests in ``tests/unit/test_integrations.py`` cover the
attribute-shape contract using ``SimpleNamespace`` stand-ins — they catch
"we look for ``.text``, got ``.content``" regressions regardless of which
version of LangChain/LlamaIndex the user has.

These tests cover the other side: if LC or LI rename / change the shape of
their canonical types on our pinned version, we notice. They're gated by
``pytest.importorskip`` so the default run still works without the heavy
deps.

Run with::

    uv pip install langchain-core llama-index-core
    uv run pytest tests/integration/test_integrations_real_libs.py -m integration
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# --- LangChain ---------------------------------------------------------------


def test_langchain_document_converts_cleanly() -> None:
    lc_docs = pytest.importorskip("langchain_core.documents")
    from citeformer.integrations.langchain import source_from_document

    doc = lc_docs.Document(
        page_content="The Transformer introduced self-attention.",
        metadata={
            "title": "Attention Is All You Need",
            "author": "Vaswani et al.",
            "url": "https://arxiv.org/abs/1706.03762",
            "year": 2017,
        },
    )
    source = source_from_document(doc)

    assert source.content == "The Transformer introduced self-attention."
    assert source.metadata["title"] == "Attention Is All You Need"
    assert source.metadata["URL"] == "https://arxiv.org/abs/1706.03762"
    assert source.metadata["issued"] == {"date-parts": [[2017]]}
    assert source.metadata["author"] == [{"literal": "Vaswani et al."}]


def test_langchain_sources_from_docs_preserves_order() -> None:
    lc_docs = pytest.importorskip("langchain_core.documents")
    from citeformer.integrations.langchain import sources_from_documents

    docs = [
        lc_docs.Document(page_content=f"body {i}", metadata={"title": f"T{i}"}) for i in range(4)
    ]
    sources = sources_from_documents(docs)
    assert [s.content for s in sources] == [f"body {i}" for i in range(4)]
    assert [s.metadata["title"] for s in sources] == [f"T{i}" for i in range(4)]


def test_langchain_document_end_to_end_with_mock_backend() -> None:
    """Full convert → generate pipeline with real `Document` objects."""
    lc_docs = pytest.importorskip("langchain_core.documents")
    from citeformer import Citeformer, Policy
    from citeformer.backends import MockBackend
    from citeformer.integrations.langchain import sources_from_documents

    docs = [
        lc_docs.Document(page_content=f"Content for source {i}", metadata={"title": f"S{i}"})
        for i in range(1, 4)
    ]
    sources = sources_from_documents(docs)
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    result = cf.generate(prompt="summarize", sources=sources)
    assert result.citations
    assert 1 <= result.citations[0].source_id <= 3


# --- LlamaIndex --------------------------------------------------------------


def test_llamaindex_textnode_converts_cleanly() -> None:
    li_schema = pytest.importorskip("llama_index.core.schema")
    from citeformer.integrations.llamaindex import source_from_node

    node = li_schema.TextNode(
        text="BERT pretrains bidirectional representations.",
        metadata={
            "title": "BERT",
            "file_path": "/papers/bert.pdf",
        },
    )
    source = source_from_node(node)
    assert source.content == "BERT pretrains bidirectional representations."
    assert source.metadata["title"] == "BERT"
    assert source.metadata["id"] == "/papers/bert.pdf"


def test_llamaindex_node_with_score_unwraps() -> None:
    li_schema = pytest.importorskip("llama_index.core.schema")
    from citeformer.integrations.llamaindex import source_from_node

    inner = li_schema.TextNode(
        text="GPT-3 is an autoregressive model.",
        metadata={"title": "GPT-3"},
    )
    nws = li_schema.NodeWithScore(node=inner, score=0.92)
    source = source_from_node(nws)
    assert source.content == "GPT-3 is an autoregressive model."
    assert source.metadata["title"] == "GPT-3"


def test_llamaindex_end_to_end_with_mock_backend() -> None:
    li_schema = pytest.importorskip("llama_index.core.schema")
    from citeformer import Citeformer, Policy
    from citeformer.backends import MockBackend
    from citeformer.integrations.llamaindex import sources_from_nodes

    nodes = [
        li_schema.NodeWithScore(
            node=li_schema.TextNode(text=f"body {i}", metadata={"title": f"T{i}"}),
            score=1.0 - i * 0.1,
        )
        for i in range(3)
    ]
    sources = sources_from_nodes(nodes)
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    result = cf.generate(prompt="summarize", sources=sources)
    assert result.references
