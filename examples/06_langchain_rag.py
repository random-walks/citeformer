"""LangChain RAG pipeline wired through citeformer.

Typical LangChain RAG: retriever returns ``List[Document]``, you stuff them
into a prompt, the LLM writes the answer. The citation step is fragile —
models fabricate sources that don't exist in the retrieval set.

citeformer's LangChain adapter bridges the two: convert retrieved
``Document`` objects to ``Source`` via ``sources_from_documents``, pass
them to ``Citeformer.generate()``, and citation fabrication becomes
structurally impossible.

This example uses hand-built ``Document`` objects to keep the script
self-contained. In a real app, ``docs`` comes from ``retriever.get_relevant_
documents(query)`` (or async equivalent).

Run:

    uv pip install langchain-core
    uv sync --extra dev --extra hf
    uv run python examples/06_langchain_rag.py
"""

from __future__ import annotations

import re
import sys


def main() -> None:
    try:
        from langchain_core.documents import Document  # type: ignore[import-not-found]
    except ImportError:
        sys.exit(
            "This example requires langchain-core. Install with:\n"
            "  uv pip install langchain-core"
        )

    from citeformer import Citeformer, Policy, build_rag_prompt
    from citeformer.backends.hf import HFBackend
    from citeformer.integrations.langchain import sources_from_documents

    # Normally: `docs = retriever.invoke(query)`
    retrieved_docs = [
        Document(
            page_content=(
                "We propose a new simple network architecture, the Transformer, "
                "based solely on attention mechanisms, dispensing with recurrence "
                "and convolutions entirely."
            ),
            metadata={
                "title": "Attention Is All You Need",
                "author": "Vaswani et al.",
                "year": 2017,
                "url": "https://arxiv.org/abs/1706.03762",
            },
        ),
        Document(
            page_content=(
                "BERT is designed to pretrain deep bidirectional representations "
                "from unlabeled text by jointly conditioning on both left and "
                "right context in all layers."
            ),
            metadata={
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "author": "Devlin et al.",
                "year": 2019,
                "url": "https://arxiv.org/abs/1810.04805",
            },
        ),
        Document(
            page_content=(
                "GPT-3 is an autoregressive language model with 175 billion "
                "parameters. Its performance in the few-shot setting is "
                "competitive with fine-tuning approaches on many tasks."
            ),
            metadata={
                "title": "Language Models are Few-Shot Learners",
                "author": "Brown et al.",
                "year": 2020,
                "url": "https://arxiv.org/abs/2005.14165",
            },
        ),
    ]

    # LC Document → citeformer Source. Duck-typed: works with anything that has
    # `.page_content` + `.metadata`. CSL-JSON metadata is derived from the
    # free-form Document.metadata via `default_metadata_converter`.
    sources = sources_from_documents(retrieved_docs)

    backend = HFBackend(model="gpt2", device="cpu")
    cf = Citeformer(backend=backend, style="apa-7", citation_policy=Policy.REQUIRED)

    prompt = build_rag_prompt(
        query="Explain how self-attention enables transformer-based pre-training.",
        sources=sources,
        system="You are a terse academic assistant. Cite every claim.",
        example=(
            "Self-attention lets the model weight relationships across all "
            "positions [1]. BERT extended this with bidirectional pretraining [2]."
        ),
    )

    result = cf.generate(
        prompt=prompt,
        sources=sources,
        max_new_tokens=100,
        max_content_chars=60,
    )

    print("=" * 72)
    print("Generated text")
    print("=" * 72)
    print(result.text[:800])
    print()

    print("=" * 72)
    print(f"Citations (N={len(result.citations)}) — structurally in-range [1..3]")
    print("=" * 72)
    cite_ids = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", result.text)]
    if cite_ids:
        unique = sorted(set(cite_ids))
        print(f"  ids used: {unique}")
        assert all(1 <= i <= 3 for i in unique), "structural guarantee violated"
    else:
        print("  (none emitted this run — try a different seed / larger model)")

    print()
    print("=" * 72)
    print(f"Rendered references (style={cf.style})")
    print("=" * 72)
    for ref in result.references:
        print(f"  {ref.inline_marker}  {ref.rendered}")


if __name__ == "__main__":
    main()
