"""LlamaIndex RAG pipeline wired through citeformer.

LlamaIndex retrievers return ``List[NodeWithScore]``. Feed them through
``sources_from_nodes`` to get citeformer ``Source`` objects, then generate
as usual. Citation markers are structurally constrained to the retrieved
set — no fabrication possible.

In a real app, ``nodes`` comes from ``index.as_retriever().retrieve(query)``.
This script hand-builds nodes so it runs without a full vector index.

Run:

    uv pip install llama-index-core
    uv sync --extra dev --extra hf
    uv run python examples/07_llamaindex_rag.py
"""

from __future__ import annotations

import re
import sys


def main() -> None:
    try:
        from llama_index.core.schema import (  # type: ignore[import-not-found]
            NodeWithScore,
            TextNode,
        )
    except ImportError:
        sys.exit(
            "This example requires llama-index-core. Install with:\n"
            "  uv pip install llama-index-core"
        )

    from citeformer import Citeformer, Policy, build_rag_prompt
    from citeformer.backends.hf import HFBackend
    from citeformer.integrations.llamaindex import sources_from_nodes

    # Normally: `nodes = retriever.retrieve(query)` — sorted by score descending.
    retrieved_nodes = [
        NodeWithScore(
            node=TextNode(
                text=(
                    "The Transformer is based solely on attention mechanisms, "
                    "dispensing with recurrence and convolutions entirely."
                ),
                metadata={
                    "title": "Attention Is All You Need",
                    "author": "Vaswani et al.",
                    "year": 2017,
                    "file_path": "/papers/1706.03762.pdf",
                },
            ),
            score=0.94,
        ),
        NodeWithScore(
            node=TextNode(
                text=(
                    "Chain-of-thought prompting significantly improves reasoning "
                    "ability on arithmetic, commonsense, and symbolic reasoning tasks."
                ),
                metadata={
                    "title": "Chain-of-Thought Prompting Elicits Reasoning",
                    "author": "Wei et al.",
                    "year": 2022,
                    "file_path": "/papers/2201.11903.pdf",
                },
            ),
            score=0.82,
        ),
        NodeWithScore(
            node=TextNode(
                text=(
                    "QLoRA applies 4-bit quantization to frozen pretrained weights "
                    "and backpropagates gradients through low-rank adapters, "
                    "enabling finetuning of 65B models on a single GPU."
                ),
                metadata={
                    "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
                    "author": "Dettmers et al.",
                    "year": 2023,
                    "file_path": "/papers/2305.14314.pdf",
                },
            ),
            score=0.71,
        ),
    ]

    sources = sources_from_nodes(retrieved_nodes)

    backend = HFBackend(model="gpt2", device="cpu")
    cf = Citeformer(backend=backend, style="ieee", citation_policy=Policy.REQUIRED)

    prompt = build_rag_prompt(
        query="Summarize three advances in transformer-era NLP.",
        sources=sources,
        system="You are writing a technical abstract. Cite every claim.",
        example="The Transformer introduced attention [1]. CoT improved reasoning [2].",
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
    print("Structural check — every emitted [N] is in [1..3]")
    print("=" * 72)
    emitted = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", result.text)]
    if emitted:
        print(f"  ids seen: {sorted(set(emitted))}")
        assert all(1 <= i <= len(sources) for i in emitted)
    else:
        print("  (none this run)")

    print()
    print("=" * 72)
    print("IEEE-formatted references")
    print("=" * 72)
    for ref in result.references:
        print(f"  {ref.rendered}")


if __name__ == "__main__":
    main()
