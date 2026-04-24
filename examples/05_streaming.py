"""Streaming demo — print chunks as the model decodes them.

`Citeformer.stream()` returns a `StreamingResult` that is both iterable
(for realtime printing) and finalizable (for the full `GenerationResult`
once the stream completes). The grammar enforcement is identical to
non-streaming `generate()` — structural guarantees apply to every yielded
chunk.

Run:

    uv sync --extra dev --extra hf
    uv run python examples/05_streaming.py

The first run of the default model downloads ~500 MB.
"""

from __future__ import annotations

import os
import sys
import time

from citeformer import Citeformer, Policy, Source, build_rag_prompt


def main() -> None:
    from citeformer.backends.hf import HFBackend

    sources = [
        Source(
            metadata={
                "id": "attention",
                "type": "article-journal",
                "title": "Attention Is All You Need",
                "author": [{"family": "Vaswani"}, {"family": "Shazeer"}],
                "issued": {"date-parts": [[2017]]},
            },
            content=(
                "We propose the Transformer, a network architecture based solely on "
                "attention, dispensing with recurrence and convolutions."
            ),
        ),
        Source(
            metadata={
                "id": "bert",
                "type": "article-journal",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "author": [{"family": "Devlin"}, {"family": "Chang"}],
                "issued": {"date-parts": [[2019]]},
            },
            content=(
                "BERT pretrains deep bidirectional representations from unlabeled "
                "text by jointly conditioning on left and right context in all layers."
            ),
        ),
    ]

    model_name = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Loading model: {model_name}")
    backend = HFBackend(model=model_name, device="cpu")
    cf = Citeformer(backend=backend, style="apa-7", citation_policy=Policy.AUTO)

    prompt = build_rag_prompt(
        query="Write two sentences about self-attention citing at least one source.",
        sources=sources,
        example="Self-attention lets a model weigh relationships across all positions [1].",
    )

    print("\n--- streamed output ---\n")
    stream = cf.stream(prompt=prompt, sources=sources, max_new_tokens=80, temperature=0.7)
    start = time.perf_counter()
    chunk_count = 0
    for chunk in stream:
        chunk_count += 1
        sys.stdout.write(chunk)
        sys.stdout.flush()
    elapsed = time.perf_counter() - start
    print(f"\n\n--- stream complete in {elapsed:.1f}s ({chunk_count} chunks) ---")

    # `.finalize()` is idempotent and builds the full GenerationResult.
    result = stream.finalize()
    print(f"\nParsed citations: {len(result.citations)}")
    for cite in result.citations:
        print(f"  source_id={cite.source_id}  span={cite.span}")
    print("\nRendered references:")
    for ref in result.references:
        print(f"  {ref.inline_marker}  {ref.rendered}")


if __name__ == "__main__":
    main()
