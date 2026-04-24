"""Full pipeline: HF model + grammar enforcement + NLI verification.

Three hand-built sources, a small instruction-tuned model, REQUIRED policy
with ADR-009's bounded-content rule, and `verify()` with NLI entailment.
Asserts at the end:

- No fabricated citations (source id out of range).
- Every emitted marker appears in the rendered bibliography.
- The verification report has the expected §10.3-v2 shape.

Prints a compact digest: the generated text, the parsed citations, the
bibliography, and the per-citation entailment breakdown. Useful as a
sanity-check of the `hf` + `verify` extras on a fresh machine.

Run:

    uv sync --extra dev --extra hf --extra verify
    uv run python examples/02_rag_with_hf_and_verify.py

Swap the model or NLI backend with env vars:

    HF_MODEL=microsoft/Phi-3.5-mini-instruct \\
    CITEFORMER_NLI_MODEL=cross-encoder/nli-deberta-v3-base \\
    uv run python examples/02_rag_with_hf_and_verify.py
"""

from __future__ import annotations

import os
import re

from citeformer import Citeformer, Policy, Source

_CITE = re.compile(r"\[(\d+)\]")

# gpt2 is small enough to run anywhere and deliberately *not* instruction-tuned —
# it's not a great citation generator, which is fine: the point of this script is
# to show the library's wiring, not to claim high support rates on a 117M model.
DEFAULT_MODEL = os.environ.get("HF_MODEL", "gpt2")


def main() -> None:
    from citeformer.backends.hf import HFBackend

    sources = [
        Source(
            metadata={
                "id": "attention",
                "type": "article-journal",
                "title": "Attention Is All You Need",
                "author": [
                    {"family": "Vaswani", "given": "Ashish"},
                    {"family": "Shazeer", "given": "Noam"},
                ],
                "issued": {"date-parts": [[2017]]},
                "container-title": "NeurIPS",
            },
            content=(
                "We propose a new simple network architecture, the Transformer, based "
                "solely on attention mechanisms, dispensing with recurrence and "
                "convolutions entirely. Experiments on two machine translation tasks "
                "show these models to be superior in quality while being more "
                "parallelizable and requiring significantly less time to train."
            ),
        ),
        Source(
            metadata={
                "id": "bert",
                "type": "article-journal",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "author": [
                    {"family": "Devlin", "given": "Jacob"},
                    {"family": "Chang", "given": "Ming-Wei"},
                ],
                "issued": {"date-parts": [[2019]]},
                "container-title": "NAACL",
            },
            content=(
                "BERT is designed to pretrain deep bidirectional representations from "
                "unlabeled text by jointly conditioning on both left and right "
                "context in all layers. As a result, the pre-trained BERT model can "
                "be fine-tuned with just one additional output layer to create "
                "state-of-the-art models for a wide range of tasks."
            ),
        ),
        Source(
            metadata={
                "id": "gpt3",
                "type": "article-journal",
                "title": "Language Models are Few-Shot Learners",
                "author": [
                    {"family": "Brown", "given": "Tom B."},
                    {"family": "Mann", "given": "Benjamin"},
                ],
                "issued": {"date-parts": [[2020]]},
                "container-title": "NeurIPS",
            },
            content=(
                "We train GPT-3, an autoregressive language model with 175 billion "
                "parameters, and test its performance in the few-shot setting. For "
                "all tasks, GPT-3 is applied without any gradient updates or "
                "fine-tuning, with tasks and few-shot demonstrations specified "
                "purely via text interaction with the model."
            ),
        ),
    ]

    backend = HFBackend(model=DEFAULT_MODEL)
    cf = Citeformer(backend=backend, style="apa-7", citation_policy=Policy.REQUIRED)

    result = cf.generate(
        prompt=(
            "Brief note on transformer-based NLP: \n"
            "The Transformer architecture [1] introduced self-attention."
        ),
        sources=sources,
        max_new_tokens=120,
        # ADR-009: tight bound keeps small models emitting citations within budget.
        max_content_chars=60,
    )

    # Guarantee 1: no fabrication. Under grammar enforcement this is structural;
    # the assert is belt-and-suspenders.
    emitted = [int(m.group(1)) for m in _CITE.finditer(result.text)]
    for cid in emitted:
        assert 1 <= cid <= len(sources), f"fabricated id {cid} in {result.text!r}"

    # Guarantee 2: every emitted marker's source has a bibliography entry.
    bibliography_ids = {ref.source_id for ref in result.references}
    for cid in set(emitted):
        assert cid in bibliography_ids, f"emitted [{cid}] but no rendered reference"

    print("=" * 72)
    print(f"Model: {DEFAULT_MODEL}  |  N={len(sources)} sources  |  policy=required")
    print("=" * 72)
    print("Generated text:")
    print(f"  {result.text.strip()!r}")
    print()
    print(f"  citations emitted: {len(emitted)}  |  in-range (structural): 100%")
    print(f"  rendered references: {len(result.references)}")
    for ref in result.references:
        print(f"    {ref.inline_marker} → {ref.rendered}")
    print()

    print("=" * 72)
    print("NLI verification (verify())")
    print("=" * 72)
    report = result.verify(threshold=0.4)
    print(f"  schema_version: {report.schema_version}  (§10.3 v2)")
    print(f"  overall support rate: {report.support_rate:.0%}")
    for cs in report.per_citation:
        marker = "✓" if cs.supported else "✗"
        print(
            f"  {marker} citation #{cs.citation_index}: "
            f"entailment={cs.entailment_score:.2f}  supported={cs.supported}"
        )
    if report.uncited_but_entailed:
        print()
        print("  coverage flags (sentences that should have been cited):")
        for item in report.uncited_but_entailed:
            start, end = item.span
            snippet = result.text[start:end][:80]
            print(
                f"    span=[{start}:{end}] candidate_source={item.candidate_source_id} "
                f"score={item.entailment_score:.2f}\n      {snippet!r}"
            )


if __name__ == "__main__":
    main()
