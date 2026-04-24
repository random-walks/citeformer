# Getting started

## Install

```bash
# Core install — Source types, render layer, metadata fetchers.
pip install citeformer

# With the HuggingFace transformers backend (local grammar-enforced decoding).
pip install 'citeformer[hf]'

# With NLI-based verify().
pip install 'citeformer[verify]'

# Everything cross-platform.
pip install 'citeformer[all]'
```

Python 3.11+ (tested through 3.14).

## Full quickstart

```python
from citeformer import Citeformer, Policy, Source
from citeformer.backends.hf import HFBackend

# 1. Gather sources — via DOI, arXiv, PDF, URL, or hand-constructed.
sources = [
    Source.from_arxiv("1706.03762"),                # Attention Is All You Need
    Source.from_arxiv("1810.04805"),                # BERT
    Source(
        metadata={
            "id": "poe-raven",
            "type": "book",
            "title": "The Raven",
            "author": [{"family": "Poe", "given": "Edgar Allan"}],
            "issued": {"date-parts": [[1845]]},
        },
        content="Once upon a midnight dreary...",
    ),
]

# 2. Instantiate a Citeformer with a grammar-enforced backend.
backend = HFBackend(model="Qwen/Qwen2.5-0.5B-Instruct")
cf = Citeformer(
    backend=backend,
    style="apa-7",
    citation_policy=Policy.REQUIRED,  # every sentence gets a citation — see ADR-009.
)

# 3. Generate. Fabricating [4] is structurally impossible (only 3 sources).
result = cf.generate(
    prompt="Write a short paragraph about transformer-based LMs, citing [N] markers.",
    sources=sources,
    max_new_tokens=120,
)

print(result.text)
# → "Transformers introduced self-attention [1]. BERT extended this with bidirectional pre-training [2]..."

# 4. Rendered references — via home-grown CSL formatter, never by the model.
for ref in result.references:
    print(ref.rendered)
# → "Vaswani, A. et al. (2017). Attention Is All You Need. arXiv preprint."
# → ...

# 5. NLI-verify every citation.
report = result.verify(threshold=0.5)
print(f"support rate: {report.support_rate:.0%}")
for cs in report.per_citation:
    print(f"  cite #{cs.citation_index}: entailment={cs.entailment_score:.2f}")
```

## Key properties to remember

- **No fabrications**: ``[4]`` cannot appear in ``result.text`` when there are 3 sources.
  The grammar masks out-of-range tokens at every decode step — see
  [architecture](reference/architecture.md) for the §10.1 contract.
- **References are home-grown**: citeformer ships six CSL formatters (APA, MLA, Chicago
  author-date, IEEE, Nature, Vancouver). The LLM never sees / writes the bibliography.
- **Verify is opt-in**: ``result.verify()`` runs NLI per citation; not free. Skip it for
  pure generation throughput.

## See also

- [Guarantees](guarantees.md) — what's structurally enforced vs. what's post-hoc checked.
- [Verification](verification.md) — how NLI-based verify() works and its limitations.
- [Architecture](reference/architecture.md) — the six-phase design and layer order.
- [Runnable examples](https://github.com/random-walks/citeformer/tree/main/examples) —
  nine scripts + a notebook covering the quickstart, full HF + verify() pipeline, standalone
  rendering, the metadata-fetch path, streaming, LangChain / LlamaIndex integration,
  BibTeX + Zotero ingest, and an end-to-end literature-review notebook. Each doubles as a living report.
- [Benchmarks](https://github.com/random-walks/citeformer/tree/main/benchmarks) —
  the AI-papers RAG comparison with actual numbers from recent runs.
