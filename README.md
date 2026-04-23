# citeformer

[![PyPI](https://img.shields.io/pypi/v/citeformer?color=blue)](https://pypi.org/project/citeformer/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://citeformer.readthedocs.io)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/citeformer)](https://pypi.org/project/citeformer/)
[![CI](https://github.com/random-walks/citeformer/actions/workflows/ci.yml/badge.svg)](https://github.com/random-walks/citeformer/actions/workflows/ci.yml)

***A bulletproof way to generate verifiably cited text from language models.***

> **Status**: pre-1.0, under active development. v0.1 ships local-backend support only (HF transformers, vLLM, llama.cpp). API-provider backends come in v0.2+. Follow [CHANGELOG.md](CHANGELOG.md) for progress.

## Why citeformer

LLM-generated citations are wrong 14–95% of the time depending on the benchmark. RAG systems still fabricate 3–13% of cited URLs. NeurIPS 2025 accepted ~50 papers with AI-generated fake references. Prompting doesn't fix it; post-hoc verification doesn't fix it. The only real fix is **structural** — make the invalid output token-impossible before the model reaches the decision point.

That's the [jsonformer](https://github.com/1rgs/jsonformer) insight applied to citations. citeformer wraps modern constrained-decoding libraries ([XGrammar](https://github.com/mlc-ai/xgrammar), [llguidance](https://github.com/guidance-ai/llguidance)) plus the 20-year-old CSL ecosystem ([citeproc-py](https://github.com/citeproc-py/citeproc-py) + 10,000 community styles) into a single API. The guarantee is honest and tiered:

- **Local backends** (HF / vLLM / llama.cpp + XGrammar/llguidance): citation markers are **structurally impossible to fabricate** at the logit level.
- **API-provider backends** (future v0.2+): schema-level enforcement via OpenAI / Gemini structured outputs with enum-constrained cite IDs.
- **Anthropic users** should use [Anthropic's Citations API](https://platform.claude.com/docs/en/build-with-claude/citations) directly — it already solves the problem on Claude; citeformer adds no value there.

## Install

```bash
# Core only — no backend yet, just the types.
pip install citeformer

# With the HF transformers backend (P2+).
pip install 'citeformer[hf]'

# Everything cross-platform (excludes vLLM, which is Linux/CUDA-only).
pip install 'citeformer[all]'
```

Python 3.11+ (tested through 3.14).

## Quickstart (target API — lands with P2)

```python
from citeformer import Citeformer, Source

sources = [
    Source.from_doi("10.1038/s41586-023-06221-2"),
    Source.from_arxiv("2305.14627"),
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

cf = Citeformer(
    backend="hf",
    model="microsoft/Phi-3.5-mini-instruct",
    style="apa-7",
    citation_policy="required",
)
result = cf.generate(
    prompt="Summarize the three works.",
    sources=sources,
)

print(result.text)          # "Poe's Raven opens... [3]"
for ref in result.references:
    print(ref.rendered)     # rendered by citeproc-py, not by the LLM

report = result.verify()    # NLI-based entailment per citation
print(report.support_rate)
```

`result.text` cannot contain `[4]`, because there are only three sources. Not "is unlikely to"; literally cannot, by grammar construction.

## Composition, not reinvention

citeformer's value is the **composition**, not the parts. The heavy lifting lives in established dependencies:

| We piggyback on | For |
|---|---|
| **XGrammar** / **llguidance** | Token-level logit masking at generation time |
| **transformers** / **vLLM** / **llama-cpp-python** | Running the model |
| **citeproc-py** + [CSL styles repo](https://github.com/citation-style-language/styles) | Reference-list rendering in 10,000+ styles |
| **lark** | Authoring the citation grammar before handing it off |
| **httpx** + **diskcache** | Metadata fetchers with caching |
| **grobid-client-python** | PDF extraction |
| **readability-lxml** | URL extraction |
| **DeBERTa-v3-MNLI** via transformers | NLI entailment for `verify()` |

The bits citeformer owns: citation grammar shape, CSL-JSON source contract, output pydantic models, marker-to-reference coupling, and the orchestration loop. Everything else is a compose.

## Is this for you?

**Probably yes if:**

- You're building RAG and need citations that can't hallucinate.
- You run open-weight models locally (HF / vLLM / llama.cpp) and want grammar-level guarantees.
- You need APA / MLA / Chicago / IEEE / Vancouver or any other CSL-compatible output style.
- You care about claim-level NLI verification out of the box.

**Probably no if:**

- You only target Anthropic — use their [Citations API](https://platform.claude.com/docs/en/build-with-claude/citations).
- You want a full agent framework — use LangChain / LlamaIndex and compose citeformer as a generation step.
- You need a TypeScript surface today — a sibling `citeformer-ts` may come later; not here yet.

## Documentation

- **Getting started**: [getting-started](https://citeformer.readthedocs.io/en/stable/getting-started.html) (P2+)
- **Guarantees**: [guarantees](https://citeformer.readthedocs.io/en/stable/guarantees.html) — what "bulletproof" actually covers.
- **Architecture**: [reference/architecture](https://citeformer.readthedocs.io/en/stable/reference/architecture.html) — layers + phase plan.
- **Contracts**: [reference/contracts](https://citeformer.readthedocs.io/en/stable/reference/contracts.html) — the three §10 invariants.
- **Historical spec**: [spec/v0](https://citeformer.readthedocs.io/en/stable/spec/v0.html) — frozen genesis.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: bug-fix PRs welcome and bump patch; feature PRs should open an issue first. The three §10 contracts (grammar shape, CSL metadata, output schemas) are deliberate ceremonies — read [docs/reference/contracts.md](docs/reference/contracts.md) before touching them.

## License

Apache-2.0. See [LICENSE](LICENSE).
