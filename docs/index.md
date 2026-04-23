# citeformer

**A bulletproof way to generate verifiably cited text from language models.**

citeformer makes citation fabrication *structurally impossible* at the logit level when you use a grammar-level constrained-decoding backend. The model can only emit citation markers that refer to sources you've actually supplied. Reference lists are rendered deterministically by `citeproc-py` in any CSL style — the model never touches the bibliography.

:::{important}
citeformer is pre-1.0 and under active development. v0.1 ships local-model backends only (HF transformers, vLLM, llama.cpp). API-provider backends come in v0.2+. See the [roadmap](reference/architecture.md) for the phase plan.
:::

## Install

```bash
# Minimal — no backends, just the core types.
pip install citeformer

# With the HF transformers backend (P2+).
pip install 'citeformer[hf]'

# Everything cross-platform.
pip install 'citeformer[all]'
```

Python 3.11+ (tested up to 3.14).

## Get started

- [Getting started](getting-started.md) — quickstart once P2 lands.
- [Guarantees](guarantees.md) — what "bulletproof" actually means, tier by tier.
- [Reference](reference/index.md) — architecture, the three §10 contracts, releasing.
- [Historical spec](spec/v0.md) — the frozen genesis plan this was built from.

## Why citeformer exists

LLM-generated citations are wrong 14–95% of the time depending on the benchmark; RAG systems still fabricate 3–13% of cited URLs; NeurIPS 2025 accepted ~50 papers with AI-generated fake references. Prompting doesn't fix it. Post-hoc verification doesn't fix it. The only real fix is **structural**: make the invalid output token-impossible before the model even reaches that decision point.

That's the jsonformer insight, applied to citation attribution in RAG. citeformer wraps modern constrained-decoding libraries (XGrammar, llguidance) plus the 20-year-old CSL ecosystem (citeproc-py + 10,000 community styles) into a single API where the guarantee is honest: **given a supported local backend, a citation marker pointing at a non-existent source is a logit-level impossibility, not a hope.**

```{toctree}
:hidden:
:caption: User guide

getting-started
guarantees
verification
```

```{toctree}
:hidden:
:caption: Reference

reference/index
reference/architecture
reference/contracts
reference/api
```

```{toctree}
:hidden:
:caption: Development

development/dev-setup
development/releasing
```

```{toctree}
:hidden:
:caption: Architecture decisions

decisions/index
```

```{toctree}
:hidden:
:caption: History

spec/v0
```
