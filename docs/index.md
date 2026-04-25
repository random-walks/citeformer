# citeformer

**A bulletproof way to generate verifiably cited text from language models.**

citeformer makes citation fabrication *structurally impossible* at the logit level when you use a grammar-level constrained-decoding backend. The model can only emit citation markers that refer to sources you've actually supplied. Reference lists are rendered deterministically by home-grown formatters (APA 7, MLA 9, Chicago author-date, IEEE, Nature, Vancouver) — the model never touches the bibliography.

:::{important}
citeformer is pre-1.0 and under active development. Ten backends ship today:
three local (HF transformers, vLLM, llama.cpp) that mask in-process; **Fireworks**
(native GBNF — citeformer's `cite-id` grammar drops in unchanged); and six
strict-structured-output API backends (OpenAI, Anthropic, OpenRouter, Together,
Gemini, Mistral) — strict structured outputs is now real token-level
constrained sampling on every modern provider. See the
[architecture doc](reference/architecture.md#tiered-enforcement--where-the-masking-runs)
for the per-provider breakdown.
:::

## Install

```bash
# Minimal — no backends, just the core types.
pip install citeformer

# With the HF transformers backend (local, logit-layer enforcement).
pip install 'citeformer[hf]'

# With an API backend.
pip install 'citeformer[openai]'    # or: anthropic, openrouter, fireworks, together, gemini, mistral

# Everything cross-platform.
pip install 'citeformer[all]'
```

Python 3.11+ (tested up to 3.14).

## Get started

- [Getting started](getting-started.md) — install, quickstart, verification.
- [Guarantees](guarantees.md) — what "bulletproof" actually means, tier by tier.
- [Verification](verification.md) — how NLI-based `verify()` works.
- [Reference](reference/index.md) — architecture, the three §10 contracts, releasing.
- [Historical spec](spec/v0.md) — the frozen genesis plan this was built from.

## Why citeformer exists

LLM-generated citations are wrong 14–95% of the time depending on the benchmark; RAG systems still fabricate 3–13% of cited URLs; NeurIPS 2025 accepted ~50 papers with AI-generated fake references. Prompting doesn't fix it. Post-hoc verification doesn't fix it. The only real fix is **structural**: make the invalid output token-impossible before the model even reaches that decision point.

That's the jsonformer insight, applied to citation attribution in RAG. citeformer wraps modern constrained-decoding libraries (XGrammar, llguidance) plus six hand-written CSL formatters (APA 7, MLA 9, Chicago author-date, IEEE, Nature, Vancouver — see [ADR-004](decisions/004-citeproc-rewrite.md) for why we rolled our own) into a single API where the guarantee is honest: **given a supported local backend, a citation marker pointing at a non-existent source is a logit-level impossibility, not a hope.**

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
