---
name: piggyback-first
description: Before writing new code in citeformer, consult the piggyback map. The hard work — token masking, CSL rendering, PDF extraction, NLI — already lives in well-maintained deps. We compose; we don't reinvent.
---

# Piggyback-first

Before writing new code in citeformer, ask: **is this already done by one of the libraries we depend on?**

The piggyback map lives in `docs/reference/architecture.md`. The short version:

| We piggyback on | For |
|---|---|
| **XGrammar** / **llguidance** | Token-level logit masking |
| **transformers** / **vLLM** / **llama-cpp-python** | Running the model |
| **citeproc-py** + CSL styles repo | Reference-list rendering |
| **lark** | Authoring grammars before handoff |
| **httpx** + **diskcache** | Metadata fetchers with caching |
| **grobid-client-python** | PDF extraction |
| **readability-lxml** | URL extraction |
| **DeBERTa-v3-MNLI** via transformers | Entailment verification |

## What citeformer actually owns

- The **citation grammar shape** (§10.1 contract) — `CITE_ID` terminal + policy rules.
- The **CSL-JSON source metadata contract** (§10.2).
- The **output pydantic models** and their schema versions (§10.3).
- The **inline-marker-to-reference coupling** (no reference without a matching marker; no marker without a reference).
- The **orchestration loop** — fetching metadata, building the grammar, handing to the backend, decoding output, rendering references, verifying.

Everything else should compose something else.

## Red flags

If you catch yourself doing any of these, stop and check whether a dep already solves it:

- Writing a token-by-token sampling loop. (Use XGrammar / llguidance.)
- Parsing CSL style XML. (Use citeproc-py.)
- Reimplementing PDF layout extraction. (Use GROBID.)
- Rolling a BibTeX parser. (Use pybtex or citeproc-py's bib support.)
- Computing entailment from scratch. (Use DeBERTa-v3-MNLI.)

When the answer is genuinely "no existing dep does this" — document why in a code comment, with the dep you considered and the gap you found.
