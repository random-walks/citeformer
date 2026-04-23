# Architecture

citeformer is deliberately thin. The hard technical work — token masking, CSL rendering, PDF extraction, NLI — already lives in well-maintained dependencies. Our job is to compose them behind a single honest API.

## Six-layer dependency order

```
CLI → orchestration (Citeformer) → verify → render → backends → grammar → core
```

Upper layers depend only on lower. A render module must never import from backends; a backend must never reach up into orchestration. Break this and the refactor radius explodes.

## Piggyback-first

Before writing new code, ask: is this already done by one of these?

| We piggyback on | For |
| --- | --- |
| **XGrammar** / **llguidance** | Grammar-level token masking at generation time |
| **transformers** (HF) | Running local causal LMs |
| **vLLM** | High-throughput inference with `--guided-decoding-backend` |
| **llama.cpp** (`llama-cpp-python`) | CPU / Apple Silicon inference with GBNF grammars |
| **citeproc-py** + the CSL styles repo | Deterministic reference-list rendering in 10,000+ styles |
| **lark** | Authoring the citation grammar before handing off to the decoder |
| **httpx** + **diskcache** | Metadata fetchers (Crossref, arXiv) with polite caching |
| **grobid-client-python** | PDF metadata + text extraction |
| **readability-lxml** | URL extraction |
| **DeBERTa-v3-MNLI** (via transformers) | NLI entailment for `verify()` |
| **pydantic** + **typer** + **rich** | Types, CLI, pretty output |

The parts citeformer owns are the glue: the citation grammar shape (§10.1), the CSL-JSON source metadata contract (§10.2), the output pydantic models (§10.3), the inline-marker-to-reference coupling, and the orchestration loop. Everything else is a composition.

## Phase plan

v0.1 ships at the end of P6. Each phase is a mergeable milestone with its own exit criterion; see the plan file at `~/.claude/plans/ok-i-setup-this-frolicking-graham.md` (also mirrored to `docs/spec/v0.md`) for the full breakdown.

| Phase | Scope | Exit criterion |
|---|---|---|
| **P0** | Scaffolding: pyproject, CI, docs skeleton, .claude/ | `make lint && make test && make docs-build` green; v0.0.1 publishes to TestPyPI |
| **P1** | Core types: `Source`, `Citation`, `Reference`, `GenerationResult`, `Policy`, `Backend` ABC | Contracts locked; mock backend works end-to-end |
| **P2** | HF backend with grammar-level logit enforcement (the flagship) | Smoke test: given N sources, model cannot emit `[N+k]` for any `k > 0`, across 100+ prompts |
| **P3** | Deterministic CSL reference rendering via citeproc-py | APA-7, MLA-9, Chicago, IEEE, Vancouver render cleanly on the fixture set |
| **P4** | Metadata adapters: DOI, arXiv, PDF, URL | VCR-backed CI tests plus a live smoke script |
| **P5** | vLLM and llama.cpp backends | All three local backends pass the same conformance suite |
| **P6** | NLI verification + v0.1 release | ALCE-style benchmark emits the headline comparison; PyPI live |

After v0.1, API-provider backends (OpenAI, Gemini), Anthropic Citations-API wrapping, and a possible `citeformer-ts` sibling are on the roadmap but not in scope here.
