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
| **openai** / **anthropic** / **google-genai** / **mistralai** | API-provider generation clients |
| **lark** | Authoring the citation grammar before handing off to the decoder |
| **httpx** + **diskcache** | Metadata fetchers (Crossref, arXiv) with polite caching |
| **pypdf** / **grobid-client-python** | PDF text extraction — pypdf default, GROBID opt-in for cleaner scientific-paper parsing |
| **readability-lxml** | URL extraction |
| **DeBERTa-v3-MNLI** (via transformers) | NLI entailment for `verify()` |
| **pydantic** + **typer** + **rich** | Types, CLI, pretty output |

The parts citeformer owns are the glue plus the render layer: the citation grammar shape (§10.1), the CSL-JSON source metadata contract (§10.2), the output pydantic models (§10.3), the inline-marker-to-reference coupling, the orchestration loop, and the six hand-written CSL formatters (APA 7, MLA 9, Chicago author-date, IEEE, Nature, Vancouver — see [ADR-004](../decisions/004-citeproc-rewrite.md)). Everything else is a composition.

## Phase plan

v0.1.0 shipped on 2026-04-24. Each phase was a mergeable milestone with its own exit criterion; see the frozen genesis at [`docs/spec/v0.md`](../spec/v0.md) for the original plan.

| Phase | Scope | Exit criterion |
|---|---|---|
| **P0** | Scaffolding: pyproject, CI, docs skeleton, .claude/ | `make lint && make test && make docs-build` green; v0.0.1 publishes to TestPyPI |
| **P1** | Core types: `Source`, `Citation`, `Reference`, `GenerationResult`, `Policy`, `Backend` ABC | Contracts locked; mock backend works end-to-end |
| **P2** | HF backend with grammar-level logit enforcement (the flagship) | Smoke test: given N sources, model cannot emit `[N+k]` for any `k > 0`, across 100+ prompts |
| **P3** | Deterministic CSL reference rendering (home-grown, see [ADR-004](../decisions/004-citeproc-rewrite.md)) | APA 7, MLA 9, Chicago author-date, IEEE, Nature, Vancouver render cleanly on the fixture set |
| **P4** | Metadata adapters: DOI, arXiv, PDF, URL, BibTeX, Zotero | VCR-backed CI tests plus a live smoke script |
| **P5** | vLLM and llama.cpp backends | All three local backends pass the same conformance suite |
| **P6** | NLI verification + hand-curated AI-papers benchmark | Coverage report shows support-rate gains; see `benchmarks/README.md` |
| **Polish** | REQUIRED progression fix ([ADR-009](../decisions/009-bounded-content-required.md)), real CLI, examples as living reports | ADR-009 integration test passes; `citeformer` CLI covers generate/verify/render; `examples/` has runnable scripts with findings READMEs |
| **Expansion** | Marker-shape enum ([ADR-011](../decisions/011-configurable-marker-styles.md)), OpenAI + Anthropic + Gemini + Mistral API backends, threshold calibration, multi-prompt + ALCE benchmarks, literature-review notebook, HF Space demo, GROBID PDF extractor | Seven backends pass a shared contract; 40-run multi-prompt sweep reports 0.0 ± 0.0 fabrication; PREPRINT.md describes the v0.1 design + evaluation |
| **P7 (shipped)** | v0.1.0 on PyPI + GitHub Release | `pip install citeformer==0.1.0` works; docs built on RTD; CI green across Python 3.11–3.14 |

Next-up (v0.2 scope TBD): full-ALCE reproducibility (ASQA / QAMPARI / ELI5), per-chunk NLI during generation, streaming refinements on API backends, and a possible `citeformer-ts` sibling if ecosystem demand materialises.

## Tiered enforcement — local vs API

v0.1 ships **three tiers** of backends with **one honest distinction**:

- **Local backends** (`HFBackend`, `VLLMBackend`, `LlamaCppBackend`) enforce at the **logit layer** via XGrammar / llguidance / llama.cpp GBNF. A fabricated cite id is *token-impossible to sample* — the sampler never sees that token distribution.
- **Schema backends** (`OpenAIBackend`, `GeminiBackend`, `MistralBackend`) enforce at the **schema layer**: `strict=true` JSON schema (OpenAI, Mistral) or OpenAPI-subset `response_schema` (Gemini) rejects non-conforming responses server-side. Fabrication is impossible *in the returned payload*, not at the sampler.
- **Provider-native** (`AnthropicBackend`) adapts Anthropic's native Citations API — the provider's own per-block `document_index` references are the guarantee; we translate them into citeformer's `Citation` / `Reference` shape.

All seven backends produce the same `GenerationResult` — the orchestration layer, verify layer, and render layer are backend-agnostic. Users who want the bulletproof structural claim pick a local backend; users who want frontier-model prose without self-hosting pick a schema or provider-native backend and accept that framing.

Tier choice doesn't change the bibliography pipeline: references are always rendered deterministically by our home-grown formatters, regardless of which backend produced the text.
