# CLAUDE.md — citeformer

One-screen brief for Claude Code working in this repo. Canonical wider guide is [`AGENTS.md`](AGENTS.md) (read by Cursor, Codex, Copilot, Aider, Zed, etc.). The living design source is [`docs/reference/`](docs/reference/index.md); the frozen genesis spec is [`docs/spec/v0.md`](docs/spec/v0.md).

## What this is

`citeformer` is a Python OSS library: a bulletproof way to generate verifiably cited text from language models. Citation markers are *structurally impossible to fabricate* at the logit level when using a grammar-level constrained-decoding backend (HF + XGrammar/llguidance, vLLM, llama.cpp) and schema-rejected on the API backends (OpenAI, Gemini, Mistral) + provider-native on Anthropic. Reference lists are rendered deterministically by six hand-written CSL formatters (APA 7, MLA 9, Chicago author-date, IEEE, Nature, Vancouver — see [ADR-004](docs/decisions/004-citeproc-rewrite.md)). The model never touches the bibliography. The library's point is **composition** — we piggyback on XGrammar, llguidance, transformers, vLLM, llama.cpp, lark, httpx, diskcache, grobid, readability, and DeBERTa-v3-MNLI. Full piggyback map + architecture in [`docs/reference/architecture.md`](docs/reference/architecture.md); **read it before writing grammar / rendering / decoding / verification code**.

## Invariants — DO NOT CHANGE SILENTLY

Three §10 contracts (full detail: [`docs/reference/contracts.md`](docs/reference/contracts.md)). Touching any is a deliberate ceremony.

1. **§10.1 — Citation marker grammar.** `cite-id ::= "[" <digits> "]"` (GBNF) + `required` / `quotes_only` / `auto` policies. Lives in `src/citeformer/grammar/`. Changing the marker shape or policy semantics = **major** bump.
2. **§10.2 — `Source.metadata` CSL-JSON shape.** The shape consumed by the home-grown render layer (and still CSL-JSON compliant for external tooling). Additive fields = minor; renames or removals = major. Regression snapshots in `tests/unit/test_render_csl.py` (4 core CSL types × 6 styles) and `tests/unit/test_csl_suite.py` (50-case fixture × 6 formatters = 300 snapshots).
3. **§10.3 — Output schemas.** `GenerationResult` + `VerificationReport` pydantic models carry `schema_version` (currently `3`, pinned by `tests/integration/test_schemas.py`). Any shape change bumps `schema_version`.

Before editing `src/citeformer/grammar/`, `src/citeformer/core.py`, or `src/citeformer/verify/report.py` — run `/contract-check` on your diff.

## Six-layer dependency order

```
CLI → orchestration (Citeformer) → verify → render → backends → grammar → core
```

Upper may import lower; never the reverse. A `render` module must never import from `backends`; a `backend` must never reach into `orchestration`. Break this and refactor radius explodes.

## Piggyback reminders

Before writing new code, ask: is this already done by one of these?

- **XGrammar** / **llguidance** — grammar-level token masking. Don't hand-roll a sampling loop.
- **transformers** / **vLLM** / **llama-cpp-python** — model runtimes.
- **lark** — authoring the grammar before handoff to the decoder.
- **httpx** + **diskcache** — fetchers with caching.
- **grobid** + **readability-lxml** — PDF + URL extraction.
- **DeBERTa-v3-MNLI** — entailment verification.

## Dev commands

```
make dev              # uv sync --all-extras + pre-commit install
make test             # full pytest suite
make test-unit        # unit tests only (fast)
make test-integration # loads real HF models; slow
make lint             # ruff + mypy strict
make format           # ruff format + --fix
make docs             # live Sphinx preview at :5190
make docs-build       # sphinx-build -W (CI mirror)
make release-check    # preflight for tag push
```

## Claude slash-commands (`.claude/commands/`)

- `/bump [patch|minor|major]` — bump `_version.py` + roll CHANGELOG. Stops before commit.
- `/release [patch|minor|major] [--direct]` — full preflight + bump + commit + tag + release-PR (default) or direct push (needs explicit `--direct`).
- `/release-check` — dry-run preflight for a release, zero side effects.
- `/contract-check` — diff-audit against the three §10 contracts.

## Skills (`.claude/skills/`) — always loaded as reminders

- `piggyback-first` — consult the piggyback map before writing new code.
- `contract-invariant` — ceremony when touching §10 files.
- `grammar-shape` — the `cite-id` rule is load-bearing.
- `release-bump` — patch/minor/major rubric.

## Phase status

Current release: **v0.3.1** (PyPI, 2026-04-27; see
`src/citeformer/_version.py` and `CHANGELOG.md`). Work happens on `main`
via short-lived feature branches. Phase breakdown in
[`docs/reference/architecture.md`](docs/reference/architecture.md).

- **P0–P6** — scaffolding through NLI verification. ✅
- **Polish** — REQUIRED progression fix (ADR-009), CLI, examples as
  living reports. ✅
- **Expansion** — marker shapes (ADR-011), backends, metadata adapters
  (DOI / arXiv / PDF via pypdf or GROBID / URL / BibTeX / Zotero),
  threshold calibration, multi-prompt + ALCE benchmarks,
  literature-review notebook, HF Space demo. ✅
- **P7** — v0.1.0 PyPI release (2026-04-24). ✅
- **v0.2.0** — Gemini + Mistral backends, richer PDF extraction, ALCE
  harness. ✅
- **v0.3.0** — async surface end-to-end (ADRs 014–017), OpenRouter +
  Fireworks + Together backends (ten backends total), token-usage on
  results. ✅
- **v0.3.1** — PyPI page link fixes, `CITATION.cff`, codecov upload
  reliability. ✅
- **Next** — scope TBD. Candidates: full-ALCE reproducibility, streaming
  refinements, per-chunk NLI during generation.

## Versioning policy

Patch bumps are cheap — prefer frequent small releases. Full policy in [`docs/development/releasing.md`](docs/development/releasing.md). When finishing a user-visible change, invoke `/bump` or the `release-bump` skill.

## Pre-merge checklist

- `make lint && make test && make docs-build` all green.
- Docstrings on every new public function (ruff D100–D103 enforced on src/).
- PR template "Invariant touched?" section filled.
- CHANGELOG `[Unreleased]` entry added.
- Any new deps folded into the right `pyproject.toml` extra (and into `all` if cross-platform).
