# CLAUDE.md — citeformer

One-screen brief for Claude Code working in this repo. Canonical wider guide is [`AGENTS.md`](AGENTS.md) (read by Cursor, Codex, Copilot, Aider, Zed, etc.). The living design source is [`docs/reference/`](docs/reference/index.md); the frozen genesis spec is [`docs/spec/v0.md`](docs/spec/v0.md).

## What this is

`citeformer` is a Python OSS library: a bulletproof way to generate verifiably cited text from language models. Citation markers are *structurally impossible to fabricate* at the logit level when using a grammar-level constrained-decoding backend (HF + XGrammar/llguidance, vLLM, llama.cpp). Reference lists are rendered deterministically by six hand-written CSL formatters (APA 7, MLA 9, Chicago author-date, IEEE, Nature, Vancouver — see [ADR-004](docs/decisions/004-citeproc-rewrite.md)). The model never touches the bibliography. The library's point is **composition** — we piggyback on XGrammar, llguidance, transformers, vLLM, llama.cpp, lark, httpx, diskcache, grobid, readability, and DeBERTa-v3-MNLI. Full piggyback map + architecture in [`docs/reference/architecture.md`](docs/reference/architecture.md); **read it before writing grammar / rendering / decoding / verification code**.

## Invariants — DO NOT CHANGE SILENTLY

Three §10 contracts (full detail: [`docs/reference/contracts.md`](docs/reference/contracts.md)). Touching any is a deliberate ceremony.

1. **§10.1 — Citation marker grammar.** `cite-id ::= "[" <digits> "]"` (GBNF) + `required` / `quotes_only` / `auto` policies. Lives in `src/citeformer/grammar/`. Changing the marker shape or policy semantics = **major** bump.
2. **§10.2 — `Source.metadata` CSL-JSON shape.** The shape consumed by the home-grown render layer (and still CSL-JSON compliant for external tooling). Additive fields = minor; renames or removals = major. Regression snapshot in `tests/unit/test_csl_rendering.py`.
3. **§10.3 — Output schemas.** `GenerationResult` + `VerificationReport` pydantic models carry `schema_version: 1`. Any shape change bumps `schema_version`.

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

v0.1 is feature-complete on the `feat/p0-scaffolding` branch; we're in a
tighten-and-polish pass before cutting the tag. Phase breakdown in
[`docs/reference/architecture.md`](docs/reference/architecture.md).

- **P0** — scaffolding. ✅
- **P1** — core types + contracts locked. ✅
- **P2** — HF backend with grammar-level enforcement (flagship). ✅
- **P3** — deterministic CSL rendering (home-grown; ADR-004). ✅
- **P4** — metadata adapters (DOI, arXiv, PDF, URL, BibTeX, Zotero). ✅
- **P5** — vLLM + llama.cpp backends. ✅
- **P6** — NLI verification + AI-papers benchmark. ✅
- **Polish** — REQUIRED progression fix (ADR-009), CLI, examples as
  living reports. ✅
- **Expansion** — marker shapes (ADR-011), OpenAI + Anthropic API
  backends (schema-tier + native), threshold calibration, multi-prompt
  benchmark, literature-review notebook, HF Space demo. ✅
- **P7** — v0.1 PyPI release. *← next*

## Versioning policy

Patch bumps are cheap — prefer frequent small releases. Full policy in [`docs/development/releasing.md`](docs/development/releasing.md). When finishing a user-visible change, invoke `/bump` or the `release-bump` skill.

## Pre-merge checklist

- `make lint && make test && make docs-build` all green.
- Docstrings on every new public function (ruff D100–D103 enforced on src/).
- PR template "Invariant touched?" section filled.
- CHANGELOG `[Unreleased]` entry added.
- Any new deps folded into the right `pyproject.toml` extra (and into `all` if cross-platform).
