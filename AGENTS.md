# AGENTS.md — citeformer

Canonical agent guide for this repo. Native readers: Cursor, Codex, GitHub Copilot, Aider, Zed, Warp, Windsurf, Gemini CLI. Claude Code reads `CLAUDE.md`, which delegates here.

## What this repo is

`citeformer` is a Python OSS library: **a bulletproof way to generate verifiably cited text from language models**. Full pitch: [`README.md`](README.md). Full design: [`docs/reference/architecture.md`](docs/reference/architecture.md). Frozen genesis spec: [`docs/spec/v0.md`](docs/spec/v0.md).

v0.1 ships local-backend support only (HF transformers, vLLM, llama.cpp). API-provider backends come in v0.2+.

## Where to start

**For new agents touching the core flow:**
- [`docs/reference/architecture.md`](docs/reference/architecture.md) — six-layer design, piggyback map, phase plan.
- [`docs/reference/contracts.md`](docs/reference/contracts.md) — the three §10 invariants that govern versioning.

**For agents adding a new backend:**
- `src/citeformer/backends/base.py` defines the ABC.
- Existing implementations in `src/citeformer/backends/` are the pattern.
- Add a conformance test to `tests/integration/test_backend_parity.py`.

**For agents adding a new metadata adapter:**
- `src/citeformer/metadata/` — one module per source kind (`doi`, `arxiv`, `pdf`, `url`).
- Entry point: a `Source.from_X` classmethod that returns a `Source` with CSL-JSON metadata + raw content.
- VCR-record the live HTTP calls into `tests/unit/test_metadata/cassettes/` so CI doesn't hit the network.

## Hard rules

- **Piggyback first.** The hard work (token masking, CSL rendering, PDF extraction, NLI) already lives in deps. We compose; we don't reimplement. See the piggyback map in [`docs/reference/architecture.md`](docs/reference/architecture.md).
- **Three §10 contracts.** Grammar shape, CSL metadata shape, output schemas. Touching one = ceremony (regenerate snapshots, bump `schema_version`, CHANGELOG note, PR template). See [`docs/reference/contracts.md`](docs/reference/contracts.md).
- **Layer discipline.** Upper layer may import lower; never the reverse. `render/` must never import from `backends/`. Break this and the refactor radius explodes.
- **The model never touches the reference list.** Reference rendering is *always* deterministic via citeproc-py. If you find yourself prompting the model to output a reference list, stop and ask why.
- **Apache-2.0**, Python ≥ 3.11 (tested through 3.14). CI matrix in `.github/workflows/ci.yml`.
- **Default install = minimum.** Every backend / adapter family is an extra. `pip install citeformer` gets you the core types; backends come via `citeformer[hf]`, `citeformer[vllm]` (Linux-only), `citeformer[llamacpp]`, etc.

## Conventions

- **Imports**: `import citeformer as cf` is fine for scripts. In library code use absolute imports (`from citeformer.core import Source`).
- **Type hints**: pydantic models for runtime / public types (`Source`, `GenerationResult`, `VerificationReport`), `@dataclass(frozen=True)` for internal value types, vanilla hints elsewhere. Strict mypy on `src/`.
- **Docstrings**: Google style, enforced by ruff D100–D103 on public API (src/). Tests and examples are exempt.
- **Tests**: `pytest`, `tests/` sibling to `src/`. Unit tests in `tests/unit/`, integration (loads real models or hits live APIs) in `tests/integration/` — marked `@pytest.mark.integration`.
- **Commits**: conventional-commits-ish (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`). No strict enforcement; readability matters more than a linter.

## Dev quickstart

```bash
git clone https://github.com/random-walks/citeformer
cd citeformer
make dev            # uv sync --all-extras + pre-commit install
make test           # pytest
make lint           # ruff + mypy
make docs           # live-reload Sphinx at :5190
```

Environment variables live in `.env.example`. Copy to `.env` when needed. For P0 none are required.

## The three §10 contracts (short form)

1. **§10.1** — `CITE_ID: "[" <digits> "]"` + the policies (`required`, `quotes_only`, `auto`) in `src/citeformer/grammar/`.
2. **§10.2** — `Source.metadata` is CSL-JSON; shape consumed by citeproc-py.
3. **§10.3** — `GenerationResult` and `VerificationReport` pydantic models with `schema_version: 1`.

Full ceremony in [`docs/reference/contracts.md`](docs/reference/contracts.md). When editing any file listed there, run `/contract-check` (Claude Code) or ask whoever's reviewing to run the audit before merging.

## Release flow

- Patch bumps are cheap. Feature PRs prefer frequent small releases over batching.
- `/bump [patch|minor|major]` → version + CHANGELOG → commit → tag → GitHub Actions publishes to PyPI via OIDC.
- Full policy: [`docs/development/releasing.md`](docs/development/releasing.md).
