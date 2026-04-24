# Contributing to citeformer

Thanks for wanting to help. citeformer is pre-1.0 and under active development — issues and PRs are both welcome.

## Local setup

```bash
git clone https://github.com/random-walks/citeformer
cd citeformer
make dev
```

`make dev` runs `uv sync --all-extras` and installs pre-commit hooks. You'll need [uv](https://docs.astral.sh/uv/) installed (`brew install uv` on macOS).

## Dev loop

```bash
make test             # full pytest
make test-unit        # fast tests only
make lint             # ruff + mypy
make format           # apply fixes
make docs             # live-reload Sphinx at http://127.0.0.1:5190
```

## Hard rules

- **Piggyback first.** Before writing new code, check the piggyback map in [`docs/reference/architecture.md`](docs/reference/architecture.md). The hard work — token masking, CSL rendering, PDF extraction, NLI — already lives in well-maintained deps. We compose; we don't reimplement.
- **Three §10 contracts.** [Grammar shape](docs/reference/contracts.md#101--citation-marker-grammar), [CSL metadata](docs/reference/contracts.md#102--sourcemetadata-shape), [output schemas](docs/reference/contracts.md#103--output-schemas). Each has a ceremony (regenerate snapshots, bump `schema_version`, CHANGELOG note, PR template). Don't touch them silently.
- **Layer discipline.** The six-layer dependency order in [`docs/reference/architecture.md`](docs/reference/architecture.md) is upper-imports-lower. `render/` must never import from `backends/`. PRs that invert the graph get rejected.
- **The model never touches the reference list.** Reference rendering is deterministic via the six hand-written formatters in [`src/citeformer/render/formatters/`](src/citeformer/render/formatters/) (see [ADR-004](docs/decisions/004-citeproc-rewrite.md) for the citeproc-py rewrite history). If you find yourself prompting the model to emit a reference list, stop.

## Commit style

Conventional-commits-ish (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`). Readability matters more than strict linting. Tell the reader *why* in the commit body if the diff alone doesn't explain it.

## PR flow

1. Branch from `main` (feature branches fine; `feat/name`, `fix/issue-N`, etc.).
2. Make your changes. Add tests. Update docs if public behavior changed.
3. Update `CHANGELOG.md` under `[Unreleased]`. If a §10 contract was touched, add a `Contracts (§10)` subsection.
4. `make lint && make test && make docs-build` all green.
5. Open a PR. Fill the "Invariant touched?" section in the template (defaults to "None").
6. Respond to review. Squash or rebase at your discretion before merge.

## Release cadence

Patch bumps are cheap. Target 1–2 releases per week while the repo is active. Minor releases land at phase boundaries (P1, P2, …) or when a coherent feature batch is ready. Major releases only come from §10 contract breaks — rare by construction.

See [`docs/development/releasing.md`](docs/development/releasing.md) for the full policy.

## Reporting issues

Open a GitHub issue with:

- citeformer version (`uv run citeformer version`)
- Python version, OS
- Minimal reproducer (Python script ≤ 20 lines if possible)
- Expected vs. actual behavior

For security issues, follow the disclosure process in [SECURITY.md](SECURITY.md) — don't open a public issue.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind. Assume good faith. If someone's being uncivil, email <blaise@ubik.studio>; we'll handle it.
