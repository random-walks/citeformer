# Changelog

All notable changes to citeformer follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format, and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versioning policy: **patch bumps are cheap**. See [docs/development/releasing.md](docs/development/releasing.md) for the full policy and the three §10 contracts that govern major bumps.

## [Unreleased]

### Added — P0 scaffolding

- Repo infrastructure matching the `random-walks` house style (cf. `jellycell`): `src/citeformer/` package layout, `hatchling` + `uv` packaging, Sphinx + furo + myst-parser + autodoc2 + sphinx-llms-txt docs, GitHub Actions CI + release workflows with PyPI OIDC trusted publishing, pre-commit hooks, Makefile with `dev/test/lint/format/docs/release-check` targets.
- `.claude/` setup: `CLAUDE.md`, `AGENTS.md`, agents (`contract-reviewer`), commands (`/bump`, `/release`, `/release-check`, `/contract-check`), skills (`piggyback-first`, `contract-invariant`, `grammar-shape`, `release-bump`), launch configurations, and local settings allowlist.
- Documentation skeleton: [index](docs/index.md), [guarantees](docs/guarantees.md), [architecture](docs/reference/architecture.md), [contracts](docs/reference/contracts.md), [releasing](docs/development/releasing.md), frozen [v0 spec](docs/spec/v0.md).
- Python 3.11 → 3.14 support in classifiers and CI matrix.
- Apache-2.0 license (switched from MIT during P0 to inherit the patent grant; preferred for LLM-adjacent OSS).
- Minimal `citeformer` CLI exposing `--version`.
- Smoke test `tests/unit/test_version.py` locking `citeformer.__version__` wiring.

### Contracts (§10)

- §10.1 grammar shape: not yet implemented (lands in P2).
- §10.2 CSL metadata: not yet implemented (lands in P3).
- §10.3 output schemas: not yet implemented (lands in P1).

[Unreleased]: https://github.com/random-walks/citeformer/compare/HEAD
