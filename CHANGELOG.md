# Changelog

All notable changes to citeformer follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format, and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versioning policy: **patch bumps are cheap**. See [docs/development/releasing.md](docs/development/releasing.md) for the full policy and the three §10 contracts that govern major bumps.

## [Unreleased]

### Added — P2b HF backend with logit-level citation enforcement

The flagship demo: load any HuggingFace causal LM, pass sources, and citation
fabrication becomes a logit-level impossibility. Five integration tests verify
the guarantee on a real (tiny) model — ``gpt2`` via MPS/CPU.

- `citeformer.backends.hf.HFBackend`: wraps ``transformers.AutoModelForCausalLM``
  + ``xgrammar.GrammarCompiler``, builds the §10.1 grammar per ``generate()``
  call, constructs a fresh ``xgrammar.contrib.hf.LogitsProcessor`` (mandatory —
  xgrammar's LogitsProcessor is stateful per-generation), and runs constrained
  decoding. Supports CUDA / MPS / CPU auto-detection, ``bf16`` / ``fp16`` /
  ``fp32`` / ``auto`` dtype selection.
- Requires the ``hf`` extra: ``pip install citeformer[hf]``. Lazy imports in
  the constructor keep the base install slim; only users who instantiate
  ``HFBackend`` pay the torch / transformers / xgrammar cost.
- Five ``@pytest.mark.integration`` tests in
  ``tests/integration/test_hf_backend.py``:
  ``grammar_compiles`` (syntax check against the real tokenizer),
  ``cannot_fabricate_citations_required_policy`` (the flagship assertion —
  no ``[N+k]`` after generation with N sources),
  ``cannot_fabricate_citations_auto_policy`` (same guarantee under AUTO),
  ``rejects_empty_sources``,
  ``compiler_caches_across_calls`` (cache-hit behavior on repeated grammars).

### Changed — grammar format: Lark → GBNF

- ``Grammar.ebnf`` → ``Grammar.gbnf``; ``build_grammar()`` now emits GBNF
  (``rule ::= production``) directly instead of Lark syntax. xgrammar requires
  GBNF (``::=`` rather than ``:``), and llama.cpp's native grammar format is
  also GBNF — emitting it directly means no Lark→GBNF translator in the hot
  path.
- ``parse_ok()`` removed. It was a Lark-based convenience check; with GBNF in
  place, the authoritative validator is xgrammar's own ``compile_grammar()``,
  exercised at integration time. Keeping a parallel Lark path would mean
  maintaining two grammar formats in lock-step for little benefit.
- The `_SHARED_TAIL`, per-policy bodies, and ``cite-id`` rule were translated
  to GBNF (rule names use kebab-case per the GBNF convention; ``root`` is the
  entry rule instead of ``start``). Semantics are unchanged. §10.1 snapshots
  regenerated accordingly; CITE_ID terminal references across docs + skills
  updated to the ``cite-id`` rule spelling.

### Added — pytest integration-marker gating

- Default ``pytest`` run now excludes ``@pytest.mark.integration`` tests via
  ``addopts = "-m 'not integration and not gpu and not network'"``. Keeps CI
  fast and hermetic; run ``pytest -m integration`` or
  ``make test-integration`` to exercise the real-model paths. The
  ``test-integration`` Makefile target now runs ``uv sync --extra dev --extra
  hf`` before pytest so the hf deps are available.

### Added — P2a citation grammar builder (§10.1 contract)

- `citeformer.grammar.builder.build_grammar(n_sources, policy) -> Grammar`
  emits a Lark-format EBNF grammar with the §10.1 load-bearing terminal
  `CITE_ID: "[" ("1" | "2" | ... | "N") "]"`. Dynamic enum reflects
  `len(sources)` per generate call.
- Three policy bodies implemented: `REQUIRED` (every sentence must end with
  `cite_group SENT_END`), `AUTO` (`cite_group` optional anywhere), and
  `QUOTES_ONLY` (`cite_group` required after each quoted span).
- `citeformer.grammar.parse_ok(grammar, text) -> bool` — lark round-trip
  check, useful for debugging and post-hoc verification.
- `lark>=1.2` promoted from `hf` extra to main dependencies — it's a core
  runtime dep now, not just an HF-backend concern.
- 17 new tests in `tests/unit/test_grammar_builder.py`: 4 pinned
  `pytest-regressions` snapshots (one per policy plus a scaling check for
  `n_sources=10`), explicit `CITE_ID` terminal assertions, input-validation
  tests (`n_sources >= 1`), and a full semantic matrix — each policy admits
  its expected syntactic shape AND rejects out-of-range `[N+k]` markers.

### Added — P1 core types + mock orchestration

- `citeformer.Citeformer` orchestrator — composes a `Backend` with a CSL style
  and a citation `Policy`, parses `[N]` markers out of backend output into
  `Citation` objects, and assembles a `GenerationResult`. Reference rendering is
  a stub until P3 (citeproc-py integration).
- `citeformer.core`: `Source`, `Citation`, `Reference`, `GenerationResult`,
  `Policy` (`StrEnum` — `required`, `quotes_only`, `auto`). All pydantic
  `model_config = ConfigDict(frozen=True, extra="forbid")`.
- `citeformer.verify.report`: `VerificationReport` and `CitationSupport` schema
  locked here even though `verify()` itself lands in P6 — that way the §10.3
  contract snapshot has something to pin from day one.
- `citeformer.backends.Backend` ABC + `MockBackend` (scripted responses for
  tests, deterministic `[1]`-emitting fallback when not scripted).
- Tests: 32 new unit + integration tests covering type validation (frozen,
  extra-forbidden, range checks), mock backend behavior, `Citeformer`
  orchestration (cite parsing, stub reference rendering, policy override), and
  snapshot-pinned §10.3 schemas for `GenerationResult` + `VerificationReport`
  via `pytest-regressions`.

### Changed

- Ruff ignore list now includes `RUF001`/`RUF002`/`RUF003` — em-dashes,
  en-dashes, and typographic quotes in prose docstrings are intentional
  typography, not ambiguous Unicode.

### Contracts (§10)

- §10.1 grammar shape: not yet implemented (lands in P2).
- §10.2 CSL metadata: `Source.metadata` type declared as `dict[str, Any]`
  annotated as a CSL-JSON item in docstrings. Full validation against the CSL
  schema lands with citeproc-py integration in P3.
- §10.3 output schemas: **locked**. `GenerationResult.schema_version == 1` and
  `VerificationReport.schema_version == 1` pinned by snapshots in
  `tests/integration/test_schemas.py`.

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
