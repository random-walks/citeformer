# Changelog

All notable changes to citeformer follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format, and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versioning policy: **patch bumps are cheap**. See [docs/development/releasing.md](docs/development/releasing.md) for the full policy and the three §10 contracts that govern major bumps.

## [Unreleased]

### Added — P6 NLI-based verification + AI-paper benchmark

The final v0.1 phase. `GenerationResult.verify()` is no longer a stub —
real NLI-backed entailment + coverage checks land now. New
`citeformer.verify` module: sentence splitter + existence check (no ML)
plus NLI-powered entailment and coverage. `GenerationResult` and
`VerificationReport` both bumped to schema_version=2 (see
[ADR-008](docs/decisions/008-generation-result-schema-v2.md)).
Benchmark harness at `benchmarks/demo.py` runs on six real AI papers
(Attention Is All You Need, BERT, GPT-3, Chain-of-Thought, LLaMA,
QLoRA) and compares constrained vs. baseline generation on a shared
model instance. Known limitation in REQUIRED policy on small models
documented in [ADR-007](docs/decisions/007-required-policy-progression-gap.md).

### Changed — render layer rewritten from citeproc-py to home-grown formatters

Replaced the citeproc-py-based `render_references` with six hand-written
`CitationFormatter` subclasses. See
[docs/decisions/004-citeproc-rewrite.md](docs/decisions/004-citeproc-rewrite.md)
for the full rationale (Chicago page-range crash, APA double-period, noisy
warnings, no canonical Vancouver in the upstream CSL bundle).

Home-grown formatters landed at `src/citeformer/render/formatters/`:
- `APAFormatter` (APA 7) — author-date.
- `MLAFormatter` (MLA 9) — author + page.
- `ChicagoAuthorDateFormatter` — author-date, no comma.
- `IEEEFormatter` — numeric, bracketed.
- `NatureFormatter` — numeric, bare.
- `VancouverFormatter` — numeric, bracketed; previously unavailable.

Shared helpers in `_base.py`: `Author`, `parse_authors()`, `parse_year()`,
`ensure_period()`, `format_page_range()`, `format_doi()`, `get_str()`,
`get_title()`. The `CitationFormatter` ABC exposes `inline()` and
`bibliography()`; each style handles `article-journal`, `book`, `chapter`,
`thesis`, `paper-conference`, `webpage`, and `report` types with sensible
fallbacks.

Dependency changes:
- `citeproc-py` removed from main dependencies.
- New optional `citeproc-compat` extra (holds `citeproc-py>=0.9`). No code
  uses it yet; reserved for a future compat wrapper that lets users plug
  arbitrary `.csl` files back in.
- Bundled `apa.csl`, `modern-language-association.csl`,
  `chicago-author-date.csl`, `ieee.csl`, `nature.csl` files deleted — no
  longer consulted by the render path.

Public API changes:
- `bundled_style_names()` now returns 6 styles (adds `"vancouver"`).
- `style_citation_format()` keeps the same shape; values come from the
  formatter's `citation_format` class attribute.
- `load_style()` and `resolve_style_path()` are GONE — they were
  citeproc-py-specific. Callers who need a formatter object use
  `get_formatter(name)` (new).

Tests:
- 60 new tests in `tests/unit/test_formatters.py` — parametrised across
  all six styles × four CSL types, plus edge cases (missing author,
  missing year, literal names, hyphenated given names, unknown CSL types,
  per-style et-al. thresholds).
- 6 snapshot tests in `tests/unit/test_render_csl.py` — the pre-existing
  snapshots were regenerated with the new home-grown output; Vancouver
  snapshot added.
- `tests/unit/test_render_styles.py` rewritten around `get_formatter`.

New skill at `.claude/skills/add-citation-format/SKILL.md` documents the
template for adding a seventh style: research, copy-formatter, register,
test matrix, document.

ADR status:
- [ADR-003](docs/decisions/003-bundle-five-csl-styles.md): superseded —
  the bundled set grew to six and the CSL files were removed.
- [ADR-004](docs/decisions/004-citeproc-rewrite.md): Proposed → Accepted
  and implemented.
- [ADR-005](docs/decisions/005-metadata-deps-in-main-install.md): updated
  to note that citeproc-py was removed after the rewrite.

§10.2 (`Source.metadata` as CSL-JSON) and §10.3 (output schemas) both
unchanged. Pure implementation-layer swap from the users' perspective.

### Added — P5 vLLM and llama.cpp backends

Two more local backends joining HFBackend, both consuming the same GBNF
grammar that the §10.1 contract pins. Citation fabrication is now
structurally impossible on all three local inference paths.

- `citeformer.backends.llamacpp.LlamaCppBackend`: wraps ``llama_cpp.Llama``;
  compiles the GBNF via ``LlamaGrammar.from_string`` and passes it to
  ``Llama.__call__(grammar=...)``. CPU, Metal, and CUDA all supported via
  `n_gpu_layers` (auto-detected by llama.cpp). Takes a local GGUF file
  path — users need ``pip install citeformer[llamacpp]`` plus a GGUF.
- `citeformer.backends.vllm.VLLMBackend`: wraps ``vllm.LLM``; compiles the
  GBNF via ``GuidedDecodingParams(grammar=..., backend="xgrammar")``.
  XGrammar chosen as default backend since it matches HFBackend's choice —
  users running the same grammar through both get identical decode-time
  semantics. Linux/CUDA only; excluded from the ``all`` extra.

Integration tests (7 new, total 11):
- ``test_llamacpp_backend.py``: auto-downloads Qwen 2.5 0.5B Instruct Q4_K_M
  GGUF (~370 MB, cached under the HF hub cache), loads the backend, runs
  the "cannot fabricate citations" assertions under REQUIRED and AUTO
  policies, plus a grammar-compiles smoke test that only needs
  ``llama_cpp`` installed (no model load).
- ``test_vllm_backend.py``: same assertions via vLLM; auto-skipped on
  non-Linux or non-CUDA hosts. Uses ``Qwen/Qwen2.5-0.5B-Instruct`` (HF
  weights) with ``enforce_eager=True`` and a small GPU memory utilization
  budget so it runs on a wide variety of CUDA hardware.

Locally on Apple Silicon: 9 integration tests pass (5 HF + 4 llama.cpp);
2 vLLM tests skipped. Llama.cpp + HF + xgrammar all confirm the same
§10.1 guarantee.

Makefile:
- ``make test-integration`` now runs ``uv sync --extra dev --extra hf
  --extra llamacpp`` before pytest so both Apple-friendly local backends
  are exercised. ``--extra vllm`` is opt-in per invocation (Linux/CUDA
  hosts can add it themselves).

### Added — P4 metadata adapters (Source.from_doi / from_arxiv / from_pdf / from_url)

Four classmethods on `Source` for building Source instances from common
identifier + content sources. Each wraps a standalone fetcher function that
can also be imported directly from `citeformer.metadata`.

- `Source.from_doi(doi)` → `citeformer.metadata.fetch_crossref`. Hits
  `api.crossref.org/works/{doi}/transform` with the CSL-JSON Accept
  header; returns the CSL-JSON dict. Honors the
  `CITEFORMER_CROSSREF_MAILTO` env var for Crossref's polite pool.
- `Source.from_arxiv(arxiv_id)` → `citeformer.metadata.fetch_arxiv`. Hits
  arXiv's Atom export API and parses the XML into CSL-JSON
  (`type: article-journal`, `container-title: "arXiv preprint"`, plus an
  `abstract` field that `from_arxiv` pops into `Source.content`).
- `Source.from_pdf(path)` → `citeformer.metadata.extract_pdf`. Uses
  `pypdf` for PDF-info dict metadata (`/Title`, `/Author`,
  `/CreationDate` → year) and per-page text extraction.
  `type: "report"` by default.
- `Source.from_url(url)` → `citeformer.metadata.extract_url`. Uses
  `readability-lxml` for the article body and `lxml` meta-tag parsing
  (OpenGraph / Twitter / article:*) for title / author / date / site.

Metadata caching via diskcache. Default path:
`~/.cache/citeformer/metadata/`. Override with `CITEFORMER_CACHE_DIR`.
Fetchers accept `use_cache=False` for cache-bypass.

Tests (16 new, total 79 unit):
- `test_metadata_cache.py`: env-var override, default path, lazy
  directory creation.
- `test_metadata_pdf.py`: generates a minimal PDF at test-time via
  `PdfWriter`, asserts title / author / year recovery. Plus the
  no-metadata fallback and the missing-file error path.
- `test_metadata_fetchers.py`: pytest-vcr cassettes replay real
  Crossref / arXiv / example.com responses. Stable identifiers
  (Nature paper DOI `10.1038/s41586-023-06221-2`, arXiv `2305.14627`).

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
