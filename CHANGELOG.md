# Changelog

All notable changes to citeformer follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format, and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versioning policy: **patch bumps are cheap**. See [docs/development/releasing.md](docs/development/releasing.md) for the full policy and the three §10 contracts that govern major bumps.

## [Unreleased]

## [0.2.0] — 2026-04-24

### Added — two new API backends, richer PDF extraction, ALCE harness

Post-v0.1.0 expansion with no §10 contract changes — additive on every
axis.

**`GeminiBackend`** (extra `gemini`, pulls `google-genai>=0.7`). Schema-
tier enforcement via Gemini's OpenAPI-subset `response_schema` on
`generate_content`. Enum-bounded citation integers mirror the OpenAI
strict-JSON path; fabrication is impossible in the returned payload.
Uses `min_items` (snake_case) and skips `additionalProperties: false`
— both quirks of Gemini's schema dialect. Env: `GEMINI_API_KEY` /
`GOOGLE_API_KEY`.

**`MistralBackend`** (extra `mistral`, pulls `mistralai>=2.0`). Schema-
tier enforcement via Mistral's `response_format={"type": "json_schema",
strict: true}` (Nov-2024 API surface). Reuses the OpenAI schema builder
since both providers accept the same shape. Env: `MISTRAL_API_KEY`.
Note: the extra floor is `>=2.0` because the mistralai 2.x namespace-
package layout (`from mistralai.client import Mistral`) is the only
supported shape; 1.x used a different entry-point name.

**Four new unit-test suites** (42 new tests) covering the new backends
with `SimpleNamespace` fake clients: schema enum bounds, strict=true
flag, source embedding in system prompt, marker-style propagation,
streaming chunking, empty-source rejection.

**Live-API integration tests** (`tests/integration/test_api_backends_live.py`,
marked `integration`, env-gated). Six tests hit OpenAI and Anthropic
production endpoints; two additional tests skip cleanly when
`GEMINI_API_KEY` / `MISTRAL_API_KEY` aren't set. Verifies the structural
invariant (out-of-scope cite ids never appear) end-to-end on the real
providers, not just against fake clients. Default CI without secrets
stays green — tests skip rather than fail when a key is absent.

**GROBID PDF extractor** — `Source.from_pdf(path, extractor="grobid")`
dispatches to a new GROBID-backed path alongside the default pypdf one.
Extra: `grobid` (`grobid-client-python>=0.0.9`). Users stand up the
GROBID Java service separately (`docker run -p 8070:8070
grobid/grobid:0.8.0`). What GROBID buys over pypdf: structured
family/given author lists, extracted abstract field, section-level body
paragraphs, and `type: "article-journal"` (vs pypdf's conservative
`report`). 14 new unit tests with a monkey-patched client stub covering
the happy path, 503 failures, missing-extra ImportError, and the
`Source.from_pdf` forwarding.

**ALCE-flavoured benchmark harness** — new `benchmarks/alce_subset.py`
runs a 3-example toy subset (hand-written, no downloads) or any
JSONL-formatted ALCE file (`--data`) and computes the three headline
metrics: citation recall, citation precision (NLI entailment per cite),
fabrication rate (canary for structural regression). 18 unit tests lock
the metric math. Full-ALCE reproducibility (ASQA / QAMPARI / ELI5)
deferred to v0.3.

**`thread-flow.png` + `thread-multi.png` + refreshed cover.** Five
tweet-friendly cover/thread images at 1200×675 rendered by
`benchmarks/generate_cover.py`: annotated side-by-side adversarial demo
(the cover), 4-stage pipeline flowchart, same-prompt-3-models grid, NLI
claim-to-source verify pipeline, and the model-vs-library bibliography
split. The cover's bottom strip now explicitly names the mechanism
("Not prompted, not retried, not checked after. Prevented at the
decode step.") so the "can't you just prompt the 6 sources?" objection
gets closed at a glance.

**Docs + community polish** — pre-merge audit that killed every stale
"API backends coming in v0.2+" claim across docs (they're already in
v0.1), updated `§10.3` schema_versions to reflect the
`GenerationResult v2` + `VerificationReport v3` bumps, refreshed
snapshot counts (100→300). New community files: `CODE_OF_CONDUCT.md`,
`SECURITY.md`, `.github/ISSUE_TEMPLATE/{bug_report, feature_request,
config}.yml`, `.github/FUNDING.yml`, `AUTHORS.md`. New `PREPRINT.md` —
2000-word paper-shaped design + evaluation write-up.

**Multi-prompt sweep — 3 → 5 seeds.** `DEFAULT_SEEDS` bumped, 40-cell
rerun landed. citeformer fabrication: 0.0 ± 0.0 (still). Baseline
`survey` drift tightened to 3.9% mean (up from 2.4% at 24 runs — more
seeds surfaced more of the long-tail).

**Smaller-NLI calibration finding.** Ran
`threshold_calibration.py --model cross-encoder/nli-deberta-v3-base`
against the 50-triple set. DeBERTa-v3-base caps at F1 0.63 (vs large's
0.96) — perfect precision but under-confident on paraphrases. Finding
4b added to `benchmarks/README.md`.

### Changed — dep floors bumped for CI + new-backend compatibility

- `torch>=2.4` → `torch>=2.8`. Needed for clean cp313 wheel resolution
  on CI; `torch 2.5.1` on py3.13/py3.14 was dragging `triton==3.1.0`
  which has no cp313 wheels. Local resolution on macOS arm64 was fine;
  CI on Linux tripped. Floor raise is narrow (2.4–2.7 was a small
  window; most HF users already on 2.8+).
- `mistralai>=1.0` → `mistralai>=2.0` as the floor for the `mistral`
  extra (new in this release, so no existing users impacted).

### Pipeline

- HF Space deploy automation: `hf-space/deploy.sh` + `make hf-space
  SPACE=<user>/<name>`. Idempotent — reruns push the latest state.

## [0.1.0] — 2026-04-24

### Tier expansion + calibration + flagship artifacts

Three-axis expansion: the public API grows to cover more models, more
evidence, and more shareable artifacts.

**33 property-based fuzz tests** (was 9). New invariants covered: grammar
bounds rejection, formatter hygiene (no leading/trailing whitespace, no
double-space, never-empty, year-present-in-author-date styles),
`render_references` ordering and out-of-range drop, `Citation` nonpositive
source_id rejection, `GenerationResult` frozen + schema_version default,
CSL validator accepts well-typed / errors on missing id/type / warns on
unknown fields, `deduplicate_adjacent_cites` idempotence + collapse, prompt
ordering invariants.

**50-case CSL suite × 6 formatters = 300 locked snapshots.** New
``tests/unit/test_csl_suite.py`` exercises the CSL 1.0 item-type registry
and field-presence matrix more broadly than the previous 4-item canonical
fixture. Covers book / article-journal / chapter / thesis / paper-conference /
report / webpage / software / dataset / patent / map / figure / speech /
interview / legislation / bill / review / review-book / broadcast /
musical_score / motion_picture / personal_communication /
entry-dictionary / entry-encyclopedia / manuscript / post-weblog /
article-newspaper / article-magazine plus edge cases (single-page vs range,
missing author/year, van der / de la / von particles, literal org author,
CJK literal names, Unicode Scandinavian surname, hyphenated given name,
BC years, 8-author et-al threshold, ISBN / ISSN, volume / issue / URL).

**BibTeX + Zotero CSL-JSON ingest adapters.** Two new zero-dep metadata
adapters:

- ``citeformer.metadata.bibtex`` — hand-rolled BibTeX parser mapping
  common entry types + ~20 fields to CSL-JSON. Handles balanced braces,
  `{value}` / `"value"` delimiters, BibTeX `and`-separated names in
  both `Family, Given` and `Given Family` conventions, month abbreviations,
  type → CSL mapping. Unknown fields land under `custom` for lossless
  round-tripping; `@string` / `@preamble` / `@comment` skipped without
  error.
- ``citeformer.metadata.zotero`` — loader for Zotero's native CSL-JSON
  export (and Better BibTeX CSL-JSON, identical schema). Dedupes
  colliding ids, drops null fields, normalises stringified date-parts,
  supports predicate filtering.

Source-level: ``Source.from_bibtex(path_or_str)`` and
``Source.from_zotero(path_or_iterable)`` return `list[Source]` with
empty content. 26 unit tests.

**Configurable marker shapes (ADR-011).** New `MarkerStyle` enum —
`BRACKET` (default, `[N]`), `PAREN` (`(N)`), `CURLY` (`{N}`),
`CARET` (`^N`). Threaded through `build_grammar`, `Citeformer`, all real
backends + MockBackend echo, and the post-hoc citation parser. §10.1
invariant (digit enum bounded by `range(1, n_sources + 1)`) holds
identically across all four shapes — additive minor change. 15 new unit
tests + 4 parametrised grammar-builder variants.

**OpenAI + Anthropic API backends (schema-tier enforcement).**

- `OpenAIBackend` uses OpenAI Structured Outputs with `strict=true` and a
  JSON schema whose `citations` items are `enum`-bounded to the source
  ids. Schema-equivalent of the GBNF `cite-id` digit enum — the provider
  validates, fabrication is impossible in the returned payload. Requires
  `gpt-4o-mini` or any post-August-2024 model.
- `AnthropicBackend` adapts Anthropic's native Citations API: passes
  documents with `citations.enabled=true`, translates per-block citation
  structure into citeformer's `Citation`/`Reference` shape so API output
  mixes with local-backend output in the same pipeline.

Both respect `marker_style`, accept per-call overrides, support streaming
via sentence-sized chunking. New extras: `openai = ["openai>=1.40"]`,
`anthropic = ["anthropic>=0.40"]`. 24 unit tests with
`SimpleNamespace`-based client stand-ins.

**NLI threshold calibration sweep.** New ``benchmarks/threshold_calibration.py``
runs 19 thresholds (0.05-0.95) against 50 hand-labelled (premise,
hypothesis, label) triples grounded in the fixture abstracts. Writes a
two-panel figure (threshold sweep + P/R scatter) and a JSON log.

Finding: DeBERTa-v3-large-mnli is **bimodal** — scores cluster at ~0
and ~1 with only 3 of 50 pairs in the mid-range. Every threshold from
0.20 to 0.95 produces the same confusion matrix (P=0.93 / R=1.00 /
F1=0.96). Default 0.5 sits on the plateau and stays. Users who want
fewer false positives should reach for chunked full-text scoring or
self-consistency — threshold tuning isn't the right knob for this NLI
head.

**Multi-prompt benchmark.** New ``benchmarks/multiprompt_sweep.py`` adds a
prompt-shape axis to the existing model × seed sweep: 4 shapes
(survey / compare / explain / critique) × N seeds × M models. Per-prompt
aggregates with `mean ± std` for fabrication + support rate + citation
density. Produces `findings/multiprompt-summary.png`. Addresses the
"one-prompt caveat" in the benchmarks README head-on.

**Literature-review notebook.** ``examples/08_literature_review.ipynb``
— 7-section end-to-end notebook: fetch 6 arXiv papers on prompt-reasoning,
build the RAG prompt, grammar-constrained generation under REQUIRED
policy, structural-invariant check, NLI verification per citation,
APA-7 bibliography render, side-by-side baseline comparison.

**HuggingFace Space / Gradio demo.** ``hf-space/`` directory: `app.py`
runs the adversarial demo in a clickable Gradio UI; `README.md` includes
the Spaces frontmatter for `git push`-to-deploy. Loads Qwen 0.5B on CPU
(~500 MB). Single model, any seed, reliably shows the 100% → 0%
fabrication swing.

Suite: 449 unit tests + integration suite. Mypy strict + ruff clean
throughout.

### Fit-and-finish: eyebrow-raisers audit

A scrutiny pass addressing specific soft spots flagged in a self-audit
of the library. Everything below is either a correctness improvement,
an honest re-documentation of a noisy measurement, or a small API for
a real gap.

**Chunked NLI premise scoring (opt-in).** DeBERTa-v3's 512-token cap
silently truncates long premises. New opt-in ``NLIModel(chunk_premise=
True, max_premise_tokens=400, chunk_stride=300)`` slides a window over
the premise, scores each window vs. the hypothesis, and reduces by max
entailment. In empirical comparison on Qwen 0.5B fulltext × 5 seeds,
chunked mode moved per-seed support rates significantly — but not
always up. It surfaces claims buried past the 512-token horizon
(seed 3: 11% → 63%) but also inflates false positives on unrelated
claims (max-over-windows gives noise more chances to cross threshold).
Net effect seed-dependent, so **default is off** for score stability;
users opt in with a bumped threshold (0.7+) when they want long-
document scoring. 13 new unit tests cover the chunking math via a
fake tokenizer; integration uses the real DeBERTa.

**Schema v3: `citations_checked` field.** Previously, reports with
zero citations returned ``support_rate = 1.0`` (vacuous entailment).
Tables that averaged support across runs blended those "no data"
entries into the mean — "baseline 100% supported!" really meant
"baseline emitted zero cites." Schema v3
([ADR-010](docs/decisions/010-verification-report-schema-v3.md)) adds
``citations_checked: int`` so consumers can gate on
``citations_checked > 0`` before aggregating. ``support_rate`` itself
unchanged for backward compatibility. Additive / minor per the §10.3
ceremony; snapshot regenerated; schema-version test updated to 3.

**CSL-JSON validation helper (opt-in).** New
``citeformer.csl.validate_csl_json`` + ``validate_source_metadata``
pair. Runs a purely-defensive check over a CSL-JSON item: required
fields (`id`, `type`), type sanity (`DOI` is `str`, `author` is list
of dicts, `issued.date-parts` is a list), known CSL 1.0 item types,
known top-level fields. Errors are hard problems; unknown types /
fields are warnings so forward-compat holds. Users call it before
constructing a `Source` when they want the §10.2 schema policed
up front. 17 unit tests lock the error/warning boundary. Top-level
exports: ``CSLValidationError``, ``ValidationReport``,
``validate_csl_json``, ``validate_source_metadata``, plus
``KNOWN_TYPES`` and ``KNOWN_FIELDS``.

**`deduplicate_adjacent_cites` helper.** REQUIRED-policy grammar
allows ``cite-group ::= cite-id (ws cite-id)*`` — which small models
use to close a sentence by cycling in-scope ids (``[1] [2] [3] [1]
[2] [3] [1]``). The grammar lets this through; users who want clean
output call the new helper to collapse runs to unique-first-appearance
order. Pure-Python regex, no state. 12 tests.

**`py.typed` marker.** Ships PEP 561 compliance. Downstream users of
a pip-installed citeformer now see our inline type annotations in
their IDE / mypy rather than the ``Skipping analyzing 'citeformer':
found module but no type hints`` wall. Included in the hatch wheel
build via ``[tool.hatch.build.targets.wheel.force-include]``.

**LangChain + LlamaIndex real-library integration tests.** Previous
adapter tests used ``SimpleNamespace`` stand-ins — they caught
attribute-shape regressions but wouldn't catch a rename of
``Document.page_content`` or ``TextNode.text``. New
``tests/integration/test_integrations_real_libs.py`` (6 tests,
marked ``integration``) imports the real ``langchain_core.documents``
and ``llama_index.core.schema`` types via ``pytest.importorskip``.
Also: new runnable examples — ``examples/06_langchain_rag.py`` and
``examples/07_llamaindex_rag.py`` — show end-to-end pipeline usage
with real LC/LI types.

**Streaming integration test on Qwen 2.5 0.5B Instruct.** Previously
streaming was only exercised on gpt2 (non-instruct, simple tokenizer).
The new integration test loads an actual instruction-tuned model and
asserts ``stream().finalize().text == generate().text`` at
temperature 0. Catches any tokenizer / chat-template interaction
that gpt2 doesn't trigger.

**Broadened formatter edge-case tests.** 85 new parametrised tests
cover formatter behaviour on unicode family names (Łopez, Zhāng),
hyphenated given names (Jean-Paul), organisational literal authors
(OpenAI), single-word names (Madonna), very long titles (~200 char
arXiv titles), missing years (working papers), page-range dashes,
DOI-as-URL rendering for APA, "van der" particles, titles with
colons and embedded quotes, given-only authors, and empty author
lists. Caught and fixed a couple of real edge cases as part of the
expansion.

**Makefile coverage target.** ``make coverage`` runs the unit suite
with pytest-cov, writes HTML to ``htmlcov/`` and JSON to
``coverage.json``. Current coverage: 81% across 44 source files.

**Honest benchmarks README rewrite.** Previously reported "Qwen 0.5B:
90.9 ± 3.8%" support rate from a 2-seed full-text run. Adding 3 more
seeds collapsed the mean to 46.6 ± 40.7 — std same order of magnitude
as the mean, which is the honest picture for small-model NLI scoring.
Rewrote the README to lead with fabrication rate (0 ± 0, stable
across all 13 sweep runs) and treat support rate as directional.
Documents the chunked-NLI tradeoff, the pypdf/GROBID extraction
difference, the ``citations_checked`` rationale, and the 100%-when-
zero-cites trap.

**plot.py improvements.** Fabrication panels now label bars ``n/a``
(grey italic) when the underlying cite count is zero, to avoid
misreading "flat 0% bar" as "baseline is perfect". premise-comparison
title updated from "Support-rate ceiling was the premise, not the
model" (which the new 5-seed data showed was overconfident) to
"Full-text NLI premise lifts support rate substantially" with a
subtitle noting per-seed variance.

### Added — bigger-model sweep, full-text NLI premise, annotated figures

Four things this round, all under the same "scrutinize and improve" umbrella:

**1. Bigger-model sweep.** Phi-3.5-mini (3.8B params) now has a 2-seed
sweep row alongside Qwen 0.5B and SmolLM 360M. Fabrication rate: 0.0 ±
0.0% on Phi across both seeds. Confirms the structural guarantee scales
from 360M to 3.8B without drift. Sweep JSONs live under
``benchmarks/findings/`` one-per-run; ``benchmarks/plot.py`` merges them
into the figures by (model, premise) key.

**2. Full-text NLI premise.** Before this round, the benchmarks README
flagged small-model "support-rate ceiling" (~1-14% NLI-verified on Qwen
0.5B) as a real limitation. Turns out the ceiling was mostly the NLI
*premise*, not the model. Swapping the premise from paper abstracts
(~1-2k chars) to PDF body text (~20k chars via ``pypdf``) lifts support
dramatically:

- Qwen 2.5 0.5B:     1.0% → **90.9%** (+90 pts)
- SmolLM 360M:       0.0% → **55.0%** (+55 pts)
- Phi-3.5-mini:      0.0% → **31.5%** (+31 pts)

New ``benchmarks.demo --premise fulltext`` /
``benchmarks.sweep --premise fulltext`` flags thread through to the new
``sources_from_fixtures(..., premise=...)`` helper. Populate via
``uv run python -m benchmarks.fetch_fixtures --fulltext`` (downloads 6
arXiv PDFs, extracts via pypdf, caps at 20k chars per paper).

Caveats: DeBERTa-v3's 512-token premise limit truncates silently
(chunked-NLI is the obvious next polish); pypdf can't cleanly separate
body text from headers/captions (GROBID would be cleaner but heavier).
Both documented in ``benchmarks/README.md``.

**3. Property-based fuzz tests (hypothesis).** New
``tests/unit/test_fuzz.py`` with strategies for random CSL-JSON items,
source counts, policies, and query text. Nine generative tests cover:

- Grammar always compiles with xgrammar for any (n_sources, policy,
  max_content_chars) triple
- cite-id rule always enumerates 1..N and never N+1
- Every formatter handles any well-typed CSL-JSON without crashing
- **No formatter ever emits `..` or `et al..`** — regression lock
- Numeric styles always emit ``[N]`` / ``N`` with the passed number
- ``build_rag_prompt`` numbers sources consistently; query appears
  verbatim
- ``[N]`` parse round-trips cleanly for any sequence of 1..20 ids
- ``Source`` is frozen and pydantic-round-trippable

Hypothesis immediately caught two real bugs on the first run:

- **Chicago formatter emitted ``n.d..``** when a CSL item had no year
  (single-period was concatenated with the helper's own trailing period)
- **Every numeric formatter emitted ``Title..``** when a CSL title itself
  ended in a period (title `"Essay."` → output `"Essay.."`)

Fix: routed all ~25 title/year append sites across APA / MLA / Chicago /
IEEE / Nature / Vancouver through the idempotent ``ensure_period`` helper.
Fuzz tests now green. The bug-catching speaks for the approach — these
specific shapes would never have appeared in a hand-enumerated unit test.

**4. Annotated benchmark figures.** New ``benchmarks/plot.py`` reads every
sweep JSON under ``findings/`` and emits three PNG figures:

- ``fabrication-structural-vs-empirical.png`` — the README cover. Two
  bars per scenario (baseline vs. citeformer); annotates the "structural:
  grammar mask eliminates out-of-range tokens" call-out on the 0% bar.
- ``sweep-summary.png`` — per-model comparison with mean ± std error
  bars. Left: citations emitted. Right: fabrication rate. Surfaces the
  SmolLM 4.8% abstract-run baseline fabrication that we couldn't easily
  see in text.
- ``premise-comparison.png`` — side-by-side abstract vs. fulltext NLI
  support rate per model. Swing arrows + "+N pts" annotations tell the
  story at a glance.

Cover figure embedded in both README.md and benchmarks/README.md.
``matplotlib>=3.8`` added to dev deps (it was already in the
``examples`` extra).

### Added — LangChain + LlamaIndex duck-typed integrations

New ``citeformer.integrations`` subpackage with adapters for the two
dominant RAG frameworks. Both are **duck-typed** — we don't import
LangChain / LlamaIndex at module load, so users can adopt the adapter
without the matching dependency installed, as long as their objects
have the expected attribute shape.

``citeformer.integrations.langchain``:

- ``source_from_document(doc)`` — LangChain ``Document`` → citeformer
  ``Source``. Pulls ``page_content`` into ``Source.content`` and derives
  CSL-JSON metadata from whatever the document's metadata dict contains.
- ``sources_from_documents(docs)`` — list wrapper, order-preserving.
- ``default_metadata_converter(meta)`` — the CSL-JSON derivation, factored
  out so users can override via ``metadata_converter=`` kwarg.

``citeformer.integrations.llamaindex``:

- ``source_from_node(node)`` — accepts either a bare ``TextNode`` shape
  or a ``NodeWithScore`` wrapper (unwraps ``.node`` transparently).
- ``sources_from_nodes(nodes)``.
- Mirror ``default_metadata_converter`` for LlamaIndex-style metadata.

Both converters recognise common ecosystem keys (title, source, url,
author, date, year, file_path) and handle structured author lists with
multiple shapes (CSL-JSON, ``{first, last}``, ``{name}``, bare string).
Unknown metadata gets stashed under ``_langchain_metadata`` /
``_llamaindex_metadata`` so callers don't lose signal.

20 unit tests in ``tests/unit/test_integrations.py`` using ``SimpleNamespace``
stand-ins (no LC/LI dependency needed to run the tests). Two end-to-end
integration tests exercise the full convert → generate → verify pipeline
via ``MockBackend``.

### Added — adversarial benchmark + multi-seed sweep

New `benchmarks/adversarial.py` wraps the six-paper fixtures with a prompt
that *explicitly instructs* the model to cite out-of-scope ids (``[7]`` for
Turing 1950, ``[8]`` for McCulloch-Pitts 1943). This is the demonstration
that had been missing from the README: on Qwen 2.5 0.5B / seed=0, the
baseline complies (100% fabrication — `[7]` and `[8]` appear repeatedly)
and citeformer structurally cannot (0% — grammar mask eliminates those
tokens at the logit level). Sends up a `SystemExit` if the constrained
run ever emits an out-of-range id, so it's effectively a contract test.

New `benchmarks/sweep.py` runs any (models × seeds) grid and reports mean
± std per metric. Ships with defaults of Qwen 0.5B + SmolLM 360M × 3 seeds
(both cached from earlier demo runs, ~20-30s per run on CPU). Writes
per-run rows + aggregates to `benchmarks/findings/sweep-<timestamp>.json`.
First run captured: citeformer 0.0 ± 0.0% fabrication across all 6 runs;
baseline fabricated on 1 of 6 (SmolLM 360M seed=2, 14.3% fab rate). That
single real-world baseline fabrication under a *non-adversarial* prompt
is the motivating evidence; the sweep makes it reproducible.

Shared helpers in new `benchmarks/_common.py` so `demo.py` / `adversarial.py`
/ `sweep.py` don't drift on fixture loading, source formatting, or
verification analysis.

`benchmarks/README.md` updated with full adversarial + sweep tables
(per-run + aggregate), the honest caveats (small-model support-rate
ceiling, noisy baseline support-rate when emit counts are low), and
open questions for future multi-model extensions.

### Added — `citeformer.prompts.build_rag_prompt()`

First-class prompt assembly helper. Users were stitching their own RAG
prompts before calling `Citeformer.generate()` — easy to get subtly wrong
(misnumber sources, forget to show the `[N]` shape, bury the task). Now:

```python
from citeformer import Source, build_rag_prompt
prompt = build_rag_prompt(
    query="Explain self-attention.",
    sources=sources,
    system="You are writing a technical survey. Cite every claim.",
    example="Self-attention weighs relationships across positions [1].",
    answer_prefix="Survey:",
)
```

String-in / string-out, every section optional. `build_rag_prompt` is
re-exported from the top-level `citeformer` package for discoverability
alongside `fetch_crossref` / `fetch_arxiv` / `render_references`. 13 unit
tests in `tests/unit/test_prompts.py` pin the section ordering, author-tag
format (Smith / Smith & Jones / Smith et al.), source numbering, and
input validation (empty query, empty sources).

Both `benchmarks/demo.py` and `benchmarks/sweep.py` were refactored to
use it — the helper is real, not just a stub shipped without first-party
adoption.

### Added — streaming support (`HFBackend.stream`, `LlamaCppBackend.stream`, `Citeformer.stream`)

`Citeformer.stream()` returns a `StreamingResult` — iterable for realtime
chunk consumption, finalizable to a complete `GenerationResult` after the
stream ends. Grammar enforcement applies to every yielded chunk exactly
as in non-streaming `generate()`.

```python
stream = cf.stream(prompt="…", sources=sources, max_new_tokens=120)
for chunk in stream:
    sys.stdout.write(chunk)
    sys.stdout.flush()
result = stream.finalize()  # full GenerationResult with refs + verify()
```

Backend-side: `Backend.stream()` added to the ABC with a concrete default
that falls back to `generate()` and yields the full text as one chunk —
so any backend works with `Citeformer.stream()`, but only those that
override deliver real token-by-token behavior. `HFBackend` uses
transformers' `TextIteratorStreamer` on a background thread; the
LogitsProcessor stays wired in. `LlamaCppBackend` uses
`llama_cpp.Llama(..., stream=True)`, which already supports grammar.
`MockBackend` splits the scripted response at 10-char boundaries for
test exercises. `VLLMBackend` uses the default ABC fallback for now —
vLLM's offline engine doesn't stream; async engine integration is
deferred.

12 new unit tests in `tests/unit/test_streaming.py` plus one new
integration test
(`test_hf_backend_stream_yields_multiple_chunks_and_matches_generate`)
cover the stream-to-finalize lifecycle, idempotent finalize, and
parity with `generate()` at temperature=0.

`examples/05_streaming.py` demonstrates end-to-end usage.

### Changed — cleanup of stale phase wording

Removed "P1 stub", "P2 lands", "P3 consumes" and similar phase-pointer
docstrings across `citeformer.py`, `core.py`, `render/styles.py`,
`grammar/builder.py`, `verify/report.py`, `backends/__init__.py`,
`backends/hf.py`, `cli/__init__.py`, `docs/index.md`, and
`docs/development/dev-setup.md`. These references pointed at milestones
we've already hit; keeping them made the code read as still-in-progress.
Historical context preserved in CHANGELOG and ADRs, where it belongs.

### Fixed — REQUIRED policy stalls on small models (ADR-009)

Supersedes [ADR-007](docs/decisions/007-required-policy-progression-gap.md).

The REQUIRED policy's grammar body used ``content ::= [^\[.!?]+`` — unbounded
repetition. Small instruction-tuned models (Qwen 2.5 0.5B, similar) could
stay in content state for the full `max_new_tokens` budget and never emit a
single ``[N]``. Documentation-as-mitigation (use AUTO on small models) left
the hero-line claim fuzzy.

xgrammar 0.1.30+ accepts bounded repetition in GBNF (``{m, n}`` syntax)
— verified by inspecting the compiled grammar's internal form and
confirmed by an integration test against gpt2 with a tight bound. The
grammar now emits:

```text
content ::= [^\[.!?]{1, 240}
```

After 240 non-terminating chars since the last sentence boundary, xgrammar's
mask reduces the valid-token set to whatever can advance ``cite-group`` —
the model *must* progress. New keyword-only argument
`build_grammar(..., max_content_chars=...)` exposes the bound (threaded
through HFBackend / VLLMBackend / LlamaCppBackend via their `**options`).
Pass `None` for legacy unbounded behavior; default is 240 via
`DEFAULT_MAX_CONTENT_CHARS`.

The §10.1 grammar contract grows a fourth admitted variant of the REQUIRED
body; snapshots cover both bounded and unbounded paths. No `schema_version`
bump — `content` is an internal rule, not a schema-level field.

Integration test:
- ``test_hf_backend_required_with_tight_bound_emits_citations`` runs
  REQUIRED on gpt2 with `max_content_chars=16` and asserts at least one
  in-range citation lands — would have failed pre-ADR-009.

Benchmark (`benchmarks/demo.py`) now defaults to REQUIRED (previously
forced to AUTO to avoid the stall), with a `--policy` flag and a
`--max-content-chars` override. See the [benchmarks README](benchmarks/README.md)
for the current numbers from Qwen 2.5 0.5B on the six-AI-paper RAG setup.

### Fixed — MLA / Chicago / Vancouver emit double-period after `et al.`

Live rendering against real multi-author Crossref / arXiv metadata
surfaced a double-period artifact: `"Vaswani, Ashish, et al.."` (MLA),
`"Auth0 et al.."` (Vancouver), `"Auth0, Gi0, et al.."` (Chicago). The
bibliography templates used `f"{authors}."` to terminate the author
chunk, which stacked a period after an `et al.` that already ended in
one. Fixed by routing all three through the pre-existing `ensure_period`
helper (which is idempotent against trailing punctuation).

Regression test: `test_et_al_bibliography_has_no_double_period` —
parametrised across the three affected formatters; asserts `"et al.."`
never appears in bibliography output.

### Added — CLI surface beyond `--version`

`citeformer` gains four subcommands built on the existing library API:

- `citeformer version` — installed version (unchanged).
- `citeformer styles` — list bundled style names.
- `citeformer render <csl.json> --style apa-7` — render CSL-JSON items
  as bibliography entries in the chosen style. Accepts single items or
  JSON arrays; works entirely offline.
- `citeformer fetch <DOI | arXiv id | URL | path.pdf>` — dispatch to the
  right metadata adapter and print CSL-JSON. `--include-content` keeps
  the extracted body text for URL / PDF sources; `--output FILE` writes
  to disk instead of stdout.

`generate` / `verify` are deliberately left out — at model sizes where
generation is useful, command-line invocation is awkward. Use the Python
API for that.

Six new unit tests under `tests/unit/test_cli.py` exercise the command
plumbing through typer's test runner.

### Added — runnable examples as living reports

New `examples/` directory with four self-contained scripts:

- `01_quickstart_mock.py` — shortest demo, no ML dependencies, works
  off `MockBackend`. Shape check for downstream users.
- `02_rag_with_hf_and_verify.py` — full pipeline on gpt2 with NLI
  verification. Asserts structural non-fabrication + every emitted
  marker having a rendered reference.
- `03_standalone_rendering.py` — all six formatters against the same
  CSL-JSON item. Useful as a preview tool and as the visual diff for
  new formatters.
- `04_fetch_and_render.py` — DOI + arXiv → full pipeline. Hits the
  network; disk-cached after the first run.

Each script doubles as a living report — rerunning is how you audit
current behavior. `examples/README.md` explains the set.

### Added — benchmark findings + fresh benchmark run data

`benchmarks/README.md` is now a living report with actual numbers from
the most recent Qwen 2.5 0.5B run on the six-AI-paper RAG setup. Covers
the REQUIRED vs AUTO vs baseline comparison, citation density, NLI
support rates, and the honest limitations (low support rate on 0.5B,
single-seed noise, abstract-only NLI premise). Reproducibility
instructions + open questions for future multi-model sweeps.

### Changed — cleaner docs + trimmed surface

- Removed the `citeproc-compat` pyproject extra (placeholder without
  an implementation; users who want citeproc-py can install it directly).
  ADR-004 + ADR-005 updated to reflect this.
- Stale references to "rendered by citeproc-py" across `README.md`,
  `docs/index.md`, `docs/guarantees.md`, `docs/reference/architecture.md`,
  `docs/reference/contracts.md`, `docs/conf.py`, `CLAUDE.md`, `AGENTS.md`,
  `src/citeformer/core.py`, `src/citeformer/citeformer.py`,
  `src/citeformer/render/csl.py`, `src/citeformer/render/styles.py` —
  all replaced with the home-grown-formatter reality.
- `docs/reference/architecture.md` phase table reflects completion through
  P6 and notes the current Polish tier.
- `CLAUDE.md` phase status updated from "P0 — scaffolding" to the accurate
  post-P6 state.

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
- `citeproc-py` removed from main dependencies. (A `citeproc-compat` extra
  was initially added as a placeholder for a future compat wrapper but
  removed same-release once we confirmed no implementation was landing;
  users who want citeproc-py can install it directly.)
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
