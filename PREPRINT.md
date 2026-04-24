# Structurally Unforgeable Citations — a library design and preliminary evaluation

**Blaise Albis-Burdige · April 2026 · v0.1.0**

**Abstract.** Citation fabrication in retrieval-augmented generation
(RAG) remains a first-class reliability problem: 2026 surveys place
fabrication rates between 14 % and 95 % depending on the benchmark and
model. Prompting does not fix it; post-hoc filters only catch what their
detector sees. This report introduces **citeformer**, an open-source
Python library that makes out-of-scope citation markers
*token-impossible to sample* at the logit layer for local-runtime
language models. The citation-marker grammar is compiled per call and
handed to XGrammar or llguidance; bibliography rendering is lifted out
of the model entirely and delegated to six hand-written CSL formatters.
Across a multi-prompt × multi-seed × multi-model sweep (4 × 5 × 2 = 40
runs), observed fabrication rate is 0.0 ± 0.0 — a structural, not
statistical, result. For users on API providers, schema-layer
equivalents (OpenAI Structured Outputs, Gemini ``response_schema``,
Mistral strict JSON schema) and the provider-native Anthropic Citations
API are wrapped behind the same ``GenerationResult`` surface. We
additionally report (i) a bimodal calibration profile for the default
DeBERTa-v3-large NLI scorer on 50 hand-labelled entailment triples, and
(ii) a recall-capped failure mode on a 5× smaller DeBERTa-v3-base
variant, quantifying the cost of the common "use the small one" choice.

## 1  Motivation

RAG systems answer a user question by retrieving passages and then
conditioning a language model on them. The failure mode this library
targets is a specific one: the model generates a citation marker —
``[3]``, ``(Smith 2023)``, ``¹`` — that either (a) refers to a source
outside the retrieved set, or (b) refers to an in-scope source that
does not actually support the preceding claim. (a) is structural:
the marker is wrong by construction. (b) is semantic: the marker is
plausible but unsupported.

(a) is easier to attack. If the model cannot *produce* ``[4]`` when
only three sources are in scope, fabrication-by-marker becomes
impossible. This is the same observation jsonformer [^jsonformer] made
for JSON schema compliance in 2023 — the model's token distribution is
constrained, not merely prompted. jsonformer has been dormant since
early 2024; its core insight was never applied to citation markers.
This library is the application.

(b) is harder; it requires natural-language inference and a
ground-truth notion of "does this passage actually support this
claim?" We address (b) as a post-generation verification pass using
DeBERTa-v3-large-MNLI; section 5 reports on its calibration.

## 2  Related work

**jsonformer** demonstrated logit-layer constraint for JSON schemas
via custom Python prefix matching. Modern equivalents — **XGrammar**,
**llguidance**, **Outlines** — implement the same primitive with GPU-
native token-mask compilers, producing microsecond-scale grammar
updates and orders-of-magnitude throughput improvements over token-by-
token Python interpretation.

**ALCE** (Gao et al., 2023) standardised citation-aware evaluation in
RAG: three tasks (ASQA, QAMPARI, ELI5) and three metrics (citation
recall, citation precision, correctness). Citeformer includes an
ALCE-metric implementation (``benchmarks/alce_subset.py``); full ALCE-
reproducibility is out of scope for v0.1 and flagged for v0.2.

**Anthropic** shipped a native Citations API in January 2025. A request
with ``"citations": {"enabled": true}`` on each document returns
structured citation blocks where every assertion is tied to a
document index and character span. Citeformer ships an adapter
(``AnthropicBackend``) rather than an enforcement layer — the
provider's guarantee is already strong; our job is to make its output
shape interoperable with the rest of the library.

**OpenAI Structured Outputs** (August 2024) and **Mistral strict JSON
schema** (November 2024) extend response validation to JSON schemas
with ``strict: true``. Server-side validation of ``enum``-bounded
citation integers gives these providers the same structural guarantee
as logit masking gives local backends — just enforced in a different
layer.

## 3  Design

Six layers, in dependency order:

```
CLI → orchestration → verify → render → backends → grammar → core
```

Upper layers may import from lower; the reverse is forbidden and
enforced by test. Three invariants pin the public surface:

1. **§10.1 Citation marker grammar.** The ``cite-id`` GBNF rule is
   ``"[" (D1 | D2 | ... | DN) "]"`` where N equals the number of
   sources passed to ``generate()``. ADR-011 extends this to four
   delimiter shapes (``[N]``, ``(N)``, ``{N}``, ``^N``); the digit
   enumeration invariant is orthogonal to the delimiter choice and
   holds across all four.
2. **§10.2 Source metadata shape.** ``Source.metadata`` is a CSL-JSON
   item [^cslspec], the shape consumed by the home-grown render layer.
   Additive fields are minor-version; renames or removals are major.
3. **§10.3 Output schemas.** ``GenerationResult`` and
   ``VerificationReport`` are pydantic models carrying a
   ``schema_version`` integer. Any field-level change bumps the
   version and is gated by snapshot regression tests.

### 3.1  Grammar construction

Per call, ``build_grammar(n_sources, policy, marker_style,
max_content_chars)`` emits a GBNF string parameterised on:

- The number of sources N (dictates the ``cite-id`` enum).
- The citation policy (``REQUIRED`` / ``QUOTES_ONLY`` / ``AUTO``) —
  shapes whether sentences *must* end with a cite-group.
- The marker delimiter shape (ADR-011).
- An optional content-bound (ADR-009) — bounds per-sentence body text
  with ``{1, max_content_chars}`` so ``REQUIRED`` progresses on small
  models. Unbounded (``+``) is available as an escape hatch.

The grammar string is handed to XGrammar (the GBNF is compiled to a
compressed trie automaton) or llguidance (PDA-based interpreter) via a
thin adapter layer. XGrammar is the default on HF + vLLM because its
compile-time cost is lower; llguidance has better TTFT for smaller
models.

### 3.2  Rendering

References are rendered by hand-written formatters (APA 7, MLA 9,
Chicago author-date, IEEE, Nature, Vancouver) rather than via
citeproc-py. ADR-004 documents the decision to fork: the upstream
library had packaging issues with Python 3.12+ and the CSL surface we
need (six core styles, six CSL-JSON item types) is ~1 000 lines of
Python vs a 50 000-line dependency.

The coupling rule between cite IDs and rendered references is enforced
in ``GenerationResult.__post_init__``: every cite-id emitted by the
model has exactly one rendered ``Reference`` in ``references``, and
vice versa. The model never touches the bibliography.

### 3.3  Verification

``result.verify()`` runs DeBERTa-v3-large-MNLI [^deberta] over every
(source content, cited sentence) pair and returns a
``VerificationReport`` with per-citation entailment scores, a coverage
flag for uncited-but-entailed sentences, and an overall support rate.

The NLI model is swappable via ``Citeformer(nli_model=...)``. Full
coverage of a threshold sweep is shipped in
``benchmarks/threshold_calibration.py``.

## 4  Structural guarantee — multi-prompt evaluation

**Setup.** Four prompt shapes (survey / compare / explain / critique)
× two models (Qwen 2.5 0.5B, SmolLM 360M) × five seeds = 40 runs. Each
run generates against a six-paper AI-literature fixture set
(Transformer, BERT, GPT-3, CoT, LLaMA, QLoRA) under the REQUIRED
policy. Both citeformer-constrained and unconstrained baseline outputs
are logged; the baseline is plain ``model.generate()`` with no
LogitsProcessor.

**Finding.** Across 40 triples, citeformer's observed fabrication
rate is 0.0. No cell of the (prompt, model, seed) cube emitted a
single out-of-scope marker. Baseline fabrication rate is non-zero on
every prompt shape: ``[7]`` and ``[8]`` appear in survey and critique
outputs where the model extrapolates cite ids past the available set.
An adversarial prompt that *explicitly* demands ``[7]`` and ``[8]``
(``benchmarks/adversarial.py``) drives baseline fabrication to 100 %
and citeformer to 0 % — the structural result made visible under
worst-case pressure.

This is a structural, not statistical, result. The cite-id enum has
three entries (one per digit class in ``cite-id ::= "[" ("1" | ... |
"N") "]"``); no sampling temperature can produce a token sequence
that's not in the automaton.

## 5  NLI calibration — 50 hand-labelled entailment triples

**Data.** ``benchmarks/calibration_data.py`` ships 50 (paper,
hypothesis, label) triples spread across 18 clear paraphrases, 18
contradictions or off-topic hypotheses, and 14 mid-range tricky cases
(over-specific numbers, unstated benchmarks, plausible-but-absent
contrasts). Labels are the author's — this is a small hand-built
calibration set, not a standardised dataset.

**Finding 1 — DeBERTa-v3-large is bimodal, not knob-tunable.** Across
the 0.05 → 0.95 threshold sweep, precision sits at 0.929 and recall at
1.000 for every threshold in [0.20, 0.95]. Of 50 pairs, 28 score below
0.002, 19 score above 0.998, and only 3 land in the middle. The
default 0.5 is fine; lowering it to 0.2 buys nothing. "Turn the knob
for higher precision" is not a useful recommendation for this model —
the two FPs at the high end are plausibility hits where NLI is
correctly predicting a statement that *could* follow from the abstract
but isn't literally there.

**Finding 2 — DeBERTa-v3-base is under-confident, not mis-classifying.**
The 5× smaller base variant (``cross-encoder/nli-deberta-v3-base``,
~180 MB) achieves precision = 1.000 at every threshold but recall
caps at 0.46. 14 of 26 true-entailment pairs score below 0.05 despite
being plain paraphrases. The failure mode is under-confidence on
paraphrased premises, not mis-classification. F1 on the same labelled
set drops from 0.96 (large) to 0.63 (base) — the cost of the
size downgrade is roughly half the recall.

## 6  Implementation notes and limitations

**Small-model prose.** REQUIRED-policy generation on Qwen 0.5B
occasionally stacks cite markers: ``[1] [2] [3] [1] [2] [3] [1]``. The
library ships a ``deduplicate_adjacent_cites`` helper. Larger
instruction-tuned models (Phi-3.5-mini, Llama 3.2 3B) don't exhibit
this; it's a small-model RL-training artefact surfacing under
grammar pressure.

**Apple Silicon.** XGrammar's token-mask compiler hits an MPS
``NDArray > 2³²`` bug on the largest tokenizers (Qwen 0.5B+) when
running with the MPS backend. Workaround: ``device="cpu"``. The bug
is in MetalPerformanceShaders, not xgrammar. Benchmarks default to
CPU on Apple Silicon for this reason.

**NLI on full-text vs abstract.** Chunked NLI over full PDF bodies
produces noticeably higher recall on technical claims than NLI over
abstracts (see benchmarks/README.md Finding 3). At v0.1 the trade-off
is exposed to users via the ``premise`` parameter on the sweep script;
library-level chunking is opt-in.

**API backends are schema-layer, not logit-layer.** Our "bulletproof"
wording applies literally only to local backends. For OpenAI / Gemini
/ Mistral, the constraint lives in the provider's JSON-schema
validator; fabrication is structurally impossible in the returned
payload, but we don't control the token distribution. Documentation
makes this distinction explicit.

## 7  Availability

citeformer v0.1.0 is on PyPI (``pip install citeformer``) under
Apache-2.0. Source: <https://github.com/random-walks/citeformer>. The
HuggingFace Space demo at ``hf-space/`` boots a Gradio app that runs
the adversarial test on CPU. All benchmark data (40-run sweep, 50-
triple calibration set, 300-snapshot render regressions) is in the
repository.

## 8  Roadmap

- **v0.2**: ALCE full-reproducibility (ASQA / QAMPARI / ELI5),
  richer streaming, per-chunk NLI during generation.
- **v0.3**: TS sibling `citeformer-ts` if demand materialises post-1.0.
- **Open**: a larger calibration set (we suspect ≥200 triples are
  needed to produce a threshold curve that actually bends on the
  large variant).

## References

[^jsonformer]: Kapur, N. jsonformer: *A Bulletproof Way to Generate
Structured JSON from Language Models.* GitHub, 2023.
<https://github.com/1rgs/jsonformer>. The original insight this
library extends to citation markers.

[^cslspec]: Citation Style Language schema,
<https://github.com/citation-style-language/schema>. CSL-JSON v1.0.2.

[^deberta]: Laurer, M., Van Atteveldt, W., Casas, A. S., & Welbers, K.
*DeBERTa-v3-large-MNLI-fever-anli-ling-wanli.* HuggingFace model card,
2023. The default NLI scorer; still the SOTA open-weight NLI model
in 2026.
