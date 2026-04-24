# citeformer benchmarks

A living report. The promise is: running these scripts reproduces the
numbers you'll read below, on any commodity laptop, without cherry-picking.

![Citation fabrication is structural, not statistical](findings/figures/fabrication-structural-vs-empirical.png)

## What's here

- [`demo.py`](demo.py) — paired grammar-enforced vs. baseline generation on
  a six-source RAG setup.
- [`adversarial.py`](adversarial.py) — prompt explicitly demands out-of-scope
  cite ids (``[7]``, ``[8]``). Baseline complies; citeformer structurally
  can't. The demo of the core guarantee.
- [`sweep.py`](sweep.py) — multi-seed + multi-model driver with mean ± std
  aggregation. Writes JSON logs to [`findings/`](findings/).
- [`plot.py`](plot.py) — generates the annotated figures above from the
  most recent findings, merging across sweep files.
- [`_common.py`](_common.py) — shared fixture loading + verification helpers.
- [`fetch_fixtures.py`](fetch_fixtures.py) — pre-fetches paper metadata
  (and, with `--fulltext`, PDF body text via pypdf) into
  [`fixtures/`](fixtures/).
- [`findings/`](findings/) — JSON logs + PNG figures.

## Sources (`N = 6`)

| # | Paper | arXiv |
|---|---|---|
| 1 | Vaswani et al. — *Attention Is All You Need* | [1706.03762](https://arxiv.org/abs/1706.03762) |
| 2 | Devlin et al. — *BERT: Pre-training of Deep Bidirectional Transformers* | [1810.04805](https://arxiv.org/abs/1810.04805) |
| 3 | Brown et al. — *Language Models are Few-Shot Learners* (GPT-3) | [2005.14165](https://arxiv.org/abs/2005.14165) |
| 4 | Wei et al. — *Chain-of-Thought Prompting Elicits Reasoning* | [2201.11903](https://arxiv.org/abs/2201.11903) |
| 5 | Touvron et al. — *LLaMA: Open and Efficient Foundation LMs* | [2302.13971](https://arxiv.org/abs/2302.13971) |
| 6 | Dettmers et al. — *QLoRA: Efficient Finetuning of Quantized LLMs* | [2305.14314](https://arxiv.org/abs/2305.14314) |

## Reproduce

```bash
uv sync --extra dev --extra hf --extra verify

# one-time
uv run python -m benchmarks.fetch_fixtures              # metadata only
uv run python -m benchmarks.fetch_fixtures --fulltext   # + PDF body text

# headline runs
uv run python -m benchmarks.demo --policy required
uv run python -m benchmarks.adversarial --seed 0
uv run python -m benchmarks.sweep --seeds 0 1 2 3 4
uv run python -m benchmarks.sweep --seeds 0 1 2 3 4 --premise fulltext

# regenerate figures
uv run python -m benchmarks.plot
```

Default model: Qwen 2.5 0.5B Instruct (~500 MB). NLI scorer: DeBERTa-v3-
large-MNLI (~850 MB). CPU-default; CUDA works, MPS has an xgrammar
tokenizer-size bug on Apple Silicon.

## Finding 1 — Adversarial: 100 → 0 fabrication

The prompt hands the model a 6-source list and *explicitly* instructs it
to cite `[7]` (Turing 1950) and `[8]` (McCulloch-Pitts 1943). This is the
exact shape citeformer is designed to block.

| Run | Cites emitted | Out-of-scope ids | Fabrication rate |
|---|---:|---|---:|
| citeformer (REQUIRED) | 87 | `[]` | **0% (structural)** |
| baseline (plain HF)   |  8 | `[7, 8]` | **100%** |

Qwen 2.5 0.5B, seed 0, 300 tokens. Baseline complies with the demand;
citeformer's grammar mask eliminates `[7]`/`[8]` at every decode step.
The `adversarial.py` script raises `SystemExit` if citeformer ever emits
an out-of-range id, so it's effectively a contract test doubled as a demo.

## Finding 2 — Sweep aggregate: 0 ± 0 fabrication across all models

![Sweep summary](findings/figures/sweep-summary.png)

Three instruction-tuned models × up to 5 seeds each. Aggregate (merged
per model, abstract-premise preferred for fab stats since we have more
seeds there):

| Model | n | C-cites | B-cites | **C-fab%** | **B-fab%** |
|---|---:|---:|---:|---:|---:|
| SmolLM-360M-Instruct    | 3 | 43.7 ± 11.1 | 4.3 ± 3.8 | **0.0 ± 0.0** |  **4.8 ± 8.2** |
| Qwen2.5-0.5B-Instruct   | 5 | 24.6 ± 17.3 | 1.8 ± 2.7 | **0.0 ± 0.0** |  0.0 ± 0.0 |
| Phi-3.5-mini-instruct   | 2 | 28.5 ± 17.7 | 4.0 ± 0.0 | **0.0 ± 0.0** |  0.0 ± 0.0 |

*C = citeformer (REQUIRED); B = baseline. fab% = fabrication rate.*

- **citeformer emits 0 fabricated cites across every seed × every model.**
  Structural, not statistical.
- **Baseline fabricated on 1 of 13 runs** — SmolLM 360M seed=2 emitted
  `[7]` unprompted on a non-adversarial prompt. Rare but real.
- **Citation density gap is ~10×** — REQUIRED forces progression, so
  citeformer emits 25-44 cites vs baseline's 2-4.

## Finding 3 — Full-text NLI premise lifts support substantially (but noisily)

![Premise comparison](findings/figures/premise-comparison.png)

Swapping the NLI premise from abstract (~1-2k chars) to PDF body text
(~20k chars via pypdf) lifts citeformer's support rate:

| Model | Premise | citeformer support rate | seeds |
|---|---|---:|---:|
| Qwen2.5-0.5B-Instruct | abstract  |  1.0 ± 1.7 % | 3 |
| Qwen2.5-0.5B-Instruct | full-text | **46.6 ± 40.7 %** | 5 |
| SmolLM-360M-Instruct  | abstract  |  0.0 ± 0.0 % | 3 |
| SmolLM-360M-Instruct  | full-text | **55.0 ± 55.8 %** | 2 |
| Phi-3.5-mini-instruct | abstract  |  0.0 ± 0.0 % | 2 |
| Phi-3.5-mini-instruct | full-text | **31.5 ± 0.3 %** | 2 |

### The honest caveat

The Qwen fulltext number is **46.6 ± 40.7** over 5 seeds. The std is the
same order of magnitude as the mean — the number is directional, not
precise. Per-seed scores ranged 11%–94%.

Earlier drafts of this file reported "Qwen 0.5B: 90.9 ± 3.8" from a 2-seed
run. That was cherry-picked: those two seeds both happened to land in the
high tail of the distribution. Adding 3 more seeds collapsed the mean to
46.6 and exploded the std.

**Takeaway**: NLI support rate on small models is noisy regardless of
premise choice. Fabrication rate (0% structural) is the only metric in
this benchmark that's stable across seeds.

### Chunked NLI — opt-in, not default

DeBERTa-v3 has a 512-token input limit; long premises get silently
truncated. We added an opt-in **chunked-NLI** mode
(`NLIModel(chunk_premise=True)`) that slides a window over the premise
and takes max entailment across windows.

Ran this as a 5-seed comparison on Qwen fulltext:

| NLI mode | support mean ± std | per-seed scores |
|---|---:|---|
| Truncated (default) | 46.6 ± 40.7 | 94, 88, 20, 11, 20 |
| Chunked (max over 400-token windows) | 30.2 ± 24.9 | 6, 12, 20, 63, 50 |

Chunked sometimes catches claims that are buried past the 512-token
truncation horizon (seed 3 went 11% → 63%). It also inflates false
positives on unrelated claims, because max across 15 windows gives each
window a chance to cross threshold by noise. **Net effect is unclear**
and seed-dependent.

We keep chunked mode **off by default** for score stability. Turn it on
with a bumped threshold (0.7+) if you're specifically trying to score
claims that live in the paper's body text rather than the abstract:

```python
from citeformer.verify import NLIModel, Verifier
verifier = Verifier(threshold=0.7, nli=NLIModel(chunk_premise=True))
```

### pypdf vs GROBID

The "fulltext" field in our fixtures is pypdf-extracted body text.
That's noisy: page numbers, arxiv IDs, figure captions all bleed into the
premise. [GROBID](https://github.com/kermitt2/grobid) would give cleaner
body/metadata separation.

We didn't bake in GROBID because it requires a running Java server
(Docker-available, but heavyweight). If you want it, swap the
`Source.from_pdf` call in your own code for a GROBID-extracted one and
feed that into `sources_from_fixtures`.

## Why the baseline fabrication rate is noisy

Any single run can get lucky and stay in-range even without the grammar.
The sweep shows this: 12 of 13 baseline runs happened to stay in [1..6].
The 13th (SmolLM 360M seed 2) emitted `[7]` unprompted — 14.3% of its
cites were fabricated.

The structural guarantee is exactly that we don't depend on luck. A
prompt that nudges the model out of the scope (see the adversarial demo
above) is the only way to see consistent fabrication without grammar
enforcement. citeformer never fabricates regardless of prompt — no luck
required.

## `citations_checked` and the "100% supported" trap

Before schema v3 ([ADR-010](../docs/decisions/010-verification-report-schema-v3.md)),
a `VerificationReport` with zero citations reported `support_rate = 1.0`
(vacuous entailment). Tables that averaged support rate across runs
blended those "no data" entries into the mean, inflating baseline
numbers: "baseline 100% supported!" really meant "baseline emitted zero
cites so there was nothing to score."

Schema v3 adds `citations_checked` — the honest signal for
"nothing-to-score". The figures' fabrication-rate panel now labels
zero-cite bars as `n/a` instead of drawing a 0% bar. Consumers
aggregating across reports should gate on `citations_checked > 0`:

```python
rate = (
    report.support_rate
    if report.citations_checked > 0
    else None  # nothing to score; don't include in averages
)
```

## Finding 5 — Multi-prompt sweep: structural guarantee is prompt-invariant

![Multi-prompt summary](findings/figures/multiprompt-summary.png)

Ran [`multiprompt_sweep.py`](multiprompt_sweep.py) with 4 prompt shapes × 2
models × 3 seeds = 24 runs. Prompt shapes exercise common RAG request
patterns — survey (trace a landscape), compare (contrast approaches),
explain (walk through a mechanism), critique (flag limitations):

| prompt   |  n  | C-fab% | B-fab% | C-supp% | B-supp% | C-cites | B-cites |
|----------|:---:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|
| survey   |  6  | **0.0 ± 0.0** | 2.4 ± 5.8  | 23.4 ± 32.9 | 0.0 ± 0.0   | 26.3 ± 22.7 | 6.0 ± 2.2 |
| compare  |  6  | **0.0 ± 0.0** | 0.0 ± 0.0  | 2.5 ± 6.2   | 31.7 ± 40.2 | 55.8 ± 8.8  | 5.8 ± 6.3 |
| explain  |  6  | **0.0 ± 0.0** | 0.0 ± 0.0  | 34.0 ± 31.2 | 62.0 ± 44.5 | 38.0 ± 27.8 | 2.8 ± 3.4 |
| critique |  6  | **0.0 ± 0.0** | 0.0 ± 0.0  | 24.5 ± 38.3 | 11.7 ± 19.3 | 55.8 ± 5.7  | 6.5 ± 3.5 |

*C = citeformer (REQUIRED); B = baseline. Both against Qwen 2.5 0.5B
Instruct and SmolLM 360M Instruct, 3 seeds each per prompt.*

- **citeformer fab rate is 0% across every (prompt, model, seed) triple.**
  Not drifting on explain, not drifting on critique, not drifting under
  the argumentative compare shape. The structural guarantee doesn't care
  what you ask.
- **Baseline drifted on one prompt shape.** 2.4% mean fab rate on survey
  — small, but non-zero across 6 runs. The multi-paper trace is the
  shape where small models occasionally wander out of scope.
- **Citation density gap is 4-10× on citeformer**, because REQUIRED
  forces every sentence to cite. The cost is an occasional stack; the
  `deduplicate_adjacent_cites` helper folds runs of adjacent ids.
- **Support rate tracks the prompt shape.** `explain` (single-source
  focus) scores higher than `compare` (multi-source claims) across both
  conditions — consistent with what you'd expect when NLI scores against
  abstracts only.

**Takeaway**: the structural guarantee holds identically across prompt
shapes. Support rate, unsurprisingly, depends on what you ask for.

Reproduce (defaults are 2 models × 3 seeds × 4 prompts = 24 runs on
CPU, ~12 min):

```bash
uv run python -m benchmarks.multiprompt_sweep
# More seeds:
uv run python -m benchmarks.multiprompt_sweep --seeds 0 1 2 3 4
# More models:
uv run python -m benchmarks.multiprompt_sweep \
    --models Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM-360M-Instruct microsoft/Phi-3.5-mini-instruct
```

Output: `findings/multiprompt-<timestamp>.json` + a two-panel summary PNG.

## Finding 4 — NLI threshold calibration: DeBERTa-v3-large is bimodal

![NLI calibration curve](findings/figures/nli_calibration_DeBERTa-v3-large-mnli-fever-anli-ling-wanli.png)

Ran `threshold_calibration.py` against 50 hand-labelled (premise,
hypothesis, label) triples (see `calibration_data.py`). Triples pair each
of the six fixture abstracts with 4–10 hypotheses balanced between
genuinely-entailed paraphrases, clear contradictions, and deliberately-
tricky mid-scoring claims (over-specific numbers, unstated benchmarks,
plausible-but-absent contrasts).

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.05–0.15 | 0.897 | 1.000 | 0.945 |
| **0.20–0.95** | **0.929** | **1.000** | **0.963** |

- **DeBERTa-v3-large-mnli-fever-anli-ling-wanli is bimodal.** 28 of 50
  pairs score < 0.002, 19 score > 0.998; only 3 pairs land in the
  middle (one 0.15, two at 0.99). Thresholds from 0.20 to 0.95 give
  identical confusion matrices.
- **Default 0.5 is fine.** The current library default sits squarely in
  the plateau zone. Lowering to 0.2 buys nothing but codifies the
  empirical separability.
- **2/50 residual FPs are plausibility hits**, not classifier bugs —
  claims like *"LLaMA-13B outperforms GPT-3 on most benchmarks"* and
  *"QLoRA reaches ChatGPT-level performance on Vicuna"* are plausible
  extrapolations from the abstracts even though neither is *explicitly*
  stated there. Realistic false positives from NLI-on-abstract scoring.

**Takeaway**: DeBERTa-v3-large shouldn't be tuned by threshold. If you
need lower FP rate, you want either (a) a stronger model with better
calibration, (b) chunked scoring against full-text (see Finding 3), or
(c) a structured self-consistency check — not a knob turn on the
threshold.

### Finding 4b — DeBERTa-v3-**base** is under-confident, not bimodal

Ran the same 50-triple calibration against `cross-encoder/nli-deberta-v3-base` (~180 MB, ~5× smaller than the default large variant):

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.05–0.20 | 1.000 | 0.462 | 0.632 |
| 0.25–0.75 | 1.000 | 0.423 | 0.595 |
| 0.95 | 1.000 | 0.308 | 0.471 |

- **Perfect precision across every threshold.** Negatives are crushed near zero (max entailment score on a label=False pair: 0.004).
- **But recall never clears 0.46**, even at the most permissive threshold. The median label=True pair scored *0.014* — below any useful bar.
- **The failure mode is under-confidence, not mis-classification.** 14 of 26 true-entailment pairs got scored < 0.05 despite being plain paraphrases ("Transformers dispense with recurrence entirely" → score 0.011 against the Attention Is All You Need abstract).

**Takeaway**: on the same labelled set, the large variant wins by 0.33 F1 (0.96 vs 0.63). If users pick base for the weight savings, they're trading ~half their recall. The library's default is the large variant for a reason — this finding quantifies the cost of the downgrade.

Reproduce:

```bash
uv run python -m benchmarks.threshold_calibration
# smaller variant (180 MB):
uv run python -m benchmarks.threshold_calibration --model cross-encoder/nli-deberta-v3-base
# any HF MNLI checkpoint works:
uv run python -m benchmarks.threshold_calibration --model microsoft/deberta-large-mnli
```

Output: `findings/nli_calibration_<model-slug>.json` + two-panel PNG.

## Known limitations

- **NLI noise.** See Finding 3. Support rate is not a stable number on
  small models at this seed count.
- **pypdf extraction quality.** See "pypdf vs GROBID" above.
- **Small-model prose is repetitive.** REQUIRED policy on Qwen 0.5B
  sometimes stacks cites: `[1] [2] [3] [1] [2] [3] [1]`. Fix with
  `citeformer.deduplicate_adjacent_cites(result.text)`.
- **DeBERTa-v3's 512-token cap.** Addressed by opt-in chunked mode; see
  "Chunked NLI" above.
- **Sample sizes.** 2–5 seeds per model. Expand with
  `--seeds 0 1 2 3 4 5 6 7 8 9` for publication-quality numbers.

## Open questions

- **Multi-prompt sweep.** One prompt shape across all runs. Varying
  prompts (summarize / compare / critique) would surface prompt sensitivity.
- **Llama-3.2-3B / Mistral-7B.** Missing from the sweep.
- **ALCE subset.** Standardized benchmark, heavier scaffolding. Deferred.
- **Threshold calibration.** Currently 0.5 everywhere. DeBERTa entailment
  isn't calibrated; a per-model threshold sweep would give firmer support
  rates.

Contributions that add data — other models, other prompts, other NLI
backends — are welcome. The "living report" model means this file grows
more evidence, not less.
