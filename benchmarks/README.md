# citeformer benchmarks

A living report: what the library does when you aim it at real AI papers,
written up with the actual numbers from the most recent run. Re-run any
time — findings are regenerated from scripts, not copy-pasted.

## What's here

- [`demo.py`](demo.py) — paired grammar-enforced vs. baseline generation on a
  six-source RAG setup using canonical AI papers. NLI-scores both sides and
  prints a side-by-side comparison.
- [`adversarial.py`](adversarial.py) — same sources, but the prompt
  explicitly instructs the model to emit out-of-scope cite ids (``[7]``,
  ``[8]``). Baseline complies; citeformer structurally can't. This is the
  demonstration of the library's core guarantee.
- [`sweep.py`](sweep.py) — multi-seed + multi-model driver that turns a
  single-run number into "mean ± std over N seeds" across any models you
  pass. Writes per-run rows + aggregates to
  [`findings/sweep-<timestamp>.json`](findings/).
- [`_common.py`](_common.py) — shared fixture loading / source-list formatting
  / analysis helpers so the three scripts don't drift.
- [`fetch_fixtures.py`](fetch_fixtures.py) — pre-fetches the six papers from
  arXiv and bakes them into [`fixtures/ai_papers.json`](fixtures/ai_papers.json)
  so the benchmarks stay network-free in the hot path.
- [`fixtures/`](fixtures/) — the pre-fetched CSL-JSON + abstracts.
- [`findings/`](findings/) — saved sweep outputs (JSON logs). One file per run.

## Sources in scope (`N = 6`)

| # | Paper | arXiv |
|---|---|---|
| 1 | Vaswani et al. — *Attention Is All You Need* | [1706.03762](https://arxiv.org/abs/1706.03762) |
| 2 | Devlin et al. — *BERT: Pre-training of Deep Bidirectional Transformers* | [1810.04805](https://arxiv.org/abs/1810.04805) |
| 3 | Brown et al. — *Language Models are Few-Shot Learners* (GPT-3) | [2005.14165](https://arxiv.org/abs/2005.14165) |
| 4 | Wei et al. — *Chain-of-Thought Prompting Elicits Reasoning* | [2201.11903](https://arxiv.org/abs/2201.11903) |
| 5 | Touvron et al. — *LLaMA: Open and Efficient Foundation LMs* | [2302.13971](https://arxiv.org/abs/2302.13971) |
| 6 | Dettmers et al. — *QLoRA: Efficient Finetuning of Quantized LLMs* | [2305.14314](https://arxiv.org/abs/2305.14314) |

These six span the transformer story from 2017 (the original architecture)
to 2023 (efficient finetuning of large pretrained models). They're mutually
on-topic so a language model is plausibly able to produce prose that cites
multiple sources per paragraph.

## How to reproduce

```bash
uv sync --extra dev --extra hf --extra verify
uv run python -m benchmarks.fetch_fixtures   # one-time; writes fixtures/ai_papers.json

# headline runs
uv run python -m benchmarks.demo --policy required   # 1-run demo
uv run python -m benchmarks.adversarial --seed 0     # show the structural guarantee
uv run python -m benchmarks.sweep                    # multi-seed averages

# variants
uv run python -m benchmarks.demo --policy auto
uv run python -m benchmarks.sweep --seeds 0 1 2 3 4 --models Qwen/Qwen2.5-0.5B-Instruct microsoft/Phi-3.5-mini-instruct
```

Default model is **Qwen/Qwen2.5-0.5B-Instruct** (~500 MB). NLI scorer is
**MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli** (~850 MB);
override with `--nli-model cross-encoder/nli-deberta-v3-base` (~180 MB) to
cut memory. The benchmark runs on CPU by default — on Apple Silicon, MPS
hits an XGrammar ndarray-size limit with Qwen-sized tokenizers, so CPU is
the default. CUDA works fine.

## Adversarial finding — 2026-04-23, Qwen 2.5 0.5B, seed=0

The adversarial prompt hands the model a 6-source list and then explicitly
instructs: *"cite Alan Turing's 1950 paper as `[7]` and McCulloch-Pitts
1943 as `[8]`"*. This is the exact shape citeformer is designed to make
impossible at the logit level.

| Run | Cites emitted | Out-of-scope ids | Fabrication rate |
|---|---:|---|---:|
| citeformer (`policy=required`) | 87 | `[]` | **0% (structural)** |
| baseline (plain HF generate) | 8 | `[7, 8]` | **100%** |

Baseline text (seed=0, 300 tokens):

> The development of AI has been marked by several key milestones:
>
> - **1950**: Alan Turing published "Computing Machinery and Intelligence" **[7]**.
> - **1951**: John McCarthy published "Computer Science and Artificial Intelligence" **[7]**.
> - **1952**: Marvin Minsky published "The Nature of the Mind" **[8]**.
> - **1957**: Marvin Minsky and Seymour Papert co-authored "Learning Machines" **[8]**.
> - **1958**: John McCarthy published "Machine Learning" **[8]**.

citeformer text (same seed, same token budget) opens with:

> The development of AI has been marked by several key milestones:
>
> - **Early To Modern**:
>   - **1950**: Alan Turing published "Computing Machinery and Intelligence" **[1] [2] [3] [4] [5] [6] [1] [2] [3] [4] [5] [6]** …

The model *wants* to emit `[7]` and `[8]` — the instruction explicitly
requested them — but the grammar mask reduces the valid-token set to
`[1..6]` at every decode step. The visible artifact is citation stacking:
the model cycles through the in-scope ids because it was told to cite
something that isn't there. Ugly, but every id is valid. That's the
structural guarantee as observed behavior.

**The demo you want to run first** is `python -m benchmarks.adversarial
--seed 0`. It's the cleanest illustration of why the library exists.

## Multi-seed sweep — 2026-04-23, CPU, REQUIRED policy, 3 seeds

Ran `demo.py`'s prompt across two small-model × three-seed combinations
(= six runs) via `sweep.py`. Full JSON log at
[`findings/sweep-20260423T212115Z.json`](findings/).

Per-run table:

| Model | Seed | C-cites | C-fab% | C-supp% | B-cites | B-fab% | B-supp% | sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0 | 47 | 0.0 | 0.0 | 0 | 0.0 | 100.0 | 30.5 |
| Qwen2.5-0.5B-Instruct | 1 | 34 | 0.0 | 2.9 | 3 | 0.0 | 0.0 | 23.9 |
| Qwen2.5-0.5B-Instruct | 2 | 5 | 0.0 | 0.0 | 0 | 0.0 | 100.0 | 24.1 |
| SmolLM-360M-Instruct | 0 | 45 | 0.0 | 0.0 | 0 | 0.0 | 100.0 | 17.7 |
| SmolLM-360M-Instruct | 1 | 54 | 0.0 | 0.0 | 6 | 0.0 | 0.0 | 20.5 |
| SmolLM-360M-Instruct | 2 | 32 | 0.0 | 0.0 | 7 | **14.3** | 0.0 | 18.1 |

(*C- = citeformer run; B- = baseline. fab% = fabrication rate on that run;
supp% = NLI support rate from `verify()`. sec = wall-clock seconds per run.*)

Aggregate (mean ± std across 3 seeds):

| Model | n | C-supp% | B-supp% | C-cites | B-cites | **C-fab%** | **B-fab%** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 3 | 1.0 ± 1.7 | 66.7 ± 57.7 | 28.7 ± 21.5 | 1.0 ± 1.7 | **0.0 ± 0.0** | 0.0 ± 0.0 |
| SmolLM-360M-Instruct | 3 | 0.0 ± 0.0 | 33.3 ± 57.7 | 43.7 ± 11.1 | 4.3 ± 3.8 | **0.0 ± 0.0** | 4.8 ± 8.2 |

### What the sweep shows

- **citeformer: 0.0% ± 0.0 fabrication across all six runs.** Every
  constrained run on every seed on every model — the structural guarantee
  is *consistent*, not average. This is the single most important row.
- **Baseline fabricated on 1 of 6 runs** — SmolLM seed=2 emitted `[7]`
  unprompted (14.3% of its 7 emitted cites). Baseline fabrication is rare
  but real on a standard (not adversarial) prompt. The adversarial run
  above is what you see when you *try* to make it fabricate; the sweep is
  what you see when you don't.
- **Baseline support% is noisy because emit count is low**: the high
  baseline support rates (100%, 100%, 100%) all correspond to runs where
  the baseline emitted 0 citations — our support-rate formula defines
  "no citations = 100% supported" by convention. In practice the baseline
  just doesn't cite much without specific prompting. citeformer's
  REQUIRED policy does force citations (28-54 per run), so its support
  rate is computed over a real denominator.
- **Citation density gap: ~10×.** citeformer's REQUIRED emits 28.7 and
  43.7 cites per run on average; baseline emits 1.0 and 4.3. That's the
  policy doing what it says on the tin, not the grammar itself.
- **Variance on citeformer cite counts is high** (std 21.5 on Qwen) —
  specific seed and content-bound interactions sometimes truncate early.
  Worth tightening `max_content_chars` on a per-model basis; the default
  (240) is a generous compromise.

## Known limitations

- **Small-model ceiling on support rate.** Qwen 0.5B produces
  locally-plausible prose that doesn't cleanly entail from short abstract
  premises. Bigger models (Phi-3.5-mini at 3.8B, Llama 3.2 3B) push the
  support rate past 50% on the same prompt — we document that anecdotally
  in [`docs/verification.md`](../docs/verification.md) but haven't baked a
  bigger-model run into this file yet (would require a ~7 GB download).
  Contributions welcome: `uv run python -m benchmarks.sweep --models
  microsoft/Phi-3.5-mini-instruct` and PR the JSON log + a row in the
  table above.
- **Abstract-only content** for the NLI premise. Body text would lift
  the support rate on technically accurate claims that aren't in the
  abstract.
- **Sample size 3 on the sweep.** Enough to see that citeformer's
  fabrication rate is consistently 0 and baseline's *isn't*. Not enough
  for tight support-rate confidence intervals. Run `--seeds 0 1 2 3 4 5
  6 7 8 9` for publication-quality averages.

## Open questions / future work

- **Bigger-model curve.** Run the sweep on Phi-3.5-mini (3.8B) and
  Llama-3.2-3B. Publish support-rate vs. model-size.
- **Longer content budget.** Does entailment improve if we feed full-text
  sections instead of abstracts? GROBID-extracted body text is already
  supported by `Source.from_pdf`.
- **ALCE subset.** ASQA + QAMPARI would give a standardized comparison
  point. Heavier scaffolding than this demo targets; left for a later
  expansion.

Contributions that add data to this file — other models, other prompts,
other NLI backends — are welcome. The "living report" model means the
file grows more evidence, not less.
