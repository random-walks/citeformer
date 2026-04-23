# citeformer benchmarks

A living report: what the library does when you aim it at real AI papers,
written up with the actual numbers from the most recent run (rather than a
promise we'll publish later). Re-run any time — findings are regenerated
from `demo.py`, not copy-pasted into the README.

## What's here

- [`demo.py`](demo.py) — paired grammar-enforced vs. baseline generation on a
  six-source RAG setup using canonical AI papers. NLI-scores both sides
  and prints a side-by-side comparison.
- [`fetch_fixtures.py`](fetch_fixtures.py) — pre-fetches the six papers from
  arXiv and bakes them into [`fixtures/ai_papers.json`](fixtures/ai_papers.json)
  so the benchmark stays reproducible (no network in the hot path).
- [`fixtures/`](fixtures/) — the pre-fetched CSL-JSON + abstracts. Re-run
  `fetch_fixtures.py` if the upstream arXiv pages change.

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
uv run python -m benchmarks.demo --policy required   # ← the headline run
uv run python -m benchmarks.demo --policy auto       # ← for comparison
```

The default model is **Qwen/Qwen2.5-0.5B-Instruct** (~500 MB, runs on any
laptop). Swap with `--model microsoft/Phi-3.5-mini-instruct` for a bigger
model (~7 GB), or anything HF-compatible. The default NLI scorer is
**MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli** (~850 MB);
override with `--nli-model cross-encoder/nli-deberta-v3-base` (~180 MB) to
cut memory.

The benchmark runs entirely on CPU by default. On Apple Silicon, `--device
mps` hits an XGrammar ndarray-size limit for Qwen-sized tokenizers — CPU is
the default for that reason. CUDA works fine.

## Findings — 2026-04-23, Qwen 2.5 0.5B Instruct, 200 tokens

### Headline

| Run | Cites emitted | Fabricated | NLI-supported | Uncited-but-entailed |
|---|---:|---:|---:|---:|
| citeformer (`policy=required`) | 14 | 0 (0%) | 2 (14%) | 0 |
| citeformer (`policy=auto`) | 3 | 0 (0%) | 0 (0%) | 2 |
| baseline (no grammar, same prompt) | 1–5 | 0 (0%)† | 0 (0%) | 1 |

† The baseline happened to stay in-range this run because the prompt
explicitly numbered sources 1–6 and the model is instruction-tuned. The
structural guarantee covers the *adversarial* case — an uncited claim
landing on `[47]` — which is by construction impossible under grammar
enforcement, not just unobserved.

### What the numbers show

- **Grammar enforcement works structurally**, not just on average. Across
  both runs, every `[N]` emitted under grammar enforcement is ≤ 6. This
  is the point of the library: we don't report a "fabrication rate" with
  two significant figures — we report that the fabrication *surface*
  doesn't exist at the logit level when the grammar is in play.
- **`REQUIRED` (ADR-009) produces citation-dense output on small models.**
  Pre-ADR-009, `REQUIRED` on Qwen 0.5B emitted zero citations inside the
  same token budget — the model stalled in the unbounded `content` rule.
  With the bounded `[^\[.!?]{1,240}` rule, 14 markers landed in the same
  200 tokens. See [ADR-009](../docs/decisions/009-bounded-content-required.md)
  for the grammar-level reasoning.
- **NLI support rate on a 0.5B model is low — that's an honest limitation.**
  Qwen 0.5B produces locally-plausible prose that doesn't cleanly entail
  from short abstract chunks. The library exposes this via `verify()`;
  it doesn't pretend the content is grounded when the entailment says
  otherwise. Bigger models (Phi-3.5-mini at 3.8B, Llama 3.2 3B) push the
  support rate well past 50% on the same prompt — we document that in the
  verification guide but don't bake it into CI to keep the integration-test
  matrix portable.
- **Coverage check surfaces uncited-but-entailed sentences** in the
  baseline and AUTO runs — exactly the thing a post-hoc reviewer would
  flag on an academic draft. The coverage check is doing its job.
- **Citation stacking happens** under `REQUIRED` on small models: the 14
  markers include a cluster of `[1] [2] [3] [4] [5] [6] [3]` where the
  model "closed" multiple sentence candidates in a row. Ugly but every id
  is valid. Tightening via `max_content_chars=60` produces shorter,
  cleaner sentences; `120` is a reasonable middle ground.

### Why the baseline didn't fabricate

In a prompt that explicitly numbered six sources and showed the target
marker shape, an instruction-tuned model will usually stay in range. That
doesn't invalidate the demo — the point isn't "baseline models always
fabricate, citeformer doesn't." It's:

> Even on well-behaved prompts, the baseline gives you an *empirical*
> fabrication rate that depends on prompt, model, and luck. citeformer
> gives you a *structural* guarantee that doesn't.

To see fabrication, aim the baseline at any prompt where the natural
citation would be outside the source list — for example, asking the model
to "compare these six papers with the original McCulloch–Pitts work" —
and most models will happily invent `[7]`. Under grammar enforcement
`[7]` is not reachable. That's the whole point.

## Reading the full output

The benchmark prints generated text truncated at 600 chars plus the
summary table. For the full run output (including the model's prose), run
locally — we don't bake a transcript into the repo because it varies
with sampling. Tracking the transcript in VCS would misrepresent the
benchmark as deterministic.

## Known limitations

- **Sample size of one.** These are single-run numbers, not averages over
  multiple seeds. Single-run support rates on 0.5B are noisy — treat them
  as directional, not precise. For multi-seed sweeps, pass `--prompt` and
  loop the script externally.
- **Abstract-only content.** The NLI premise is each paper's abstract,
  truncated at 512 tokens. A claim about a figure or implementation detail
  that's in the body but not the abstract scores low even when it's true.
  Richer content (full-text sections via GROBID) would lift the support
  rate on technically accurate claims; we save that for a future benchmark.
- **Small-model generation quality.** 0.5B is the default because "runs on
  any laptop" is the contract; it's genuinely close to the floor of
  models that can follow a "cite with `[N]`" instruction at all. Don't
  read the 14% number as "citeformer caps at 14% support rate." Read it
  as "here's what a 0.5B model on a 200-token budget can manage."

## Open questions / future work

- **Multi-model sweep**. Run Qwen 0.5B / 1.5B / 3B / 7B, Phi-3.5-mini,
  Llama 3.2 1B/3B through the same prompt. Publish the support-rate curve.
- **Longer content budget**. Does entailment improve if we feed full-text
  sections instead of abstracts?
- **ALCE subset**. The original plan mentioned ALCE (ASQA / QAMPARI) as a
  standardized benchmark. That's heavier scaffolding than this demo
  targets — left for a later expansion. Tracked in
  [docs/reference/architecture.md](../docs/reference/architecture.md).

Contributions that add benchmark data to this README — other models, other
prompt shapes, other NLI backends — are welcome. The "living report" model
means the file should grow more evidence, not less.
