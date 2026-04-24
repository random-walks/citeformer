# Guarantees

What "bulletproof" actually means in citeformer.

## What's enforced, where

| Property | Local backends (HF, vLLM, llama.cpp) | API backends (OpenAI, Gemini, Mistral) | Provider-native (Anthropic) |
|---|---|---|---|
| Citation marker cannot refer to a non-existent source | **Logit-layer** — `enum`-bounded GBNF cite-id rule; out-of-scope tokens are masked to zero probability before sampling | **Schema-layer** — `strict=true` JSON schema rejects non-conforming payloads server-side; fabrication is impossible in the returned payload | **Provider-native** — the Citations API returns per-block `document_index` references; fabricated indices are a provider-side impossibility |
| Reference list always renders deterministically | **Yes** — never touches the LLM | Same | Same |
| Every inline marker has a matching reference, and vice versa | **Yes** — coupled at render time | Same | Same |
| Claim is actually supported by the cited source | **Verified** — NLI entailment via `verify()` | Same | Same |
| Sentence without a citation is actually non-factual | **Flagged** — NLI coverage check via `verify()` | Same | Same |
| Format matches the requested CSL style exactly | **Yes** — six hand-written formatters (APA 7, MLA 9, Chicago author-date, IEEE, Nature, Vancouver) | Same | Same |

All seven backends produce the same `GenerationResult` — verify / render / streaming work identically across tiers.

## What's *not* enforced

- **Claim truth.** citeformer enforces that *if* a claim is cited, the citation points at a real source; and via `verify()`, that the source entails the claim. It does *not* verify the source itself is correct. Garbage in, cited garbage out.
- **Policy appropriateness.** You pick a `citation_policy` (`required`, `quotes_only`, `auto`). citeformer enforces the grammar that policy implies — it doesn't decide for you whether that policy is right for your domain.
- **Retrieval quality.** citeformer is downstream of your retriever. If you retrieved irrelevant chunks, the model has to cite them anyway (or hit the coverage-flag branch of `verify()`).
- **Natural sentence length under REQUIRED.** The `REQUIRED` policy bounds per-sentence content at 240 characters (configurable) to guarantee progression on small models — see [ADR-009](decisions/009-bounded-content-required.md) for the structural fix to the [ADR-007](decisions/007-required-policy-progression-gap.md) stall. Sentences that would naturally run longer get clipped mid-clause, with the citation landing at clip point. Tune `max_content_chars` higher for very long-sentence technical writing, or pass `None` to disable bounding entirely.

## Tier semantics — why we frame it honestly

Calling schema-layer enforcement "bulletproof" is defensible; calling it identical to logit-layer isn't. The difference:

- **Logit tier** (local backends): the sampler never sees an out-of-scope token. Fabrication is impossible in the **generation process itself**.
- **Schema tier** (OpenAI, Gemini, Mistral): the provider's validator rejects non-conforming responses server-side. Fabrication is impossible **in the returned payload** — but the provider's own inner sampler is opaque to us.
- **Provider-native** (Anthropic Citations API): the provider's own citation system is the guarantee. We adapt their shape into `GenerationResult`; we don't add enforcement on top.

In practice the downstream consequence is identical: your code receives a `GenerationResult` whose cite ids are all in-scope. The framing difference matters only if you're writing a threat model — for most users, all three tiers deliver "no fabricated citations, ever."

## Live evidence

- [Adversarial](https://github.com/random-walks/citeformer/blob/main/benchmarks/README.md#finding-1--adversarial-100--0-fabrication): prompt explicitly demands `[7]` and `[8]` when only 6 sources are in scope. Baseline complies (100% fab); citeformer structurally can't (0%).
- [40-run multi-prompt sweep](https://github.com/random-walks/citeformer/blob/main/benchmarks/README.md#finding-5--multi-prompt-sweep-structural-guarantee-is-prompt-invariant): 0.0 ± 0.0 fabrication across every (prompt, model, seed) cell.
- [OpenAI + Anthropic live-API tests](https://github.com/random-walks/citeformer/blob/main/tests/integration/test_api_backends_live.py): env-gated integration suite that hits production endpoints and verifies the structural invariant holds end-to-end.

## Further reading

- [Architecture](reference/architecture.md) — the 6-layer design, the piggyback map, and the tiered enforcement section.
- [Contracts](reference/contracts.md) — the three §10 invariants that govern versioning.
- [PREPRINT](https://github.com/random-walks/citeformer/blob/main/PREPRINT.md) — longer design + evaluation write-up.
