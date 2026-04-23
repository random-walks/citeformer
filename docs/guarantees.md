# Guarantees

What "bulletproof" actually means in citeformer.

## What's enforced, where

| Property | v0.1 backends (HF, vLLM, llama.cpp) | API-provider backends (future v0.2+) |
|---|---|---|
| Citation marker cannot refer to a non-existent source | **Structural** — logit-masked grammar | Schema-enforced via enum |
| Reference list always renders deterministically | **Yes** — never touches the LLM | Same |
| Every inline marker has a matching reference, and vice versa | **Yes** — coupled at render time | Same |
| Claim is actually supported by the cited source | **Verified** — NLI entailment via `verify()` | Same |
| Sentence without a citation is actually non-factual | **Flagged** — NLI coverage check via `verify()` | Same |
| Format matches the requested CSL style exactly | **Yes** — citeproc-py + 10k CSL styles | Same |

## What's *not* enforced

- **Claim truth.** citeformer enforces that *if* a claim is cited, the citation points at a real source; and via `verify()`, that the source entails the claim. It does *not* verify the source itself is correct. Garbage in, cited garbage out.
- **Policy appropriateness.** You pick a `citation_policy` (`required`, `quotes_only`, `auto`). citeformer enforces the grammar that policy implies — it doesn't decide for you whether that policy is right for your domain.
- **Retrieval quality.** citeformer is downstream of your retriever. If you retrieved irrelevant chunks, the model has to cite them anyway (or hit the coverage-flag branch of `verify()`).
- **Sentence closure under REQUIRED.** The ``REQUIRED`` policy's grammar says every sentence must end with a cite-group — but small models (≤1B params) sometimes stall in "content" state indefinitely and never emit a ``[``, producing prose with zero citations that gets truncated at ``max_new_tokens``. The guarantee that *any emitted* cite is in-range still holds; the guarantee that a sentence *will* emit one doesn't. See [ADR-007](decisions/007-required-policy-progression-gap.md). Workaround: use ``AUTO`` + a citation-dense prompt on small models, or a ≥3B model for REQUIRED.

## Why v0.1 skips API providers

OpenAI, Gemini, and Anthropic all expose *some* form of structured output, but not all of them give you grammar-level logit control. OpenAI and Gemini's schema-level enforcement is bulletproof within the enum constraints — good enough for a future tier. Anthropic's [Citations API](https://platform.claude.com/docs/en/build-with-claude/citations) (launched Jan 2025) already solves the citation problem on Anthropic — citeformer adds nothing there.

v0.1 focuses on the case where the guarantee is *strongest* (local models, grammar-level logit enforcement) and the gap is *largest* (no equivalent library exists). API providers are a planned v0.2+ expansion with honestly-documented schema-level guarantees.

## Further reading

- [Architecture](reference/architecture.md) — the 6-layer design and phase plan.
- [Contracts](reference/contracts.md) — the three §10 invariants that govern versioning.
