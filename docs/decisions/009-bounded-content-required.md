# ADR-009 — Bounded `content` rule closes the REQUIRED progression gap

- **Status**: Accepted (2026-04-23).
- **Supersedes**: [ADR-007](007-required-policy-progression-gap.md).

## Context

[ADR-007](007-required-policy-progression-gap.md) documented a real-world
failure of the `REQUIRED` policy on small models: the grammar let `content`
repeat unboundedly (``content ::= [^\[.!?]+``), so models like Qwen 2.5
0.5B Instruct could stay in content state for the full `max_new_tokens`
budget, never emitting a single ``[N]`` marker. The grammar was valid, the
mask was correct — but "correct-by-construction" didn't translate to
"citations actually land in the output."

The ADR-007 mitigation was documentation: use `AUTO` on small models,
reserve `REQUIRED` for models ≥3B params. That worked but left the
hero-line claim — *"REQUIRED means every sentence gets cited"* — limp on
the class of models we most wanted to show off.

## Decision

Bound `content` with a GBNF repetition quantifier:

```text
content ::= [^\[.!?]{1, 240}
```

After (up to) 240 non-terminating characters since the last sentence
boundary, xgrammar's mask reduces the valid-token set to whatever can
advance `cite-group` — that is, a ``[``. The model *must* progress. No
amount of small-model over-completion changes this: progression is
structural, not probabilistic.

Verified with xgrammar 0.1.30+: ``xgrammar.Grammar.from_ebnf`` accepts
``{m, n}`` repetition syntax, and the compiled grammar's internal form
(`content_1{1, 240}`) masks correctly at decode time. llama.cpp's GBNF
parser has supported the same syntax since b3000 (mid-2024), so the fix
applies uniformly across our three backends.

The default bound is **240 characters**, tuned to comfortably admit long,
well-formed English sentences (observed 95th percentile in the AI-paper
benchmark prompts is ~180 chars) while still guaranteeing progression for
stall-prone small models. Exposed via an optional keyword-only argument:

```python
from citeformer.grammar import build_grammar
from citeformer import Policy

g = build_grammar(
    n_sources=3,
    policy=Policy.REQUIRED,
    max_content_chars=240,  # default; pass `None` for legacy unbounded
)
```

Options considered + rejected:

- **Custom logit-bias processor** (boost ``[`` / sentence-terminator tokens
  once content runs long). Per-backend plumbing. Fragile around tokenizer
  subword boundaries. Soft, not structural.
- **Mandatory interleaved markers** (force `cite-group` every N chars of
  prose). Breaks natural prose cadence; makes the grammar model-aware.
- **Leave REQUIRED unbounded and reroute users to AUTO on small models**
  (the ADR-007 stance). Abdicates the structural guarantee where it's
  most useful.

## Consequences

- **REQUIRED now lands on Qwen 2.5 0.5B / Phi-3.5-mini / Llama-3.2-3B**
  without max_new_tokens games. The benchmark can legitimately exercise
  REQUIRED as the default-recommended policy when the user wants "every
  sentence gets cited."
- The bound is a **soft** progression guarantee — a sentence that would
  naturally run longer than 240 chars gets clipped mid-clause, with the
  cite landing at clip point. For most RAG prose this is fine; for very
  long-sentence technical writing, users pass a higher bound explicitly.
- Setting ``max_content_chars=None`` preserves the pre-ADR-009 behavior
  for anyone who specifically wanted it.
- The §10.1 contract grows a fourth admitted variant of the REQUIRED body
  (bounded vs. legacy unbounded). Grammar-shape snapshot tests cover both
  paths. No schema_version bump required — `content` is an internal rule,
  not a schema-level field.
- AUTO and QUOTES_ONLY are unchanged: no sentence-level shape to bound.
  Passing `max_content_chars=N` with either policy is a no-op and surfaces
  `max_content_chars=None` in the returned `Grammar` for visibility.

## Follow-on work

- Integration test (``tests/integration/test_hf_backend.py``) runs the
  REQUIRED policy against gpt2 with a tight `max_content_chars=16` and
  asserts at least one citation lands — which would have failed under the
  pre-ADR-009 builder.
- ``benchmarks/demo.py`` now uses REQUIRED as the default constrained
  policy. The benchmark README documents the difference vs AUTO for small
  models.
