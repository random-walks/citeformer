# ADR-007 — REQUIRED policy lets the model stall in content state

- **Status**: Superseded by [ADR-009](009-bounded-content-required.md) (2026-04-23).
  Kept for historical context — the problem described below is real, but the
  "accept as documented" response was replaced same-day by a structural fix
  (bounded `content` repetition). See ADR-009 for the current behavior.

## Context

The §10.1 grammar for the `REQUIRED` policy is:

```
root ::= sentence (ws sentence)*
sentence ::= content cite-group sent-end
content ::= [^\[.!?]+
sent-end ::= "." | "!" | "?"
cite-group ::= cite-id (ws cite-id)*
cite-id ::= "[" ("1" | "2" | ... | "N") "]"
```

`content` is defined as `1+` non-``[.!?`` characters. At every decode step,
the grammar lets the model either (a) emit another `content` character or
(b) transition to `cite-group` by emitting ``[``. There is no upper bound on
`content` length. Well-behaved instruction-tuned models *want* to end
sentences — their probability mass is biased toward ``.``, ``!``, ``?``,
and brackets once content runs long enough. Small models (≤1B params) often
aren't: they happily continue prose indefinitely, never emitting ``[`` until
they hit `max_new_tokens` — at which point generation truncates mid-sentence
with zero cite markers.

We observed this on Qwen 2.5 0.5B Instruct on CPU during the P6 benchmark:
``REQUIRED`` output at ``max_new_tokens=200`` contained zero ``[N]``
markers. The grammar was valid, xgrammar was masking correctly, but the
model never transitioned out of `content` state.

Options considered:

- **Bounded repetition** (``content ::= char{1,60}``) — unsupported in
  vanilla GBNF. llama.cpp's extension supports it; xgrammar doesn't yet.
- **Force progression via interleaved mandatory markers** — e.g. insert a
  mandatory ``content cite-group`` pair every N characters. Breaks the
  natural prose flow and makes the grammar model-aware.
- **Logit-bias the ``[`` token upward** after some content length — cross-
  cutting hack that conflates backend plumbing with grammar semantics.
- **Accept as documented behavior** — ship ``REQUIRED`` with a clear note
  that it requires a model big enough to want to close sentences, and
  recommend ``AUTO`` + prompt pressure for small models.

## Decision

Accept as documented. The ``REQUIRED`` policy is correct-by-construction —
if the model emits a citation, it's in-range, and every completed sentence
has one. The gap is in *progression*: the grammar doesn't guarantee the
model will complete any sentence before hitting ``max_new_tokens``.

- The guarantee stands: **no fabricated citations** under any policy.
- The ``AUTO`` policy is fine for small models + citation-dense prompts.
- The ``REQUIRED`` policy is useful for larger models (≥3B params) where
  sentence closure is a strong prior.

Document this in:
- ``docs/guarantees.md`` — the "what's not guaranteed" section.
- The `REQUIRED` policy docstring in ``src/citeformer/core.py``.
- The demo benchmark's comments — ``benchmarks/demo.py`` uses `AUTO` with
  a note pointing here.

## Consequences

- Users hitting this on a small model either (a) switch to `AUTO` and push
  on the prompt, or (b) increase `max_new_tokens` and hope the model
  eventually closes.
- A future v0.2+ revision could implement bounded-content enforcement by
  compiling to XGrammar's lower-level API and injecting a length guard —
  revisit when we have demand.
- The integration tests use small models and must pass under `AUTO` or
  generate enough tokens to stabilise. The HFBackend integration tests
  (``tests/integration/test_hf_backend.py``) already use modest generation
  budgets and don't assert citations land — only that when they do, they're
  in range. No test changes needed.
