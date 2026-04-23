---
name: grammar-shape
description: The cite-id rule and per-policy grammar bodies in src/citeformer/grammar/ are §10.1 load-bearing. Tokenizer alignment, marker shape, and policy semantics matter. Read before editing.
---

# Grammar shape

The grammar machinery in `src/citeformer/grammar/` is where citeformer's core guarantee lives. Getting this wrong breaks the "structurally impossible to fabricate" promise.

## The load-bearing rule

GBNF format (xgrammar / llama.cpp / llguidance native):

```
cite-id ::= "[" ("1" | "2" | ... | "N") "]"
```

`("1" | "2" | ... | "N")` is the `<digits>` enum from the §10.1 contract — dynamically constrained at decode time to the 1-indexed source ids in scope. This is what makes fabricating a citation to a non-existent source a logit-level impossibility when the backend masks against the grammar.

## Why GBNF and not Lark

xgrammar's `compile_grammar()` expects GBNF (`rule ::= production`) rather than Lark (`rule: production`). llama.cpp's constrained-decoding also uses GBNF natively. We emit GBNF directly from `build_grammar()` to keep the pipeline one hop — no Lark→GBNF translator in the middle. Earlier iterations used Lark (with a now-removed `parse_ok` helper); the switch landed in P2b. See [`docs/reference/contracts.md`](../../../docs/reference/contracts.md) for the ceremony around DSL changes.

## Tokenizer alignment

`[1]` tokenizes differently across models — one token for GPT-2, three tokens for Llama, sometimes fragmented across whitespace for Mistral. XGrammar handles this via `TokenizerInfo.from_huggingface()`, but only if the grammar and the tokenizer info agree. Red flags:

- Hardcoding assumptions about token boundaries in code that calls xgrammar.
- Using `" ["` (with a leading space) unless you've confirmed the tokenizer aligns — xgrammar will still mask correctly, but it's a sign of shaky reasoning.
- Passing `tokenizer.vocab_size` instead of `config.vocab_size` to `TokenizerInfo.from_huggingface()` — these can differ, and the mismatch silently misaligns logit indices.

## Per-policy grammar bodies

Each policy lays different rules on top of `cite-id`:

- **`required`** — every sentence must end with at least one `cite-group`. GBNF:

  ```
  root ::= sentence (ws sentence)*
  sentence ::= content cite-group sent-end
  content ::= [^\[.!?]+
  sent-end ::= "." | "!" | "?"
  ```

- **`quotes_only`** — only quoted spans require a cite. GBNF:

  ```
  root ::= (text | quoted-cite)+
  quoted-cite ::= quote cite-group
  quote ::= "\"" [^"]* "\""
  ```

- **`auto`** — cite-group optional everywhere. GBNF:

  ```
  root ::= (text | cite-group)+
  text ::= [^\[]+
  ```

Adding a new policy = new body + new snapshot in `tests/unit/test_grammar_builder.py`. That's a §10.1 additive change (minor bump).

## What NOT to do

- **Don't change the marker shape without a major bump.** Users in the wild may have downstream parsers keyed on `[N]`.
- **Don't embed policy-specific masking into the backend layer.** The grammar is computed in `grammar/builder.py` and handed down; backends only see the compiled GBNF.
- **Don't skip the `test_hf_backend_grammar_compiles` integration test** when touching the grammar. That's the authoritative syntax check — xgrammar's parser rejecting your grammar is the definitive "this is broken".

## Design rationale

Why `[N]` instead of `(Smith 2023)`?

- `[N]` is tokenizer-friendly across every model we care about.
- `[N]` is trivially parseable client-side for post-processing.
- `[N]` decouples inline markers from metadata — the number is an index, not a semantic claim. Rendering `(Smith 2023)` from the same internal representation is a `render/inline.py` responsibility (P3), not a grammar responsibility.

If someone wants numeric-alphabetic markers, author-year markers, or footnote-style markers in the *rendered* output, that's a render-layer choice, not a grammar change.
