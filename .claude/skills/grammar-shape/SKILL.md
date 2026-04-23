---
name: grammar-shape
description: The CITE_ID terminal and per-policy grammar rules in src/citeformer/grammar/ are §10.1 load-bearing. Tokenizer alignment, marker shape, and policy semantics matter. Read before editing.
---

# Grammar shape

The grammar machinery in `src/citeformer/grammar/` is where citeformer's core guarantee lives. Getting this wrong breaks the "structurally impossible to fabricate" promise.

## The load-bearing terminal

```
CITE_ID: "[" <digits> "]"
```

`<digits>` is constrained at decode time to the enum of in-scope source indices (`"1" | "2" | ... | "N"` for `N = len(sources)`). This is what makes fabricating a citation to a non-existent source a logit-level impossibility.

## Tokenizer alignment

`[1]` tokenizes differently across models — one token for GPT-2, three tokens for Llama, sometimes fragmented across whitespace for Mistral. The grammar compiler (XGrammar or llguidance) handles this, but only if the grammar is expressed correctly. Red flags:

- Hardcoding assumptions about token boundaries.
- Using `" ["` (with a leading space) unless you've confirmed the tokenizer aligns.
- Not testing against the full model zoo in `tests/integration/test_backend_parity.py`.

## Per-policy grammar rules

Each policy lays different rules on top of the terminal:

- **`required`** — every sentence must end with at least one `cite_group`. Grammar: `sentence: TEXT cite_group "."` (non-optional).
- **`quotes_only`** — only quoted spans require a cite. Grammar: `quote: "\"" TEXT "\"" cite_group` (non-optional *inside* a quote); sentences otherwise may end without a cite.
- **`auto`** — cites optional at any position. Grammar: `sentence: TEXT cite_group? "."`. The `verify()` coverage check is what surfaces missing citations here.

Adding a new policy = new grammar rule + new snapshot in `tests/unit/test_grammar_builder.py`. That's a §10.1 additive change (minor bump).

## What NOT to do

- **Don't change the marker shape without a major bump.** Users in the wild may have downstream parsers keyed on `[N]`.
- **Don't embed policy-specific token masking into the backend layer.** The grammar is computed in `grammar/builder.py` and handed down; backends only see the compiled grammar.
- **Don't skip the tokenizer compat test.** If you add support for a new tokenizer family (Gemma, DeepSeek, etc.), add it to `tests/integration/test_tokenizer_compat.py` and run it locally.

## Design rationale

Why `[N]` instead of `(Smith 2023)`?

- `[N]` is tokenizer-friendly across every model we care about.
- `[N]` is trivially parseable client-side for post-processing.
- `[N]` decouples inline markers from metadata — the number is an index, not a semantic claim. Rendering `(Smith 2023)` from the same internal representation is a citeproc responsibility, not a grammar responsibility.

If someone wants numeric-alphabetic markers, author-year markers, or footnote-style markers in the *rendered* output, that's a `render/inline.py` choice, not a grammar change.
