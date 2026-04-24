# ADR-011 — Configurable inline marker shapes

**Status:** Accepted (2026-04-23)
**Context:** §10.1 grammar contract

## Context

Until this ADR, the citation-marker grammar terminal was hard-coded to the
`[N]` bracket form: `cite-id ::= "[" <digits> "]"`. Downstream pipelines that
already reserve `[` / `]` for other syntax (Markdown link markup, LaTeX
options) had to strip markers and reinsert them post-hoc, which defeats the
whole "structurally unforgeable at decode time" pitch — the post-hoc rewrite
is, by definition, an unchecked mutation.

Three asks pushed us to make the shape configurable:

1. Markdown-first workflows where `[N]` collides with link syntax.
2. Citation styles that conventionally render numeric markers as `(N)` or
   superscript.
3. A caret-prefixed `^N` shape requested for console-facing pipelines where
   bracket escaping is annoying.

## Decision

Add a `MarkerStyle` enum in `citeformer.core` with four variants:

| Variant | Marker shape | Open char | Close char |
|---|---|---|---|
| `BRACKET` (default) | `[N]` | `[` | `]` |
| `PAREN` | `(N)` | `(` | `)` |
| `CURLY` | `{N}` | `{` | `}` |
| `CARET` | `^N` | `^` | — |

Plumbed through:

- `build_grammar(..., marker_style=...)` — the `cite-id` terminal + the
  `text` / `content` negated character classes both derive from the style's
  `open_char`.
- `Citeformer(..., marker_style=...)` and `Citeformer.generate(...,
  marker_style=...)` (per-call override wins).
- Each real backend (`HFBackend`, `LlamaCppBackend`, `VLLMBackend`) reads
  the option from `**options` and passes it into `build_grammar`.
- `MockBackend` honours it in its fallback echo so test fixtures render in
  the right shape without per-test plumbing.
- Citation parsing uses a `MarkerStyle → re.Pattern` table; the parser
  regex matches whatever shape the grammar emitted.

## Contract impact

§10.1's invariant is *"the cite-id digit enum is bounded by
`range(1, n_sources + 1)`"*. That holds identically across all four marker
styles — the enum is the same, only the delimiters change. This is therefore
an **additive minor change**, not a breaking one:

- Default value is `BRACKET`, matching the prior hard-coded shape.
- Existing callers see no behaviour change.
- `Grammar` gains a `marker_style: MarkerStyle` field; grammar snapshot
  YAMLs grew by one line.

## Alternatives considered

- **Unicode superscript markers (¹²³)** — deferred. Requires token-level
  multi-char terminals (`"¹⁰"` for N=10 with two characters) and a
  tokenizer-aware mapping table; lots of rendering fragility for marginal
  value. Revisit if a concrete consumer asks.
- **Arbitrary user-supplied open/close chars** — rejected. Invites combos
  that clash with GBNF special syntax (`"` / `\`) and bloats the parser's
  regex cache. The four enum values cover every asked-for shape.

## Consequences

- `Grammar` dataclass + serialization gains one field; downstream
  consumers reading the dataclass directly may need a minor update.
- Documentation: a short note in `docs/reference/contracts.md` explains
  that `marker_style` is part of the §10.1 surface but doesn't rock the
  digit-enum invariant.
- Tests: 15 new unit tests in `tests/unit/test_marker_styles.py` exercising
  mock-echo + parse + streaming + reference-render for all four styles.
  Grammar-builder tests got 4 new parametrised variants.
