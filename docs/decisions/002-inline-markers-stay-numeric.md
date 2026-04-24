# ADR-002 — `GenerationResult.text` keeps `[N]` markers verbatim

- **Status**: Accepted (P3, 2026-04-23).

## Context

CSL styles render inline citations in different shapes:

- Numeric styles (IEEE, Nature, Vancouver): `[1]`.
- Author-date styles (APA, Chicago): `(Smith, 2023)`.
- Author-page (MLA): `(Smith 45)`.
- Footnote-based (Chicago-notes): superscript `¹`.

The model emits `[N]` under grammar constraint (§10.1). That matches numeric styles directly but "looks wrong" for author-date or footnote styles — a user picking APA sees `[1]` in prose instead of `(Smith, 2023)`.

Two design options:

1. **Rewrite the text** — post-process `GenerationResult.text` to replace `[N]` with the style's native inline form. Users get beautiful, style-correct prose straight from `result.text`.
2. **Keep the text** — `result.text` stays exactly what the model emitted; `Reference.inline_marker` carries the style-specific form for users who want to do their own substitution.

## Decision

Option 2 — keep the text.

## Consequences

- `result.text` always has `[N]` markers that match `Citation.span` char offsets precisely. Users of `Citation.span` don't have to re-index when switching styles.
- Users who want fully-rendered prose write a small post-processing step:
  ```python
  out = result.text
  for cite, ref in zip(result.citations, result.references):
      out = out.replace(f"[{cite.source_id}]", ref.inline_marker, 1)
  ```
  That's honest about what happens. The alternative hides the grammar-enforcement machinery — which is the *whole point* of citeformer.
- For v0.2+ we can add an optional `cf.generate(..., render_inline=True)` mode. When we do, we'll need to rewrite spans too, and it'll be clearly opt-in.
- Documentation responsibility: users reading "citeformer generates APA citations" expect to see `(Smith, 2023)`. Our README + quickstart need to surface this gap *before* they're surprised by the output. Landed in `docs/guarantees.md`.
