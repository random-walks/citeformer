# ADR-013 — `Citation` extended with cited_text + source_span + document_title

- **Status**: Accepted and implemented (2026-04-25).

## Context

Anthropic's Citations API returns rich per-citation metadata on every
text block: `document_index`, `cited_text` (the exact span the assistant
cited from), `start_char_index` / `end_char_index` (offsets into the
source content), and (since 2025) `document_title`. The pre-bump
`AnthropicBackend._flatten_blocks` extracted only `document_index` and
discarded the rest — useful information thrown on the floor.

Two downstream uses matter enough to surface this:

1. **Display.** A user reading citeformer output gets `[1]` markers and
   a bibliography. They can't see *which passage* in the source the
   model cited from. The information exists; we just weren't keeping it.
2. **Verification precision.** `verify()` runs NLI against the entire
   source content. If we kept the cited span, NLI could score against
   just that span — much sharper signal, fewer false positives from
   unrelated passages in the same document.

Other API backends (OpenAI, Gemini, Mistral) and local backends don't
have a notion of "the model cited this exact span" — they emit a
source-id integer and that's it. So the fields must be optional.

## Decision

1. Extend `citeformer.core.Citation` with three new optional fields:
   - `cited_text: str | None = None`
   - `source_span: tuple[int, int] | None = None`
   - `document_title: str | None = None`
2. Combine with the ADR-012 `usage` bump — both shape changes land in
   the same `schema_version: 3`. The branch is unreleased; bumping
   twice for two changes that ship together would be ceremony for no
   one. The `Citation` snapshot was regenerated to include the new
   fields with null defaults; pre-existing v2 serialisations
   deserialise cleanly (the new fields default to `None`).
3. The `AnthropicBackend` populates the metadata via a side-channel
   `last_rich_citations: list[dict]` instance attribute — one entry
   per marker emitted, in the same left-to-right order the
   orchestrator's regex parser sees. The `_flatten_blocks` helper
   takes an optional `record=` list parameter that gets appended to as
   it walks Claude's citation events.
4. The `Citeformer` orchestrator pulls the rich list via
   `getattr(backend, "last_rich_citations", None)` (mirrors the
   `last_usage` pattern from ADR-012) and zips it with the parsed
   marker list inside `_parse_citations`. Length-mismatch falls
   through silently with the new fields left `None` — misaligned data
   is worse than no data.
5. The `StreamingResult.finalize()` path reads the same side-channel,
   so streaming outputs carry the rich metadata too once the stream
   exhausts.

## Consequences

- **§10.3 contract**: `GenerationResult.schema_version` stays at 3 (set
  by ADR-012). Snapshot regenerated to include the new Citation
  fields. Pre-bump v2 serialisations deserialise cleanly.
- **AnthropicBackend** carries a new `last_rich_citations: list[dict]`
  instance attribute — populated at the end of every `generate()` /
  exhausted `stream()` call. Empty list before the first call.
- **Other backends** are untouched; their `Citation` objects come back
  with `cited_text=None` / `source_span=None` / `document_title=None`.
  This is honest signalling (the backend doesn't know the cited span)
  rather than a missing capability.
- **Verification** doesn't yet exploit the new fields — that's a
  follow-up: `verify()` could optionally score NLI against
  `cited_text` instead of `Source.content` when available, sharpening
  precision on long documents. Out of scope for this PR; tracked as a
  v0.3 candidate.
- **Anthropic-incompatibility note**: per Anthropic's docs, Citations
  and Structured Outputs are mutually exclusive — combining them
  returns a 400. citeformer's Anthropic backend doesn't combine them
  (we use Citations exclusively for this backend), so the constraint
  doesn't bite us, but the AnthropicBackend docstring now flags it
  for users who might add custom system prompts.

## Why not extend with a separate `CitationAttribution` sub-model?

Considered: nesting the rich fields under
`Citation.attribution: CitationAttribution | None`. Rejected because
- Most consumers want the cited text directly; a level of nesting
  hurts ergonomics for the common case.
- Pydantic frozen models with optional sub-models complicate the
  serialisation/round-trip story for downstream tooling.
- The three new fields are conceptually one cluster (provider-side
  span attribution). Flattening keeps `Citation` as the single
  authoritative type per inline marker.
