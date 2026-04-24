# ADR-008 — `GenerationResult` + `VerificationReport` bumped to schema_version=2

- **Status**: Accepted and implemented (2026-04-23).

## Context

P6 landed NLI-based `verify()`. Two shape changes followed from it:

1. **`GenerationResult.sources`** — `verify()` needs access to the source
   list to score entailment, pair cites to sources, and run the coverage
   check. Options were:
   - Pass `sources` as an argument to `verify()`. Awkward — callers would
     have to keep the source list alongside the result.
   - Carry `sources` on `GenerationResult`. Self-contained; serialised
     results round-trip cleanly.
2. **`VerificationReport.uncited_but_entailed`** — P1's schema used
   ``list[int]`` (sentence indices). P6 needs richer information per flagged
   sentence: span, candidate source, and entailment score. Introduced the
   new ``UncitedClaim`` pydantic model.

Both are breaking changes to the §10.3 contract (adding / reshaping fields
on the pinned pydantic models).

## Decision

Bump `GenerationResult.schema_version` and `VerificationReport.schema_version`
from **1 → 2**. Update both snapshot tests in
`tests/integration/test_schemas.py`; update the §10.3-related tests in
`tests/unit/test_core.py`.

## Consequences

- `Citeformer.generate()` now sets `result.sources = list(sources)` so
  default callers don't think about it.
- Users hand-constructing `GenerationResult` (tests, demo scripts, custom
  pipelines) must pass `sources=[...]`; if they don't, `verify()` raises
  `ValueError` with a clear hint rather than downstream NLI crashes.
- `VerificationReport.uncited_but_entailed` now carries
  `UncitedClaim(span, candidate_source_id, entailment_score)` — richer
  debug output for users writing their own presentation layer over the
  report.
- Snapshot tests regenerated; CHANGELOG documents the bump in the
  `Contracts (§10)` block.
- Pre-P6 serialised results (schema_version=1) are not forward-compatible —
  the `sources` field would be missing, plus `uncited_but_entailed` items
  would be ints, not objects. We considered writing a migration shim; since
  citeformer is pre-1.0 and hasn't shipped real releases, we skipped it.
  Post-1.0 we'd add `model_validator` aliases to accept the v1 shape.
