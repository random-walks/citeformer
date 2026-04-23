<!--
Thanks for the PR! A few quick things before you click Create:

1. CHANGELOG.md has an [Unreleased] entry describing the change.
2. `make lint && make test && make docs-build` all pass locally.
3. If you touched a §10 contract (grammar shape, CSL metadata schema, output
   schema_version), `/contract-check` is clean and the ceremony note below is filled.
-->

## Summary

<!-- One paragraph: what changed and why. -->

## Invariant touched?

<!-- Which §10 contract, if any, was affected? Options:
     - None — no contract changed.
     - §10.1 grammar shape — the citation marker terminal or per-policy grammar rules changed.
     - §10.2 CSL metadata — the expected CSL-JSON shape on Source.metadata changed.
     - §10.3 output schemas — GenerationResult or VerificationReport schema_version bumped.
     If yes, describe the ceremony (snapshot regenerated? schema_version bumped? etc.). -->

None.

## Test plan

<!-- Bulleted checklist: what you ran, what you verified. -->

- [ ] `make lint` green
- [ ] `make test` green
- [ ] `make docs-build` green (if docs touched)
