---
name: contract-invariant
description: Before modifying src/citeformer/grammar/, src/citeformer/core.py, or src/citeformer/verify/report.py, remember these are §10 contract files. Touching them requires ceremony — not silent edits.
---

# Contract invariant — §10 ceremony

Three files carry §10 contracts. Edits require a ceremony, not a silent commit.

## §10.1 — `src/citeformer/grammar/`

Changes the grammar shape (`CITE_ID` terminal) or the semantics of a policy (`required` / `quotes_only` / `auto`).

**Ceremony:**
1. Regenerate `tests/unit/test_grammar_builder.py` snapshots via `pytest --regen-all tests/unit/test_grammar_builder.py`.
2. Classify the change:
   - Adding a new policy → **minor** bump.
   - Changing an existing policy's semantics → **major** bump.
   - Changing the `CITE_ID` terminal shape → **major** bump.
3. Add a "Contracts (§10)" section to `CHANGELOG.md` [Unreleased] noting the change.
4. Fill the PR template's "Invariant touched?" section with §10.1.

## §10.2 — `src/citeformer/core.py` (`Source.metadata` type) + `src/citeformer/render/csl.py`

Changes the expected CSL-JSON shape that `Source.metadata` must conform to (and that `citeproc-py` consumes).

**Ceremony:**
1. Regenerate `tests/unit/test_csl_rendering.py` snapshots (`pytest --regen-all tests/unit/test_csl_rendering.py`).
2. Classify:
   - Passing through a new optional CSL field → **minor**.
   - Renaming / removing a field we read → **major**.
3. CHANGELOG + PR template as above.

## §10.3 — `GenerationResult` + `VerificationReport`

Pydantic models with `schema_version: 1`. Any shape change bumps `schema_version` AND requires PR-description callout.

**Ceremony:**
1. Bump `schema_version` on the owning model (1 → 2, etc.).
2. Update `tests/integration/test_generation_result_schema.py` or `test_verification_report_schema.py`.
3. Document migration path in CHANGELOG if the change is breaking.
4. CHANGELOG + PR template as above.

## The meta-rule

If you're editing one of these files and you haven't thought about which contract you're touching — **stop**. Run `/contract-check` first. "I didn't think it was a contract change" is how breaking changes ship as patches and ruin everyone's week.
