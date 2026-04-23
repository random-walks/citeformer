---
name: release-bump
description: Use this rubric when deciding between patch, minor, and major version bumps for citeformer. Patch bumps are cheap — prefer frequent small releases over feature-batching.
---

# Release bump rubric

citeformer versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with patch bumps preferred over feature-batching.

## Patch — bug fixes, docs, internal refactors

Pick patch if:

- You fixed a bug without changing the public API.
- You clarified documentation, typos, or examples.
- You refactored internals with no user-visible effect.
- You pinned a dependency tighter (e.g. `torch>=2.4` → `torch>=2.5`) for a compatibility reason.
- You updated CI, build, or tooling configs.

## Minor — new features or additive contract changes

Pick minor if:

- You added a new public function, class, backend, policy, CSL style, or metadata adapter.
- You added a new optional parameter to an existing function.
- You added a new optional dependency extra.
- You added a new field to `Source.metadata` passthrough (§10.2 additive).
- You added a new policy (§10.1 additive — e.g. `numeric_only` alongside `required`/`quotes_only`/`auto`).
- You added a new field to `GenerationResult` or `VerificationReport` without bumping `schema_version` (§10.3 additive; fields are non-breaking because existing consumers ignore them).

## Major — §10 contract break

Pick major if any of the three §10 contracts is *broken* (not additive):

- **§10.1** — the `CITE_ID` terminal shape changes; the semantics of an existing policy change; a policy is removed.
- **§10.2** — a field is renamed or removed from the `Source.metadata` expected shape.
- **§10.3** — a field is renamed or removed from `GenerationResult` or `VerificationReport`; `schema_version` must be bumped (1 → 2).

Major bumps must include:

- A `schema_version` bump on the affected model (§10.3 only).
- A CHANGELOG `Contracts (§10)` section describing the break and the migration path.
- `contracts:breaking` PR label if the CI enforces it.

## When in doubt

- Unsure whether a change is additive or breaking? Run `/contract-check`. The agent will classify and tell you the recommended bump.
- Unsure whether a fix is a bug or a behavior change? If it changes observable output for any valid input, it's at least minor. If it changes the shape of the output, it's probably major.

## Cadence

Target 1–2 patch releases per week while the repo is under active development. Minor releases land when a phase completes (P1, P2, …) or when a coherent feature batch is ready. Major releases are rare by construction — they only come from contract breaks.
