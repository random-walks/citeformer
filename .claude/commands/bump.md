---
description: Bump version in _version.py + CITATION.cff and move [Unreleased] → [X.Y.Z] in CHANGELOG. Stops before commit.
argument-hint: "[patch|minor|major]"
---

Bump the citeformer version. Usage: `/bump patch` (or `minor`, or `major`).

## What to do

1. Read current `src/citeformer/_version.py` — parse `__version__`.
2. Compute new version per the requested bump type (patch / minor / major).
3. Write the new version into `src/citeformer/_version.py`.
4. Read `CHANGELOG.md`.
5. Rename the `## [Unreleased]` section to `## [X.Y.Z] — YYYY-MM-DD` using today's date (UTC).
6. Insert a new empty `## [Unreleased]` section above it.
7. Update the compare link at the bottom: `[Unreleased]: .../compare/vX.Y.Z...HEAD` and add `[X.Y.Z]: .../releases/tag/vX.Y.Z`.
8. Update `CITATION.cff` at repo root:
   - Replace the `version: ...` line with `version: X.Y.Z`.
   - Replace the `date-released: ...` line with `date-released: YYYY-MM-DD` (same UTC date as CHANGELOG).
   - Don't touch any other field. Validate with `cffconvert --validate -i CITATION.cff` if available; if not, just sanity-check that the file is still valid YAML.
9. Print a summary: old version, new version, dated entries, and the three files touched (`_version.py`, `CHANGELOG.md`, `CITATION.cff`).

## Stop before committing

Do NOT run `git add`, `git commit`, or `git tag`. That's the user's decision — they may want to amend the changelog, add a release-note paragraph, etc.

## Rubric for picking the bump type (if the user didn't)

- **Patch** — bug fixes, doc clarifications, additive internal refactors, dependency pins that don't change public API.
- **Minor** — new features, additive §10 contract changes (new CSL field passed through, new policy), `schema_version` additive with same number.
- **Major** — §10 contract break: renamed/removed CSL field, renamed `GenerationResult` field, changed citation marker shape, dropped policy. Must include a `schema_version` bump on the affected model.

If in doubt, run `/contract-check` first to confirm the bump level.
