---
description: Full release pipeline — preflight, bump, commit, tag. Pushes only if --direct.
argument-hint: "[patch|minor|major] [--direct]"
---

End-to-end release flow for citeformer. Usage: `/release minor` (opens a PR) or `/release minor --direct` (pushes main + tag directly; requires permission).

## Steps

1. **Preflight** — run `/release-check` and confirm green. If anything fails, stop.
2. **Bump** — invoke `/bump <type>` to update `_version.py` + `CHANGELOG.md` + `CITATION.cff`.
3. **Commit** — stage `src/citeformer/_version.py` + `CHANGELOG.md` + `CITATION.cff`, commit with message `release: vX.Y.Z`.
4. **Tag** — create `vX.Y.Z` annotated tag locally.
5. **Push**:
   - **Default (no flag)**: open a branch `release/vX.Y.Z`, push it, open a PR via `gh pr create` titled `release: vX.Y.Z` with the new changelog section as the body. Human review + merge required before the tag hits main. Tag lands when someone pushes it after merge.
   - **With `--direct`**: push `main` and the tag directly (`git push origin main && git push origin vX.Y.Z`). Only use this for patch releases on a clean branch; the GitHub Actions release workflow triggers on the tag.

## Safety

- Do not invoke `--direct` unless the user explicitly asked for it.
- Never force-push.
- Confirm the version in `_version.py` matches the tag before pushing.
