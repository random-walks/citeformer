# Releasing

## Policy

**Patch bumps are cheap.** Prefer frequent small releases over feature-batching. Contract changes (§10.1/§10.2/§10.3 — see [contracts.md](../reference/contracts.md)) are the one exception: they demand a deliberate ceremony.

Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **Patch** — bug fixes, doc clarifications, additive internal refactors.
- **Minor** — new features, additive contract changes (new CSL field, new policy, `schema_version` additive).
- **Major** — contract breaks (renamed/removed CSL field, renamed `GenerationResult` field, changed citation marker shape, dropped policy).

## The release flow

1. Land all PRs for the release into `main`.
2. Run `/bump [patch|minor|major]` (or invoke the `release-bump` skill) — updates `src/citeformer/_version.py` and moves `[Unreleased]` into a numbered `CHANGELOG.md` entry.
3. Commit, push, open a "release vX.Y.Z" PR. Merge after CI is green.
4. Tag on main: `git tag vX.Y.Z && git push --tags`.
5. GitHub Actions `release.yml` picks up the tag, builds sdist + wheel, publishes to PyPI via OIDC trusted publishing, and attaches artifacts to a GitHub Release with a CHANGELOG-link note.

## Pre-release checklist

Before pushing the tag:

- [ ] `make lint && make test && make docs-build` all green locally
- [ ] CI green on `main`
- [ ] `CHANGELOG.md` has the new version section (not `[Unreleased]`)
- [ ] `/contract-check` clean if any §10 files were touched
- [ ] `src/citeformer/_version.py` matches the tag
- [ ] Any new deps reflected in the right `pyproject.toml` extra

## ReadTheDocs — one-time import

The `.readthedocs.yaml` in the repo root is wired up; what's left is the
one-time import on readthedocs.org.

1. Sign in at <https://app.readthedocs.org/> with the GitHub account that
   owns [`random-walks/citeformer`](https://github.com/random-walks/citeformer).
2. **Import a project → import manually** (or use the GitHub integration
   if `random-walks` is connected).
   - Repository URL: `https://github.com/random-walks/citeformer`
   - Default branch: `main`
   - Documentation type: Sphinx
3. After the first build succeeds, enable the **Automation rule** that
   activates new tags (so `v0.1.1`, `v0.1.2`, … appear under the version
   picker without manual intervention).
4. Confirm the URL resolves: <https://citeformer.readthedocs.io/en/latest/>.

Once live, the RTD badge in the README (which currently points at an
unimported slug) will start rendering the real build status.
