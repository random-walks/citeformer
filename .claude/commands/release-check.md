---
description: Dry-run preflight for a release. Zero side effects.
---

Run all the checks a release tag push would run, without pushing or committing anything.

## Steps

1. `make lint` — ruff check, ruff format check, mypy strict.
2. `make test` — full pytest suite.
3. `make docs-build` — Sphinx build with `-W` (warnings are errors).
4. `uv build` — builds sdist + wheel into `dist/`. Verifies the package is actually buildable.
5. `uv run python -c "import citeformer; print(citeformer.__version__)"` — sanity-check the installed version.
6. Grep for any lingering TODO / FIXME / XXX markers in `src/` and flag them (non-blocking, but worth noting for the release narrative).
7. Verify `CHANGELOG.md` has content under `[Unreleased]` (or has been freshly bumped to a new version).

## Output

Terse summary of pass/fail per step. On failure, say exactly which step failed and surface the relevant error output — don't hide it.

If everything passes: print `release-check: OK — version X.Y.Z ready`.

## No side effects

Do not commit, tag, push, or modify any tracked file. `dist/` is gitignored and safe to leave behind.
