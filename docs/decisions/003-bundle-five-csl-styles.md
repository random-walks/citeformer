# ADR-003 — Bundle five CSL styles; accept user paths; drop Vancouver from v0.1

- **Status**: Accepted (P3, 2026-04-23).
- **Likely superseded by**: [ADR-004](004-citeproc-rewrite.md) — once the home-grown formatter lands, Vancouver returns and "bundled" changes meaning.

## Context

citeproc-py ships only `harvard1.csl`. The broader `citation-style-language/styles` repo has 10,000+ styles; `citeproc-py-styles` is a separate PyPI package that ships them all.

We need:

- Determinism. The `.csl` files must be *pinned* so a citeformer version always produces the same reference output.
- Minimal install size. Shipping 10,000 styles is ~150 MB of XML — prohibitive for a rendering default.
- Coverage of the canonical five English-speaking academic styles: APA, MLA, Chicago, IEEE, Vancouver.

## Decision

Bundle five CSL files directly under `src/citeformer/render/`, pinned to commit `77bcd6d` of the upstream styles repo (2026-04-23):

- `apa.csl` (APA 7) — author-date.
- `modern-language-association.csl` (MLA 9) — author.
- `chicago-author-date.csl` — author-date.
- `ieee.csl` — numeric.
- `nature.csl` — numeric. **Replaces Vancouver** in this tier: the upstream repo has no canonical bare `vancouver.csl`; all Vancouver variants are institution-specific. Nature is numeric and universally recognized.

The aliases `"apa"` / `"apa-7"`, `"mla"` / `"mla-9"`, `"chicago"` / `"chicago-author-date"` resolve to the same files.

Users wanting any other style pass an absolute path to a `.csl` file; `resolve_style_path` accepts both bundled names and paths.

## Consequences

- `pip install citeformer` gives you a five-style working set immediately with no extra downloads.
- Vancouver support in v0.1 is *via a user-supplied CSL file* — not bundled. We document this limitation in the release notes.
- Refreshing the bundled styles is a deliberate act: bump the pin comment in `src/citeformer/render/styles.py`, regenerate the `render_snapshot_*` tests, call it out in the CHANGELOG. Prevents silent churn in rendered output.
- Adding a new bundled style means: download `.csl` → add to force-include list in `pyproject.toml` → add alias in `_BUNDLED_FILENAMES` → add snapshot test. A repeatable procedure, but each addition is a conscious choice.
