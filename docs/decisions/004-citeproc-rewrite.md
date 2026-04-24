# ADR-004 — Replace citeproc-py with a home-grown formatter

- **Status**: Accepted and implemented (2026-04-23).

## Context

P3 shipped rendering via `citeproc-py`. It works but has accumulated friction:

- **Chicago page-range bug**: `UnboundLocalError` in citeproc-py's `minimal-two` page-range formatter for multi-page citations. We worked around by using single-page test data.
- **APA double-period**: `"Poe, E. A.. (1845). …"` — citeproc-py appends a period after the given-name initials regardless of existing punctuation.
- **Noisy warnings**: Every render prints `UserWarning` about unsupported CSL-JSON fields (`indexed`, `reference-count`, …) that Crossref's transform response includes.
- **Vancouver gap**: no canonical `vancouver.csl` in the upstream styles repo (see [ADR-003](003-bundle-five-csl-styles.md)). Implementing Vancouver ourselves is trivial; translating a CSL style file for it isn't.
- **Maintenance concentration**: citeproc-py's CSL test suite passes at ~60%; a volunteer team of three maintains the project. Our ability to fix edge cases upstream is limited.
- **Home-grown feel**: the rest of citeformer is small, owned code. The render layer being an external lib with known quirks is inconsistent with the rest of the library's ethos.

## Decision

Rewrite `citeformer.render` to a home-grown formatter. Specifically:

- Remove the `citeproc-py` dependency entirely — both main deps and the initial
  `citeproc-compat` extra. Users who want the "any of 10,000 CSL files"
  escape hatch install `citeproc-py` themselves and feed it our
  CSL-JSON-compliant `Source.metadata`; no shim needed from us.
- Implement six styles procedurally in Python: APA 7, MLA 9, Chicago (author-date), IEEE, Nature, **and Vancouver**. Each style is a `CitationFormatter` subclass with `inline(item: CSLItem, number: int) -> str` and `bibliography(item: CSLItem, number: int) -> str` methods.
- The styles still consume **CSL-JSON** as the input shape (`Source.metadata`). §10.2 doesn't change. Users can keep feeding us Crossref / arXiv output verbatim.
- Author a `.claude/skills/add-citation-format/SKILL.md` that documents the exact template, test matrix, and edge cases to cover when implementing a new style. Adding a seventh style becomes a 30-minute skill-driven task.

## Consequences

- Full control over output. Chicago page-range crash, APA double-period, noisy warnings — all fixed by construction. The six built-ins go through zero citeproc-py code paths.
- Vancouver joined the bundle (six styles total: APA 7, MLA 9, Chicago author-date, IEEE, Nature, Vancouver).
- Initially we shipped a `citeproc-compat` extra as a placeholder for a future
  compat module, but since it had no implementation we removed it to keep the
  promised surface honest. Anyone wanting arbitrary CSL files today can
  `pip install citeproc-py` themselves — `Reference` / `Source.metadata` stay
  CSL-JSON compliant.
- Loss of "10,000 styles for free" out-of-the-box. Mitigated by the
  `add-citation-format` skill at `.claude/skills/add-citation-format/SKILL.md`
  (adding a new style is a 30-minute skill-driven task) and by the option to
  drop down to `citeproc-py` directly.
- Implementation landed at `src/citeformer/render/formatters/` (one file per style; `_base.py` for shared helpers: `Author`, `parse_authors`, `parse_year`, `ensure_period`, `format_page_range`, `format_doi`, `get_str`, `get_title`). Total ~1,200 LOC across formatters, 60+ granular tests, 6 snapshot tests, 1 skill file.
- `citeproc-py` removed from main dependencies. Bundled `.csl` files deleted (`apa.csl`, `modern-language-association.csl`, `chicago-author-date.csl`, `ieee.csl`, `nature.csl`).
- Public-API changes: `load_style()` and `resolve_style_path()` are gone (they were citeproc-py-specific). Callers use `get_formatter(name)` to obtain a formatter object.
- §10.2 (`Source.metadata` as CSL-JSON) unchanged. §10.3 (output schemas) unchanged. Pure implementation-layer refactor from users' perspective.
- [ADR-003](003-bundle-five-csl-styles.md) marked superseded in this round.
