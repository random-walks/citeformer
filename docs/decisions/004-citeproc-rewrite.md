# ADR-004 — Plan: replace citeproc-py with a home-grown formatter

- **Status**: Proposed (planned, 2026-04-23). Implementation lands in a dedicated phase after P5.

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

- Remove the `citeproc-py` dependency from `citeformer`'s main deps. Add it to an optional `citeproc-compat` extra for users who want the "any of 10,000 CSL files" escape hatch — that path remains available, but off by default.
- Implement six styles procedurally in Python: APA 7, MLA 9, Chicago (author-date), IEEE, Nature, **and Vancouver**. Each style is a `CitationFormatter` subclass with `inline(item: CSLItem, number: int) -> str` and `bibliography(item: CSLItem, number: int) -> str` methods.
- The styles still consume **CSL-JSON** as the input shape (`Source.metadata`). §10.2 doesn't change. Users can keep feeding us Crossref / arXiv output verbatim.
- Author a `.claude/skills/add-citation-format/SKILL.md` that documents the exact template, test matrix, and edge cases to cover when implementing a new style. Adding a seventh style becomes a 30-minute skill-driven task.

## Consequences

- Full control over output. We fix the Chicago / APA quirks by construction.
- Vancouver lands.
- `citeformer[citeproc-compat]` remains a clean path for users who want to use a random `.csl` file from the upstream 10,000 — but they pay the quirks cost themselves and know it.
- Loss of "10,000 styles for free" out-of-the-box. Mitigated by the compatibility extra.
- Effort estimate: ~1,500 LOC across six formatters + ~180 tests + the skill file. Bigger than any single prior P-phase. Scheduled as its own refactor commit with clear boundaries.
- §10.2 (CSL-JSON metadata shape) unchanged. §10.3 (output schemas) unchanged. This is a pure implementation-layer refactor.
- Cross-references to this ADR: [ADR-003](003-bundle-five-csl-styles.md) (bundled-style decision will be revisited with Vancouver back in scope).
