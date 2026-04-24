# §10 Contracts

Three invariants govern citeformer's versioning. Breaking any one of them = **major** version bump. Additive changes = **minor**. Everything else = **patch**.

Before editing `src/citeformer/grammar/builder.py`, `src/citeformer/core.py`, `src/citeformer/verify/report.py`, or any pydantic model with a `schema_version` field — run `/contract-check` on your diff.

## §10.1 — Citation marker grammar

The citation marker rule is fixed in [GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) format (xgrammar / llama.cpp native) as:

```
cite-id ::= <open> <digits> <close>
```

where `<digits>` is dynamically constrained at decode time to the enum of in-scope source indices (`"1" | "2" | ... | "N"` for `N = len(sources)`) and `<open>`/`<close>` are the delimiters of the chosen [`MarkerStyle`](../decisions/011-configurable-marker-styles.md) (default `"[" ... "]"`; also `( )`, `{ }`, `^`). The implementation lives in [`src/citeformer/grammar/builder.py`](../../src/citeformer/grammar/builder.py); the emitted grammar is consumed by XGrammar's `compile_grammar()` in [`src/citeformer/backends/hf.py`](../../src/citeformer/backends/hf.py) and by the other local backends.

Policy-level rules layer on top of this rule:

- `required` — every sentence must end with at least one `cite-group`.
- `quotes_only` — only quoted spans (`"..."`) require a `cite-group`.
- `auto` — `cite-group` is optional at any position; missing citations are surfaced by `verify()` coverage checks instead of rejected at decode time.

The **load-bearing invariant** is the digit enum — `("1" | ... | "N")` for `N = len(sources)`. Any change that admits a digit outside that range, changes the set of policies, or alters the semantics of an existing policy is a **§10.1 break** and bumps major. Adding a new delimiter shape (ADR-011 did this) is additive / minor — the digit enum invariant holds across all shapes. Switching the grammar DSL (e.g. GBNF → something else) is implementation-level — minor/patch as long as marker-rule semantics are unchanged — but it still requires regenerating the snapshot and calling out the change in CHANGELOG.

Regression snapshot: `tests/unit/test_grammar_builder.py` pins the GBNF grammar serialized for a representative source set across each policy.

## §10.2 — `Source.metadata` shape

`Source.metadata` is a [CSL-JSON item](https://github.com/citation-style-language/schema/blob/master/csl-data.json): `{id, type, author, title, issued, container-title, DOI, URL, ...}`. Our home-grown formatters consume this shape directly; it's also compatible with external CSL tooling (`citeproc-py`, Pandoc, Zotero) if you need to plug it in downstream.

- Adding optional fields = **additive / minor**.
- Renaming or removing fields = **breaking / major**.
- Changing the required set (e.g. making `type` optional) = **breaking / major**.

Regression snapshot: `tests/unit/test_csl_rendering.py` pins canonical renderings of the four core CSL types × six styles, and `tests/unit/test_csl_suite.py` pins an expanded **50-case CSL fixture × 6 formatters = 300 snapshots** covering every mapped CSL 1.0 item type and common edge cases (Unicode surnames, particles, et al. threshold, missing year, page ranges, DOI rendering). A change in CSL shape that reshuffles the fixture output is caught there.

## §10.3 — Output schemas

Both public output models carry a `schema_version` field and are pinned by snapshot tests:

- `citeformer.core.GenerationResult` — `schema_version: 2`. Bumped from 1 in [ADR-008](../decisions/008-generation-result-schema-v2.md). Pinned by `tests/integration/test_schemas.py`.
- `citeformer.verify.report.VerificationReport` — `schema_version: 3`. Bumped to 2 when verify() landed ([ADR-008](../decisions/008-generation-result-schema-v2.md)), then to 3 to add the `citations_checked` field ([ADR-010](../decisions/010-verification-report-schema-v3.md)). Pinned by `tests/integration/test_schemas.py`.

Adding / removing / renaming any field in either model requires bumping `schema_version` in the owning pydantic model **and** calling it out in the PR description (see the "Invariant touched?" section in the PR template).

## Ceremony summary

| Change | Bump | Snapshot regen | PR label |
|---|---|---|---|
| Fix typo in a docstring | patch | no | — |
| Add a new optional CSL metadata field we pass through | minor | regenerate §10.2 snapshots | `contracts:additive` |
| Rename `GenerationResult.text` → `GenerationResult.body` | major | regenerate §10.3 snapshots; bump `schema_version` | `contracts:breaking` |
| Add a fourth citation policy (e.g. `numeric_only`) | minor | regenerate §10.1 snapshot for the new policy | `contracts:additive` |
| Add a new marker shape alongside `[N]` (e.g. `(N)`) via `MarkerStyle` | minor | regenerate §10.1 snapshot for the new variant | `contracts:additive` |
| Change the **default** marker shape from `[N]` to something else | major | regenerate every §10.1 snapshot | `contracts:breaking` |
| Remove `[N]` as a supported shape entirely | major | regenerate §10.1 snapshot | `contracts:breaking` |
