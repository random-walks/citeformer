# §10 Contracts

Three invariants govern citeformer's versioning. Breaking any one of them = **major** version bump. Additive changes = **minor**. Everything else = **patch**.

Before editing `src/citeformer/grammar/builder.py`, `src/citeformer/core.py`, `src/citeformer/verify/report.py`, or any pydantic model with a `schema_version` field — run `/contract-check` on your diff.

## §10.1 — Citation marker grammar

The citation marker rule is fixed in [GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) format (xgrammar / llama.cpp native) as:

```
cite-id ::= "[" <digits> "]"
```

where `<digits>` is dynamically constrained at decode time to the enum of in-scope source indices (`"1" | "2" | ... | "N"` for `N = len(sources)`). The implementation lives in [`src/citeformer/grammar/builder.py`](../../src/citeformer/grammar/builder.py); the emitted grammar is consumed by XGrammar's `compile_grammar()` in [`src/citeformer/backends/hf.py`](../../src/citeformer/backends/hf.py) and by the other backends as they land.

Policy-level rules layer on top of this rule:

- `required` — every sentence must end with at least one `cite-group`.
- `quotes_only` — only quoted spans (`"..."`) require a `cite-group`.
- `auto` — `cite-group` is optional at any position; missing citations are surfaced by `verify()` coverage checks instead of rejected at decode time.

Any change to the marker shape (e.g. moving to `(1)` or `{1}`), the set of policies, or the semantics of an existing policy is a **§10.1 break** and bumps major. Switching the grammar DSL (e.g. GBNF → something else) is implementation-level — minor/patch as long as marker-rule semantics are unchanged — but it still requires regenerating the snapshot and calling out the change in CHANGELOG.

Regression snapshot: `tests/unit/test_grammar_builder.py` pins the GBNF grammar serialized for a representative source set across each policy.

## §10.2 — `Source.metadata` shape

`Source.metadata` is a [CSL-JSON item](https://github.com/citation-style-language/schema/blob/master/csl-data.json): `{id, type, author, title, issued, container-title, DOI, URL, ...}`. Our home-grown formatters consume this shape directly; it's also compatible with external CSL tooling (`citeproc-py`, Pandoc, Zotero) if you need to plug it in downstream.

- Adding optional fields = **additive / minor**.
- Renaming or removing fields = **breaking / major**.
- Changing the required set (e.g. making `type` optional) = **breaking / major**.

Regression snapshot: `tests/unit/test_csl_rendering.py` — 20 fixture citations × 5 CSL styles = 100 rendered outputs pinned via `pytest-regressions`. A change in CSL shape that reshuffles the fixture output is caught there.

## §10.3 — Output schemas

Both public output models carry a `schema_version` field and are pinned by snapshot tests:

- `citeformer.core.GenerationResult` — `schema_version: 1`. Pinned by `tests/integration/test_generation_result_schema.py`.
- `citeformer.verify.report.VerificationReport` — `schema_version: 1`. Pinned by `tests/integration/test_verification_report_schema.py`.

Adding / removing / renaming any field in either model requires bumping `schema_version` in the owning pydantic model **and** calling it out in the PR description (see the "Invariant touched?" section in the PR template).

## Ceremony summary

| Change | Bump | Snapshot regen | PR label |
|---|---|---|---|
| Fix typo in a docstring | patch | no | — |
| Add a new optional CSL metadata field we pass through | minor | regenerate §10.2 snapshots | `contracts:additive` |
| Rename `GenerationResult.text` → `GenerationResult.body` | major | regenerate §10.3 snapshots; bump `schema_version` | `contracts:breaking` |
| Add a fourth citation policy (e.g. `numeric_only`) | minor | regenerate §10.1 snapshot for the new policy | `contracts:additive` |
| Change `cite-id` from `[N]` to `{N}` | major | regenerate §10.1 snapshot | `contracts:breaking` |
