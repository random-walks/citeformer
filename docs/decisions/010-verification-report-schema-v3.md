# ADR-010 — `VerificationReport` gains `citations_checked` (schema v3)

- **Status**: Accepted (2026-04-23).
- **Supersedes parts of**: [ADR-008](008-generation-result-schema-v2.md) on
  the specific point of "how consumers tell `no citations` apart from
  `all citations supported`".

## Context

`VerificationReport.support_rate` is defined as "fraction of citations with
``supported == True``, in [0, 1]." To keep the type simple (a plain
``float`` rather than ``float | None``), we also defined it as ``1.0`` when
``per_citation`` is empty — the vacuous "no unsupported claims exist"
reading.

In practice this makes benchmark charts misleading: runs where the baseline
emits zero citations (common on non-instruct models like gpt2, and on small
instruct models with short budgets) show ``support_rate = 100%``, next to
runs where citeformer emitted many citations and genuinely scored ~30% of
them as supported. Reader interpretation is "baseline is perfect, citeformer
is mid", which inverts the actual story.

Options considered:

1. **Change `support_rate` semantics to 0.0 when empty.** Breaks the
   vacuous-entailment reading and makes "no citations" look like "all
   claims unsupported" — also misleading, just in a different direction.
2. **Make `support_rate` `float | None`.** Honest but breaks anyone
   reading the field as a float. Requires a major bump on a field that
   downstream tooling probably treats as a primitive.
3. **Keep `support_rate = 1.0` as the semantic default, add a sibling
   field so consumers can detect the "nothing to check" case
   explicitly.** Additive / minor change.

## Decision

Take option 3. Add ``citations_checked: int`` to `VerificationReport`.

- ``citations_checked == len(per_citation)`` by construction — literally
  the count of citations the verifier scored.
- ``citations_checked == 0`` is the honest "no citations existed" signal.
  Consumers that report aggregate ``support_rate`` numbers should gate on
  ``citations_checked > 0`` to avoid publishing "100% supported" for
  zero-citation runs.
- ``support_rate`` itself unchanged — still ``1.0`` for empty
  ``per_citation`` (backward-compat with anything reading it as
  ``>= threshold`` boolean).

Schema version: **2 → 3**. Additive/minor per the
[§10.3 ceremony](../reference/contracts.md): a new optional field with a
default. Snapshot test in
``tests/integration/test_schemas.py`` regenerated with the new field;
``test_verification_report_schema_version_is_3`` pins the version explicitly.

## Consequences

- **benchmarks/plot.py** can now filter out "no cites" entries instead of
  averaging them into misleadingly-high support rates. (Applied in the
  same PR as this ADR.)
- **External consumers** that deserialize `VerificationReport` from JSON
  need to be aware that ``schema_version`` is 3 now. The extra field has
  a default of 0, so older JSON deserializes fine (pydantic populates the
  default), but comparisons against a pinned schema may flag the bump.
- **No breaking change** to the ``support_rate`` semantics — the
  "1.0 when empty" quirk stays for now. Callers that want strict
  semantics write ``report.citations_checked > 0 and report.support_rate
  >= threshold``.
- **CHANGELOG ``Contracts (§10)``** entry documents the additive field.

## Follow-on work

- ``benchmarks/plot.py`` annotates zero-cite bars as ``n/a`` rather than
  drawing a full-height "100%" bar.
- ``benchmarks/_common.py::analyze_run`` now includes ``citations_checked``
  in the row it emits to the sweep JSON log (tracked via the existing
  ``baseline_cites`` / ``constrained_cites`` fields, so no separate
  plumbing).
