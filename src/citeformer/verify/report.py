"""`VerificationReport` pydantic schema — §10.3 contract.

The `schema_version` is bumped any time the model's shape changes. Current
version: **3** — added `citations_checked` so downstream tooling can tell
"no citations" apart from "all citations supported" without probing
`per_citation`. The old ``support_rate = 1.0 when per_citation is empty``
convention is preserved for continuity; consumers that want the
honest "there's nothing to rate" signal read `citations_checked` now.

§10.3 ceremony: every change goes through the `release-bump` rubric, the
snapshot tests in `tests/integration/test_schemas.py`, and a CHANGELOG
``Contracts (§10)`` note. See ``docs/reference/contracts.md``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CitationSupport(BaseModel):
    """Entailment detail for a single citation.

    Attributes:
        citation_index: Position in `GenerationResult.citations` this entry describes.
        entailment_score: NLI entailment probability in [0, 1]. Computed by the
            configured NLI model (DeBERTa-v3-large-MNLI by default).
        supported: `True` iff `entailment_score >= threshold` AND the
            citation's source_id resolves to an in-range source. Threshold
            defaults to 0.5; configurable on the `Verifier`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_index: int = Field(
        ge=0,
        description="Index into GenerationResult.citations this entry describes.",
    )
    entailment_score: float = Field(
        ge=0.0,
        le=1.0,
        description="NLI entailment probability.",
    )
    supported: bool = Field(
        description="True iff entailment_score >= threshold and source_id is in range.",
    )


class UncitedClaim(BaseModel):
    """An uncited sentence that NLI flags as likely needing a citation.

    Attributes:
        span: ``(start, end)`` char offsets of the sentence in
            `GenerationResult.text`.
        candidate_source_id: 1-indexed source that most strongly entails the
            sentence — suggested citation target if the author wanted to fix
            the gap.
        entailment_score: Entailment probability of the best-matching
            source, in [0, 1].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    span: tuple[int, int] = Field(
        description="(start, end) char offsets of the uncited sentence.",
    )
    candidate_source_id: int = Field(
        ge=1,
        description="1-indexed source that best entails the sentence.",
    )
    entailment_score: float = Field(
        ge=0.0,
        le=1.0,
        description="NLI entailment probability of the best-matching source.",
    )


class VerificationReport(BaseModel):
    """Output of `GenerationResult.verify()`.

    §10.3 contract: shape is locked by `tests/integration/test_schemas.py`.

    Attributes:
        schema_version: Contract version. Bump on any shape change.
        support_rate: Fraction of citations with `supported == True`, in [0, 1].
            Kept at 1.0 when `per_citation` is empty for backward compat, but
            consumers reporting aggregate numbers should gate on
            `citations_checked > 0` to avoid publishing "100% supported" for
            runs that emitted no citations at all.
        citations_checked: Count of citations the verifier actually scored
            (equals ``len(per_citation)``). ``0`` is the honest signal that
            the run produced no citations for NLI to check — not that
            everything was supported.
        per_citation: One `CitationSupport` entry per citation in the
            `GenerationResult`, in the same order.
        uncited_but_entailed: Sentences where the NLI coverage check flagged
            a missing citation — an uncited claim that one of the available
            sources would entail. Each entry carries the span, the best-
            matching source, and the entailment score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(
        default=3,
        description="§10.3 contract version. Bumped on any shape change.",
    )
    support_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of citations with supported=True.",
    )
    citations_checked: int = Field(
        default=0,
        ge=0,
        description=(
            "Count of citations the verifier scored; 0 means 'no citations "
            "existed', not 'all supported'. Added in schema v3."
        ),
    )
    per_citation: list[CitationSupport] = Field(
        default_factory=list,
        description="Entailment detail for each citation.",
    )
    uncited_but_entailed: list[UncitedClaim] = Field(
        default_factory=list,
        description="Sentences flagged as missing a citation.",
    )
