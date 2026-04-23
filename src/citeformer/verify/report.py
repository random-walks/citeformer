"""`VerificationReport` pydantic schema — §10.3 contract.

Shape is locked in P1 even though `verify()` itself isn't implemented until P6.
That's deliberate: locking the contract first means we can't accidentally
ship a v0.1 with verification fields users depend on and then reshape them later.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CitationSupport(BaseModel):
    """Entailment detail for a single citation.

    Attributes:
        citation_index: Position in `GenerationResult.citations` this entry describes.
        entailment_score: NLI entailment probability in [0, 1]. Computed by the
            configured NLI model (DeBERTa-v3-large-MNLI by default in P6).
        supported: `True` iff `entailment_score >= threshold`. Threshold defaults to
            0.5 but is configurable on the Citeformer instance.
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
        description="True iff entailment_score >= threshold.",
    )


class VerificationReport(BaseModel):
    """Output of `GenerationResult.verify()`.

    §10.3 contract: shape is locked by `tests/integration/test_schemas.py`.

    Attributes:
        schema_version: Contract version. Bump on any shape change.
        support_rate: Fraction of citations with `supported == True`, in [0, 1].
        per_citation: One `CitationSupport` entry per citation in the
            `GenerationResult`, in the same order.
        uncited_but_entailed: Sentence indices where the NLI coverage check flagged a
            missing citation — i.e. an uncited claim that one of the available sources
            would entail. Indices reference sentence spans in `GenerationResult.text`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(
        default=1,
        description="§10.3 contract version. Bumped on any shape change.",
    )
    support_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of citations with supported=True.",
    )
    per_citation: list[CitationSupport] = Field(
        default_factory=list,
        description="Entailment detail for each citation.",
    )
    uncited_but_entailed: list[int] = Field(
        default_factory=list,
        description="Sentence indices flagged as missing a citation.",
    )
