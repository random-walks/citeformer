"""Unit tests for the core types in `citeformer.core` and the §10.3 verify schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from citeformer import (
    Citation,
    CitationSupport,
    GenerationResult,
    Policy,
    Reference,
    Source,
    VerificationReport,
)

# --- Policy -------------------------------------------------------------------


def test_policy_values_match_contract_10_1() -> None:
    assert {p.value for p in Policy} == {"required", "quotes_only", "auto"}
    assert Policy.REQUIRED.value == "required"
    assert Policy("quotes_only") is Policy.QUOTES_ONLY


# --- Source -------------------------------------------------------------------


def test_source_basic() -> None:
    src = Source(
        metadata={"id": "poe-raven", "type": "book", "title": "The Raven"},
        content="Once upon a midnight dreary...",
    )
    assert src.metadata["title"] == "The Raven"
    assert src.content.startswith("Once")


def test_source_is_frozen() -> None:
    src = Source(metadata={"id": "a"}, content="c")
    with pytest.raises(ValidationError):
        src.content = "changed"  # type: ignore[misc]


def test_source_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Source.model_validate({"metadata": {}, "content": "c", "id": 1})


# --- Citation -----------------------------------------------------------------


def test_citation_defaults() -> None:
    c = Citation(span=(10, 13), source_id=1)
    assert c.verified is False
    assert c.entailment_score is None


def test_citation_source_id_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Citation(span=(0, 3), source_id=0)


def test_citation_is_frozen() -> None:
    c = Citation(span=(0, 3), source_id=1)
    with pytest.raises(ValidationError):
        c.verified = True  # type: ignore[misc]


# --- Reference ----------------------------------------------------------------


def test_reference_shape() -> None:
    ref = Reference(source_id=1, inline_marker="[1]", rendered="Poe, E. A. (1845).")
    assert ref.source_id == 1
    assert "Poe" in ref.rendered


def test_reference_rejects_source_id_zero() -> None:
    with pytest.raises(ValidationError):
        Reference(source_id=0, inline_marker="[0]", rendered="x")


# --- GenerationResult ---------------------------------------------------------


def test_generation_result_defaults() -> None:
    result = GenerationResult(text="plain text with no markers")
    assert result.schema_version == 1
    assert result.citations == []
    assert result.references == []


def test_generation_result_schema_version_is_1() -> None:
    """§10.3 contract — schema_version MUST be 1 in this release."""
    result = GenerationResult(text="x")
    assert result.schema_version == 1


def test_generation_result_roundtrip() -> None:
    result = GenerationResult(
        text="Quoth the raven [1].",
        citations=[Citation(span=(16, 19), source_id=1)],
        references=[Reference(source_id=1, inline_marker="[1]", rendered="Poe (1845).")],
    )
    reconstructed = GenerationResult.model_validate(result.model_dump(mode="json"))
    assert reconstructed == result


def test_generation_result_verify_raises_until_p6() -> None:
    """verify() is a stub in P1-P5; should raise with a clear message."""
    result = GenerationResult(text="x")
    with pytest.raises(NotImplementedError, match="P6"):
        result.verify()


# --- VerificationReport (§10.3 schema locked in P1) ---------------------------


def test_citation_support_shape() -> None:
    cs = CitationSupport(citation_index=0, entailment_score=0.92, supported=True)
    assert cs.supported is True
    assert 0.0 <= cs.entailment_score <= 1.0


def test_citation_support_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        CitationSupport(citation_index=0, entailment_score=1.5, supported=True)


def test_verification_report_defaults() -> None:
    report = VerificationReport(support_rate=1.0)
    assert report.schema_version == 1
    assert report.per_citation == []
    assert report.uncited_but_entailed == []


def test_verification_report_schema_version_is_1() -> None:
    """§10.3 contract — schema_version MUST be 1."""
    report = VerificationReport(support_rate=0.5)
    assert report.schema_version == 1
