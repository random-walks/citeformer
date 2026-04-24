"""§10.3 contract tests — pin the JSON-serialized shape of GenerationResult + VerificationReport.

These tests snapshot a canonical instance (not the full pydantic JSON schema, which
can drift across pydantic minor versions for harmless internal reasons). A shape
change that renames, removes, or adds a field fails here until the snapshot is
regenerated — which must only happen as part of the ceremony described in
`docs/reference/contracts.md`.
"""

from __future__ import annotations

from citeformer import (
    Citation,
    CitationSupport,
    GenerationResult,
    Reference,
    Source,
    VerificationReport,
)
from citeformer.verify import UncitedClaim


def test_generation_result_canonical_snapshot(data_regression) -> None:  # type: ignore[no-untyped-def]
    """§10.3 — snapshot the JSON serialization of a canonical GenerationResult."""
    result = GenerationResult(
        text="Poe's Raven opens with mystery [1]. Melville begins with identity [2].",
        citations=[
            Citation(
                span=(31, 34),
                source_id=1,
                verified=False,
                entailment_score=None,
            ),
            Citation(
                span=(66, 69),
                source_id=2,
                verified=True,
                entailment_score=0.88,
            ),
        ],
        references=[
            Reference(
                source_id=1,
                inline_marker="[1]",
                rendered="Poe, E. A. (1845). The Raven.",
            ),
            Reference(
                source_id=2,
                inline_marker="[2]",
                rendered="Melville, H. (1851). Moby-Dick.",
            ),
        ],
        sources=[
            Source(
                metadata={"id": "poe", "type": "book", "title": "The Raven"},
                content="Once upon a midnight dreary...",
            ),
            Source(
                metadata={"id": "mel", "type": "book", "title": "Moby-Dick"},
                content="Call me Ishmael...",
            ),
        ],
    )
    data_regression.check(result.model_dump(mode="json"))


def test_verification_report_canonical_snapshot(data_regression) -> None:  # type: ignore[no-untyped-def]
    """§10.3 — snapshot the JSON serialization of a canonical VerificationReport."""
    report = VerificationReport(
        support_rate=0.5,
        citations_checked=2,
        per_citation=[
            CitationSupport(citation_index=0, entailment_score=0.32, supported=False),
            CitationSupport(citation_index=1, entailment_score=0.88, supported=True),
        ],
        uncited_but_entailed=[
            UncitedClaim(span=(10, 42), candidate_source_id=2, entailment_score=0.71),
            UncitedClaim(span=(100, 140), candidate_source_id=1, entailment_score=0.64),
        ],
    )
    data_regression.check(report.model_dump(mode="json"))


def test_generation_result_schema_version_is_2() -> None:
    """Belt-and-suspenders: even if snapshots are regenerated carelessly, this asserts
    the contract version number explicitly.
    """
    result = GenerationResult(text="x")
    assert result.schema_version == 2


def test_verification_report_schema_version_is_3() -> None:
    """VerificationReport schema_version bumped to 3 when `citations_checked`
    was added. See CHANGELOG for the additive-field ceremony.
    """
    report = VerificationReport(support_rate=1.0)
    assert report.schema_version == 3
    # Default is 0 — the honest "no citations to check" signal.
    assert report.citations_checked == 0
