"""Top-level verification orchestrator.

Composes the three checks (existence, entailment, coverage) into one
`VerificationReport`. Users typically reach this via
`GenerationResult.verify(sources=...)` rather than constructing `Verifier`
directly, but the class is public so advanced users can plug in a custom
`NLIModel` or skip the NLI paths entirely.
"""

from __future__ import annotations

from typing import Any

from citeformer.core import Citation, Source
from citeformer.verify.coverage import find_uncited_but_entailed
from citeformer.verify.entailment import score_entailment
from citeformer.verify.existence import check_existence
from citeformer.verify.nli import NLIModel
from citeformer.verify.report import CitationSupport, UncitedClaim, VerificationReport
from citeformer.verify.sentences import split_sentences


class Verifier:
    """Runs the three verification checks against a `GenerationResult`.

    Attributes:
        threshold: Entailment probability above which a citation is
            ``supported`` and an uncited sentence is ``flagged``.
        nli: The NLI backend. If ``None``, a default `NLIModel` is created
            lazily on first `verify()` call.
    """

    threshold: float
    nli: NLIModel | None

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        nli: NLIModel | None = None,
    ) -> None:
        """Construct a `Verifier`.

        Args:
            threshold: Entailment cutoff for ``supported`` / flagging.
            nli: Preconstructed NLI model, or None to build a default on
                first use.
        """
        self.threshold = threshold
        self.nli = nli

    def verify(
        self,
        text: str,
        citations: list[Citation],
        sources: list[Source],
        *,
        run_coverage: bool = True,
        **_options: Any,
    ) -> VerificationReport:
        """Run the full verification pipeline.

        Args:
            text: The generated text (``GenerationResult.text``).
            citations: Citations emitted in generation.
            sources: Sources that were in scope.
            run_coverage: If False, skip the NLI coverage check entirely.
                Useful on REQUIRED policy where grammar guarantees every
                sentence has a cite.

        Returns:
            A fully-populated `VerificationReport`.
        """
        existence = check_existence(citations, sources)
        sentence_spans = split_sentences(text)

        if not citations and not (run_coverage and sources):
            # Nothing to check — every citation is supported vacuously.
            return VerificationReport(
                support_rate=1.0,
                per_citation=[],
                uncited_but_entailed=[],
            )

        nli = self.nli if self.nli is not None else NLIModel()

        per_citation: list[CitationSupport]
        if citations:
            per_citation = score_entailment(
                citations=citations,
                sentence_spans=sentence_spans,
                sources=sources,
                nli=nli,
                threshold=self.threshold,
            )
            # Out-of-range citations override `supported` → False regardless
            # of the NLI score, to keep existence and entailment aligned.
            if existence.missing:
                missing = set(existence.missing)
                per_citation = [
                    (
                        CitationSupport(
                            citation_index=cs.citation_index,
                            entailment_score=cs.entailment_score,
                            supported=False,
                        )
                        if citations[cs.citation_index].source_id in missing
                        else cs
                    )
                    for cs in per_citation
                ]
        else:
            per_citation = []

        uncited: list[UncitedClaim] = []
        if run_coverage:
            uncited = find_uncited_but_entailed(
                citations=citations,
                sentence_spans=sentence_spans,
                sources=sources,
                nli=nli,
                threshold=self.threshold,
            )

        support_rate = (
            sum(1 for cs in per_citation if cs.supported) / len(per_citation)
            if per_citation
            else 1.0
        )

        return VerificationReport(
            support_rate=support_rate,
            per_citation=per_citation,
            uncited_but_entailed=uncited,
        )
