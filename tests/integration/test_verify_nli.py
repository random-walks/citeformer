"""Integration test for the real NLI backend.

Loads a small DeBERTa-v3-MNLI variant (~180 MB) and verifies that the
entailment / coverage flow produces sensible scores on pairs with clear
semantic relationships. Marked ``integration`` so the default ``pytest``
run skips it.

Uses ``cross-encoder/nli-deberta-v3-base`` (~180 MB) to keep CI runtime
bounded. Same label ordering as the default (entailment / neutral /
contradiction); quality is lower than the large variant but these
assertions are lax enough to pass either way. Users in production should
stick with the default ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-
ling-wanli``.
"""

from __future__ import annotations

import pytest

from citeformer import Citation, GenerationResult, Source
from citeformer.verify import NLIModel, Verifier

_SMALL_MODEL = "cross-encoder/nli-deberta-v3-base"


@pytest.fixture(scope="module")
def nli_model() -> NLIModel:
    """Load the small DeBERTa model once for the whole module."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    return NLIModel(model_name=_SMALL_MODEL, batch_size=4)


@pytest.mark.integration
def test_nli_model_entailment_ordering(nli_model: NLIModel) -> None:
    """Semantically-related pairs should score entailment higher than
    orthogonal ones — a basic sanity check that the model loaded correctly
    and the label mapping is right.
    """
    supported = nli_model.entail(
        premise="The Transformer architecture uses self-attention to process tokens.",
        hypothesis="Transformers rely on self-attention.",
    )
    orthogonal = nli_model.entail(
        premise="The Transformer architecture uses self-attention to process tokens.",
        hypothesis="Bananas are yellow fruits.",
    )
    assert supported.entailment > orthogonal.entailment
    assert supported.entailment > 0.5


@pytest.mark.integration
def test_verifier_end_to_end_with_real_nli(nli_model: NLIModel) -> None:
    """Full Verifier flow with the real NLI model on a clear fixture.

    Two sources, two cited claims — one genuinely supported, one orthogonal.
    We expect one ``supported=True`` and one ``supported=False``.
    """
    sources = [
        Source(
            metadata={"id": "att", "type": "article-journal", "title": "Attention"},
            content=(
                "The Transformer architecture relies entirely on self-attention "
                "mechanisms, dispensing with recurrence and convolutions entirely."
            ),
        ),
        Source(
            metadata={"id": "bert", "type": "article-journal", "title": "BERT"},
            content=(
                "BERT is designed to pre-train deep bidirectional representations "
                "by masked language modeling."
            ),
        ),
    ]
    text = "Transformers rely on self-attention [1]. The capital of France is Paris [2]."
    citations = [
        Citation(span=(text.index("[1]"), text.index("[1]") + 3), source_id=1),
        Citation(span=(text.index("[2]"), text.index("[2]") + 3), source_id=2),
    ]

    verifier = Verifier(threshold=0.5, nli=nli_model)
    report = verifier.verify(
        text=text,
        citations=citations,
        sources=sources,
        run_coverage=False,
    )

    # First citation about self-attention → entailed by source 1.
    # Second citation about France/Paris → not entailed by source 2 (BERT).
    assert report.per_citation[0].supported is True
    assert report.per_citation[0].entailment_score > 0.5
    assert report.per_citation[1].supported is False
    assert report.per_citation[1].entailment_score < 0.5
    assert report.support_rate == 0.5


@pytest.mark.integration
def test_verifier_coverage_with_real_nli(nli_model: NLIModel) -> None:
    """Coverage check should flag an uncited claim that a source entails."""
    sources = [
        Source(
            metadata={"id": "att", "type": "article-journal", "title": "Attention"},
            content=(
                "The Transformer architecture relies entirely on self-attention "
                "mechanisms, dispensing with recurrence and convolutions entirely."
            ),
        ),
    ]
    # Uncited sentence that source 1 clearly entails.
    text = "The Transformer uses self-attention. But we added no citation."
    citations: list[Citation] = []

    verifier = Verifier(threshold=0.5, nli=nli_model)
    report = verifier.verify(text=text, citations=citations, sources=sources)

    # At least one sentence should be flagged. The first ("The Transformer uses
    # self-attention.") should match source 1.
    assert len(report.uncited_but_entailed) >= 1
    assert any(uc.candidate_source_id == 1 for uc in report.uncited_but_entailed)


@pytest.mark.integration
def test_generation_result_verify_end_to_end(nli_model: NLIModel) -> None:
    """Confirms `GenerationResult.verify(nli=...)` works against a real model.

    Uses a longer premise so the base-sized NLI model has enough signal to
    cross the 0.5 entailment threshold; short premises can score below
    threshold even on obviously-related pairs.
    """
    sources = [
        Source(
            metadata={"id": "att", "type": "article-journal", "title": "Attention"},
            content=(
                "The Transformer is the first transduction model relying entirely "
                "on self-attention to compute representations of its input and "
                "output without using sequence-aligned RNNs or convolution."
            ),
        ),
    ]
    # Keep the hypothesis close to the premise wording — the base-sized
    # cross-encoder NLI is strict about added claims. "Rely on self-attention"
    # is faithful; "use self-attention to process tokens" is interpreted as
    # adding a claim about token-level processing.
    text = "Transformers rely on self-attention [1]."
    result = GenerationResult(
        text=text,
        citations=[Citation(span=(text.index("[1]"), text.index("[1]") + 3), source_id=1)],
        sources=sources,
    )
    report = result.verify(nli=nli_model, run_coverage=False)
    assert report.per_citation[0].supported is True
    assert report.per_citation[0].entailment_score > 0.5
