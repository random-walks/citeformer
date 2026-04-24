"""Tests for the ALCE-flavoured metric helpers.

The subset runner itself loads a real HF model and is covered by the
integration tier (`make test-integration`). These unit tests exercise
the pure-function metric helpers using a stubbed NLI scorer so the
math is verified without any model load.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from benchmarks.alce_subset import (
    _citation_precision,
    _citation_recall,
    _cites_in,
    _fabrication_rate,
    _sentences,
)
from citeformer import Source


@dataclass
class _FakeNLIResult:
    entailment: float


class _FakeNLI:
    """Returns pre-seeded scores in order of arrival."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)

    def entail_batch(self, pairs: list[tuple[str, str]]) -> list[_FakeNLIResult]:
        assert len(pairs) <= len(self._scores), (
            f"stub has {len(self._scores)} scores, got {len(pairs)} pairs"
        )
        return [_FakeNLIResult(s) for s in self._scores[: len(pairs)]]


@pytest.fixture
def sources() -> list[Source]:
    return [
        Source(metadata={"id": "a"}, content="Source one body."),
        Source(metadata={"id": "b"}, content="Source two body."),
        Source(metadata={"id": "c"}, content="Source three body."),
    ]


# --- Sentence splitter -------------------------------------------------------


def test_sentences_splits_on_terminators() -> None:
    text = "First. Second! Third? Fourth."
    assert _sentences(text) == ["First.", "Second!", "Third?", "Fourth."]


def test_sentences_strips_empty_tail() -> None:
    assert _sentences("   Only one.   ") == ["Only one."]


# --- Cite extractor ----------------------------------------------------------


def test_cites_in_extracts_bracketed_ints() -> None:
    assert _cites_in("Claim [1] and [2]") == [1, 2]


def test_cites_in_preserves_order_and_duplicates() -> None:
    assert _cites_in("[3] [1] [3] [2]") == [3, 1, 3, 2]


def test_cites_in_ignores_non_integer_brackets() -> None:
    """The `\\[(\\d+)\\]` pattern requires the ENTIRE bracket body to be digits."""
    assert _cites_in("Ref [ibid] or [fig. 3] but cite [1]") == [1]


# --- Citation recall ---------------------------------------------------------


def test_citation_recall_all_cited() -> None:
    sentences = ["One [1].", "Two [2]."]
    assert _citation_recall(sentences) == 1.0


def test_citation_recall_none_cited() -> None:
    sentences = ["One.", "Two."]
    assert _citation_recall(sentences) == 0.0


def test_citation_recall_half_cited() -> None:
    sentences = ["One [1].", "Two.", "Three [2].", "Four."]
    assert _citation_recall(sentences) == 0.5


def test_citation_recall_empty_sentences_returns_zero() -> None:
    assert _citation_recall([]) == 0.0


# --- Fabrication rate --------------------------------------------------------


def test_fabrication_rate_no_cites() -> None:
    assert _fabrication_rate([], n_sources=3) == 0.0


def test_fabrication_rate_all_in_scope() -> None:
    assert _fabrication_rate([1, 2, 3], n_sources=3) == 0.0


def test_fabrication_rate_mixed() -> None:
    assert _fabrication_rate([1, 5, 3, 7], n_sources=3) == pytest.approx(0.5)


def test_fabrication_rate_all_out_of_scope() -> None:
    assert _fabrication_rate([5, 6, 7], n_sources=3) == 1.0


# --- Citation precision (NLI-mocked) -----------------------------------------


def test_citation_precision_all_entail(sources: list[Source]) -> None:
    # Two sentences each citing one source → two pairs → both entail.
    sentences = ["First claim [1].", "Second claim [2]."]
    nli = _FakeNLI([0.99, 0.95])
    prec = _citation_precision(sentences, sources, nli=nli, threshold=0.5)
    assert prec == 1.0


def test_citation_precision_half_entail(sources: list[Source]) -> None:
    sentences = ["First claim [1].", "Second claim [2]."]
    nli = _FakeNLI([0.99, 0.10])
    prec = _citation_precision(sentences, sources, nli=nli, threshold=0.5)
    assert prec == 0.5


def test_citation_precision_threshold_matters(sources: list[Source]) -> None:
    """A middle score clears 0.3 but not 0.8."""
    sentences = ["Claim [1]."]
    nli = _FakeNLI([0.55])
    assert _citation_precision(sentences, sources, nli=nli, threshold=0.3) == 1.0
    assert _citation_precision(sentences, sources, nli=nli, threshold=0.8) == 0.0


def test_citation_precision_no_cites_is_vacuous(sources: list[Source]) -> None:
    """Nothing to score → precision is 1.0 by convention, not a division error."""
    sentences = ["Uncited claim.", "Another uncited claim."]
    nli = _FakeNLI([])  # Won't be called.
    prec = _citation_precision(sentences, sources, nli=nli, threshold=0.5)
    assert prec == 1.0


def test_citation_precision_skips_out_of_scope(sources: list[Source]) -> None:
    """Fabricated cites (id > N) don't get scored — fabrication_rate handles them."""
    sentences = ["Claim [7]."]
    nli = _FakeNLI([])
    prec = _citation_precision(sentences, sources, nli=nli, threshold=0.5)
    assert prec == 1.0  # No in-scope cites → vacuous.
