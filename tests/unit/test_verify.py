"""Unit tests for the verification module.

Uses a mock NLI backend (no transformers load) so the full orchestration flow
runs in milliseconds. The integration test at
``tests/integration/test_verify_nli.py`` exercises the real DeBERTa-v3-MNLI
weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from citeformer import Citation, GenerationResult, Source
from citeformer.verify import (
    ExistenceResult,
    NLIResult,
    Verifier,
    check_existence,
    sentence_containing,
    split_sentences,
)
from citeformer.verify.nli import NLIModel

# --- Sentence splitter -------------------------------------------------------


def test_split_sentences_simple() -> None:
    spans = split_sentences("Hello world. Second sentence. Third.")
    assert [s.text for s in spans] == [
        "Hello world.",
        "Second sentence.",
        "Third.",
    ]
    assert [s.index for s in spans] == [0, 1, 2]


def test_split_sentences_honors_char_offsets() -> None:
    text = "First [1]. Second [2]."
    spans = split_sentences(text)
    # Spans should round-trip via `text[start:end]` to the sentence content.
    for s in spans:
        # Stripped sentence text is what split returns; the slice may include
        # the trailing terminator (which is part of the sentence).
        assert text[s.start : s.end].strip() == s.text


def test_split_sentences_handles_questions_and_exclamations() -> None:
    spans = split_sentences("Wait! Really? Yes.")
    assert [s.text for s in spans] == ["Wait!", "Really?", "Yes."]


def test_split_sentences_preserves_abbreviations() -> None:
    spans = split_sentences("Prof. Smith wrote it. Dr. Jones confirmed.")
    # The "Prof." should NOT end a sentence because it's in the abbrev list.
    assert len(spans) == 2
    assert spans[0].text.startswith("Prof.")
    assert spans[1].text.startswith("Dr.")


def test_split_sentences_empty_returns_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_split_sentences_single_sentence_without_terminator() -> None:
    # Tail-only content with no terminator is still returned.
    spans = split_sentences("no terminator here")
    assert len(spans) == 1
    assert spans[0].text == "no terminator here"


def test_sentence_containing_maps_offset_to_span() -> None:
    text = "First [1]. Second [2]. Third [3]."
    spans = split_sentences(text)
    # Marker [2] starts at position 18.
    pos = text.index("[2]")
    span = sentence_containing(spans, pos)
    assert span is not None
    assert span.text.startswith("Second")


def test_sentence_containing_returns_none_for_out_of_range() -> None:
    spans = split_sentences("Only one.")
    assert sentence_containing(spans, 1000) is None


# --- Existence check ---------------------------------------------------------


def test_check_existence_all_in_range() -> None:
    sources = [
        Source(metadata={"id": f"s{i}", "type": "book", "title": f"T{i}"}, content="c")
        for i in range(1, 4)
    ]
    citations = [Citation(span=(0, 3), source_id=1), Citation(span=(5, 8), source_id=3)]
    result = check_existence(citations, sources)
    assert result == ExistenceResult(all_exist=True, missing=())


def test_check_existence_flags_out_of_range() -> None:
    sources = [
        Source(metadata={"id": "s1", "type": "book", "title": "T"}, content="c"),
        Source(metadata={"id": "s2", "type": "book", "title": "T"}, content="c"),
    ]
    citations = [
        Citation(span=(0, 3), source_id=1),
        Citation(span=(5, 8), source_id=5),  # fabricated — only 2 sources
        Citation(span=(10, 13), source_id=7),  # also fabricated
    ]
    result = check_existence(citations, sources)
    assert not result.all_exist
    assert result.missing == (5, 7)


def test_check_existence_empty_citations_is_trivially_true() -> None:
    sources = [Source(metadata={"id": "s1", "type": "book", "title": "T"}, content="c")]
    assert check_existence([], sources) == ExistenceResult(all_exist=True, missing=())


# --- Mock NLI + Verifier orchestration --------------------------------------


@dataclass
class _ScriptedNLI(NLIModel):
    """NLI backend that returns canned scores without loading transformers.

    Scores are looked up by (premise_fragment, hypothesis_fragment) — if any
    of the mapping's premise keys appears in the pair's premise AND the
    hypothesis key appears in the hypothesis, use that score. Falls back to
    0.1 entailment (neutral/contradiction 0.45 each).
    """

    def __init__(  # type: ignore[no-untyped-def]
        self, rules: dict[tuple[str, str], float] | None = None
    ) -> None:
        # Skip NLIModel.__init__ (it would try to import torch).
        self.model_name = "scripted-mock"
        self.device = "cpu"
        self.batch_size = 8
        self.rules: dict[tuple[str, str], float] = rules or {}

    def entail_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:  # type: ignore[override]
        out: list[NLIResult] = []
        for premise, hypothesis in pairs:
            score = 0.1
            for (premise_key, hypothesis_key), mapped in self.rules.items():
                if premise_key in premise and hypothesis_key in hypothesis:
                    score = mapped
                    break
            other = (1 - score) / 2
            out.append(NLIResult(entailment=score, neutral=other, contradiction=other))
        return out


def _sources(contents: list[str]) -> list[Source]:
    return [
        Source(
            metadata={"id": f"s{i}", "type": "book", "title": f"T{i}"},
            content=c,
        )
        for i, c in enumerate(contents, start=1)
    ]


def test_verifier_everything_supported() -> None:
    sources = _sources(
        [
            "The Transformer architecture uses self-attention.",
            "BERT introduced masked language modeling.",
        ]
    )
    text = "Transformers use self-attention [1]. BERT uses masked language modeling [2]."
    citations = [
        Citation(span=text.index("[1]"), source_id=1)
        if False
        else Citation(span=(32, 35), source_id=1),
        Citation(span=(text.index("[2]"), text.index("[2]") + 3), source_id=2),
    ]
    nli = _ScriptedNLI(
        rules={
            ("self-attention", "self-attention"): 0.92,
            ("masked language modeling", "masked language"): 0.88,
        }
    )
    verifier = Verifier(threshold=0.5, nli=nli)
    report = verifier.verify(text=text, citations=citations, sources=sources)
    assert report.support_rate == 1.0
    assert all(cs.supported for cs in report.per_citation)
    assert report.uncited_but_entailed == []


def test_verifier_flags_unsupported_citations() -> None:
    sources = _sources(
        [
            "The paper discusses dogs.",
            "The paper discusses cats.",
        ]
    )
    text = "Transformers use self-attention [1]."
    citations = [Citation(span=(32, 35), source_id=1)]
    # No rules → default 0.1 entailment → below 0.5 threshold → unsupported.
    nli = _ScriptedNLI()
    verifier = Verifier(threshold=0.5, nli=nli)
    report = verifier.verify(
        text=text,
        citations=citations,
        sources=sources,
        run_coverage=False,
    )
    assert report.support_rate == 0.0
    assert report.per_citation[0].supported is False
    assert report.per_citation[0].entailment_score < 0.5


def test_verifier_flags_fabricated_citation_as_unsupported() -> None:
    sources = _sources(["real source content"])
    text = "Fabricated claim [7]."  # only 1 source in scope, [7] is fake
    citations = [Citation(span=(17, 20), source_id=7)]
    nli = _ScriptedNLI(rules={("real", "Fabricated"): 0.99})  # even if NLI lies
    verifier = Verifier(threshold=0.5, nli=nli)
    report = verifier.verify(
        text=text,
        citations=citations,
        sources=sources,
        run_coverage=False,
    )
    # Out-of-range source_id overrides entailment score → supported=False.
    assert report.per_citation[0].supported is False


def test_verifier_coverage_flags_uncited_but_entailed() -> None:
    sources = _sources(
        [
            "The Transformer architecture uses self-attention.",
            "BERT introduced masked language modeling.",
        ]
    )
    # First sentence cites nothing but should match source 1.
    text = "Transformers use self-attention heavily. But we cite nothing."
    citations: list[Citation] = []
    nli = _ScriptedNLI(
        rules={
            ("self-attention", "self-attention"): 0.92,
            ("masked language modeling", "self-attention"): 0.05,
        }
    )
    verifier = Verifier(threshold=0.5, nli=nli)
    report = verifier.verify(text=text, citations=citations, sources=sources)
    # The first sentence ("Transformers use self-attention heavily.") should
    # be flagged as uncited-but-entailed by source 1.
    assert len(report.uncited_but_entailed) >= 1
    flagged = report.uncited_but_entailed[0]
    assert flagged.candidate_source_id == 1
    assert flagged.entailment_score >= 0.5


def test_verifier_run_coverage_false_skips_coverage_pass() -> None:
    sources = _sources(["content mentioning self-attention"])
    text = "Uncited claim about self-attention."
    citations: list[Citation] = []
    nli = _ScriptedNLI(rules={("self-attention", "self-attention"): 0.9})
    verifier = Verifier(threshold=0.5, nli=nli)
    report = verifier.verify(
        text=text,
        citations=citations,
        sources=sources,
        run_coverage=False,
    )
    assert report.uncited_but_entailed == []


def test_verifier_empty_inputs_returns_vacuous_report() -> None:
    verifier = Verifier(threshold=0.5, nli=_ScriptedNLI())
    report = verifier.verify(text="", citations=[], sources=[])
    assert report.support_rate == 1.0
    assert report.per_citation == []
    assert report.uncited_but_entailed == []


# --- GenerationResult.verify() integration with mock -------------------------


def test_generation_result_verify_with_mock_nli() -> None:
    """`GenerationResult.verify()` wraps Verifier — same orchestration."""
    sources = _sources(["Premise content about transformers and self-attention."])
    text = "Self-attention is central [1]."
    result = GenerationResult(
        text=text,
        citations=[Citation(span=(26, 29), source_id=1)],
        sources=sources,
    )
    # Build a prefilled Verifier and pass via `nli=` — a user could do this
    # to avoid the heavy default load on every call.
    nli = _ScriptedNLI(rules={("self-attention", "Self-attention"): 0.95})
    report = result.verify(nli=nli, run_coverage=False)
    assert report.support_rate == 1.0
    assert report.per_citation[0].supported is True


# --- ADR-013: verify() uses cited_text as premise when populated ------------


class _CapturingNLI(_ScriptedNLI):
    """Captures every ``(premise, hypothesis)`` pair the verifier sends.

    Lets us assert that ``Citation.cited_text`` (when populated) is used
    as the NLI premise instead of the full source content — the sharper-
    signal upgrade enabled by ADR-013.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self, rules: dict[tuple[str, str], float] | None = None
    ) -> None:
        super().__init__(rules=rules)
        self.captured_premises: list[str] = []

    def entail_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:  # type: ignore[override]
        for premise, _hyp in pairs:
            self.captured_premises.append(premise)
        return super().entail_batch(pairs)


def test_verify_uses_cited_text_as_premise_when_populated() -> None:
    """When ``Citation.cited_text`` is set (Anthropic Citations API path),
    NLI scores against that span — not the whole source content."""
    sources = _sources(
        [
            "A very long source with lots of irrelevant material before "
            "the actual cited passage which appears way down here at the "
            "very end of the document body, surrounded by yet more noise."
        ]
    )
    text = "The model says something [1]."
    citations = [
        Citation(
            span=(25, 28),
            source_id=1,
            cited_text="the actual cited passage",
            source_span=(60, 84),
            document_title="Long Source",
        )
    ]
    nli = _CapturingNLI()
    verifier = Verifier(threshold=0.5, nli=nli)
    verifier.verify(text=text, citations=citations, sources=sources, run_coverage=False)
    assert nli.captured_premises == ["the actual cited passage"]


def test_verify_falls_back_to_source_content_when_cited_text_absent() -> None:
    """Backends without span attribution (everyone except Anthropic today)
    leave ``cited_text`` ``None`` — the verifier falls back to the full
    source content, preserving the v0.1 behaviour."""
    sources = _sources(["The full source content used as premise."])
    text = "Claim [1]."
    citations = [Citation(span=(6, 9), source_id=1)]  # no cited_text
    nli = _CapturingNLI()
    verifier = Verifier(threshold=0.5, nli=nli)
    verifier.verify(text=text, citations=citations, sources=sources, run_coverage=False)
    assert nli.captured_premises == ["The full source content used as premise."]


def test_verify_mixes_cited_text_and_full_source_per_citation() -> None:
    """In a single result, some citations carry ``cited_text`` (Anthropic)
    and some don't (other backends mixed in the same pipeline). Each
    citation should use the sharpest premise available to *it*."""
    sources = _sources(
        [
            "A very long source with the relevant snippet hidden in the middle.",
            "Second full source content.",
        ]
    )
    text = "First claim [1]. Second claim [2]."
    citations = [
        Citation(
            span=(13, 16),
            source_id=1,
            cited_text="the relevant snippet",
        ),
        Citation(span=(31, 34), source_id=2),  # no cited_text — full source
    ]
    nli = _CapturingNLI()
    verifier = Verifier(threshold=0.5, nli=nli)
    verifier.verify(text=text, citations=citations, sources=sources, run_coverage=False)
    assert nli.captured_premises == [
        "the relevant snippet",
        "Second full source content.",
    ]
