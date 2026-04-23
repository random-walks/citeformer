"""Uncited-but-entailed coverage check.

Walks every sentence in the generated text. For each sentence that has *no*
citation, pairs it with every source and runs NLI. If any source's
entailment exceeds the threshold, we flag the sentence — under the
``AUTO`` policy this is how users catch claims that *should* have been
cited; under ``REQUIRED`` this should be empty (the grammar guarantees
every sentence has a cite).
"""

from __future__ import annotations

from citeformer.core import Citation, Source
from citeformer.verify.nli import NLIModel
from citeformer.verify.report import UncitedClaim
from citeformer.verify.sentences import SentenceSpan, strip_citation_markers


def find_uncited_but_entailed(
    citations: list[Citation],
    sentence_spans: list[SentenceSpan],
    sources: list[Source],
    *,
    nli: NLIModel,
    threshold: float = 0.5,
) -> list[UncitedClaim]:
    """Flag uncited sentences that any source would entail.

    Args:
        citations: Citations emitted in generation.
        sentence_spans: Sentence spans over the generated text.
        sources: All sources in scope.
        nli: The NLI backend.
        threshold: Entailment probability above which a sentence is flagged.

    Returns:
        One `UncitedClaim` per flagged sentence, with its best-matching
        source as `candidate_source_id`. Sentences that have at least one
        citation, or that no source entails, are omitted.
    """
    if not sentence_spans or not sources:
        return []

    # Identify sentences that already carry a citation. A citation "belongs
    # to" the sentence whose char span contains the marker's start offset.
    cited_sentence_indices: set[int] = set()
    for cit in citations:
        start, _ = cit.span
        for span in sentence_spans:
            if span.start <= start < span.end:
                cited_sentence_indices.add(span.index)
                break

    # Build (premise, hypothesis) pairs: for every uncited sentence, pair
    # it with every source. Track the (sentence_idx, source_id) for each
    # pair so we can decode scores back.
    pairs: list[tuple[str, str]] = []
    coords: list[tuple[int, int]] = []  # (sentence span index, source id)
    for span in sentence_spans:
        if span.index in cited_sentence_indices:
            continue
        scrubbed = strip_citation_markers(span.text)
        for i, source in enumerate(sources, start=1):
            pairs.append((source.content or "", scrubbed))
            coords.append((span.index, i))

    if not pairs:
        return []

    scores = nli.entail_batch(pairs)

    # For each sentence, pick the source with the highest entailment score.
    best_per_sentence: dict[int, tuple[int, float]] = {}  # idx → (source_id, score)
    for (sent_idx, source_id), score in zip(coords, scores, strict=True):
        best = best_per_sentence.get(sent_idx)
        if best is None or score.entailment > best[1]:
            best_per_sentence[sent_idx] = (source_id, score.entailment)

    # Build `UncitedClaim` for sentences whose best-matching source crosses
    # the threshold.
    span_by_index = {span.index: span for span in sentence_spans}
    out: list[UncitedClaim] = []
    for sent_idx, (source_id, entail_score) in best_per_sentence.items():
        if entail_score < threshold:
            continue
        span = span_by_index[sent_idx]
        out.append(
            UncitedClaim(
                span=(span.start, span.end),
                candidate_source_id=source_id,
                entailment_score=entail_score,
            )
        )
    # Stable order: by sentence start offset.
    out.sort(key=lambda uc: uc.span[0])
    return out
