"""Per-citation NLI entailment check.

For every `Citation` in a `GenerationResult`, we look up the sentence that
*contains* the marker and check whether the cited source entails it. The
output is one `CitationSupport` record per citation carrying the NLI
entailment score and a ``supported`` boolean thresholded at the caller's
preferred value.

The premise is normally the **source content** (truncated on the NLI's
512-token budget); the hypothesis is the **sentence containing the
citation**. When the backend has populated :attr:`Citation.cited_text`
(Anthropic's Citations API does — see ADR-013), we use that span as the
premise instead — sharper signal than the whole document, especially on
long sources where the relevant assertion is buried past the 512-token
horizon.
"""

from __future__ import annotations

from citeformer.core import Citation, Source
from citeformer.verify.nli import NLIModel
from citeformer.verify.report import CitationSupport
from citeformer.verify.sentences import SentenceSpan, sentence_containing, strip_citation_markers


def score_entailment(
    citations: list[Citation],
    sentence_spans: list[SentenceSpan],
    sources: list[Source],
    *,
    nli: NLIModel,
    threshold: float = 0.5,
) -> list[CitationSupport]:
    """Score every citation's entailment against its cited source.

    Args:
        citations: The citations to score.
        sentence_spans: Sentence spans over `GenerationResult.text` (from
            `citeformer.verify.sentences.split_sentences`).
        sources: All sources in scope; used to look up content by source_id.
        nli: The NLI backend.
        threshold: Entailment probability above which `CitationSupport.
            supported` is True.

    Returns:
        One `CitationSupport` per citation, in the same order.
    """
    if not citations:
        return []

    pairs: list[tuple[str, str]] = []
    for citation in citations:
        src_idx = citation.source_id - 1
        if not (0 <= src_idx < len(sources)):
            # Out-of-range — shouldn't happen under Tier 1 grammar
            # enforcement. We pair against an empty premise so the NLI
            # model returns "neutral" and `supported` is False.
            pairs.append(("", _claim_for(citation, sentence_spans)))
            continue
        # Prefer the provider-supplied cited span (Anthropic's
        # Citations API populates ``cited_text``) — it's the exact
        # passage the model claimed to draw from, so it's a sharper
        # NLI premise than the whole document. Falls back to full
        # source content for backends that don't surface span-level
        # attribution (everyone except Anthropic today).
        premise = citation.cited_text or sources[src_idx].content or ""
        hypothesis = _claim_for(citation, sentence_spans)
        pairs.append((premise, hypothesis))

    scores = nli.entail_batch(pairs)
    out: list[CitationSupport] = []
    for i, score in enumerate(scores):
        out.append(
            CitationSupport(
                citation_index=i,
                entailment_score=score.entailment,
                supported=score.entailment >= threshold,
            )
        )
    return out


def _claim_for(citation: Citation, sentence_spans: list[SentenceSpan]) -> str:
    """Find the sentence text containing the citation's span, minus markers.

    ``[N]`` markers contain no semantic content. Leaving them in the NLI
    hypothesis drops entailment scores dramatically on some DeBERTa MNLI
    variants (we observed 0.996 → 0.011 on the cross-encoder/DeBERTa-v3-
    base checkpoint). We strip all inline markers before scoring.
    """
    start, _ = citation.span
    span = sentence_containing(sentence_spans, start)
    if span is None:
        return ""
    return strip_citation_markers(span.text)
