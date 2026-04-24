"""Sentence splitter for verification paths.

Verification needs to identify per-sentence char spans so that:

1. Each `Citation` can be associated with "the sentence containing this marker".
2. Uncited sentences can be scored against every source for coverage checks.

We avoid heavy NLP dependencies (nltk with punkt download, spacy) and emit
spans via a small regex-based splitter. This handles the common cases — ASCII
and Unicode terminators, multiple terminators (``!?``, ``!!``), abbreviations
common enough to skip (``Dr.``, ``et al.``, ``e.g.``, ``i.e.``). It will
mis-split on exotic cases (abbreviated initials in names, URLs with dots);
that's an accepted limitation for v0.1.

Trade-off discussion lives in the verification docs
(``docs/verification.md#limitations``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split points: period / exclamation / question followed by whitespace + an
# uppercase letter or quote (new sentence) OR end of string. The lookbehind
# filters the small set of abbreviations that are sentence-internal.
_COMMON_ABBREVS = {
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "cf.",
    "al.",  # for "et al."
    "Fig.",
    "fig.",
    "No.",
    "no.",
    "Vol.",
    "vol.",
    "pp.",
    "Ch.",
}

_TERMINATORS = ".!?"

# Pattern for an ASCII sentence terminator followed by whitespace and
# something that looks like a new sentence start (uppercase letter, open
# quote, or a digit for numbered lists).
_SPLIT_PATTERN = re.compile(rf"([{re.escape(_TERMINATORS)}])(\s+)(?=[\"'\(\[]?[A-Z0-9])")


@dataclass(frozen=True)
class SentenceSpan:
    """One sentence extracted from a text, carrying its char offsets.

    Attributes:
        index: 0-indexed position among the sentences in the source text.
        start: Inclusive char offset into the original text.
        end: Exclusive char offset into the original text.
        text: The sentence slice (stripped of leading/trailing whitespace).
    """

    index: int
    start: int
    end: int
    text: str


def split_sentences(text: str) -> list[SentenceSpan]:
    """Split ``text`` into sentence spans.

    Spans are returned in source order and cover the full text (modulo
    leading / trailing whitespace). Empty / whitespace-only inputs return
    an empty list.

    Args:
        text: The text to split.

    Returns:
        A list of `SentenceSpan` records.
    """
    if not text or not text.strip():
        return []

    # Walk the split candidates and build spans. We skip a split if the
    # left-of-terminator token is an abbreviation (``Dr.``, ``e.g.``).
    spans: list[SentenceSpan] = []
    segment_start = 0
    index = 0
    cursor = 0

    while cursor < len(text):
        match = _SPLIT_PATTERN.search(text, cursor)
        if match is None:
            break

        term_pos = match.start(1)  # position of ., !, or ?
        # Extract the word ending at term_pos + 1 (inclusive of the terminator).
        # This is the last whitespace-separated token. If it's a known
        # abbreviation, don't split here.
        token_start = term_pos
        while token_start > 0 and not text[token_start - 1].isspace():
            token_start -= 1
        token = text[token_start : term_pos + 1]
        if token in _COMMON_ABBREVS:
            cursor = match.end()
            continue

        segment_end = term_pos + 1  # include the terminator
        segment = text[segment_start:segment_end].strip()
        if segment:
            stripped_start = segment_start + (
                len(text[segment_start:segment_end]) - len(text[segment_start:segment_end].lstrip())
            )
            spans.append(
                SentenceSpan(
                    index=index,
                    start=stripped_start,
                    end=segment_end,
                    text=segment,
                )
            )
            index += 1
        segment_start = match.end()
        cursor = match.end()

    # Trailing segment (no terminator or unmatched terminator at end).
    tail = text[segment_start:].strip()
    if tail:
        raw_tail = text[segment_start:]
        stripped_start = segment_start + (len(raw_tail) - len(raw_tail.lstrip()))
        spans.append(
            SentenceSpan(
                index=index,
                start=stripped_start,
                end=segment_start + len(raw_tail.rstrip()),
                text=tail,
            )
        )

    return spans


def sentence_containing(spans: list[SentenceSpan], char_offset: int) -> SentenceSpan | None:
    """Return the `SentenceSpan` containing ``char_offset``, or None if not found.

    Handy for mapping a `Citation.span` to the sentence it belongs to.
    """
    for span in spans:
        if span.start <= char_offset < span.end:
            return span
    return None


# Matches ``[N]`` / ``[1,2]`` / ``[1-3]`` inline markers. Used to scrub
# citation markers out of text before handing to an NLI model — the
# markers contain no semantic content and can confuse the entailment
# scorer ("[1]" at the end of a sentence drops entailment dramatically on
# some DeBERTa variants).
_CITATION_MARKER_PATTERN = re.compile(r"\s?\[\d+(?:[,\-–]\s?\d+)*\]")


def strip_citation_markers(text: str) -> str:
    """Remove ``[N]`` style citation markers from ``text``.

    Leading spaces before the marker are consumed to avoid leaving
    double-spaces. Preserves trailing punctuation.
    """
    return _CITATION_MARKER_PATTERN.sub("", text).strip()
