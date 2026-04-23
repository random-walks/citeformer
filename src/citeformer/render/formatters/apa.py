"""APA 7 citation style.

Author-date inline markers (``(Smith, 2023)`` with single author; ``(Smith &
Jones, 2023)`` with two; ``(Smith et al., 2023)`` with three+).

Bibliography shape:

- Article: ``Smith, A. B., & Jones, C. D. (2023). Title. Journal Name, 12(3), 45–67. https://doi.org/…``
- Book: ``Smith, A. B. (2023). Book title. Publisher.``
- Chapter: ``Smith, A. B. (2023). Chapter title. In E. Editor (Ed.), Book title (pp. 45–67). Publisher.``
- Thesis: ``Smith, A. B. (2023). Title [Doctoral dissertation, University]. Repository.``

Authors: ``Last, F. M.``, ampersand before the final. For 21+ authors, APA
lists the first 19, then an ellipsis, then the final author. For our v0.1
scope we clamp at 20 (threshold gated for future polish).
"""

from __future__ import annotations

from citeformer.render.formatters._base import (
    Author,
    CitationFormatter,
    CSLItem,
    format_doi,
    format_page_range,
    get_str,
    get_title,
    parse_authors,
    parse_year,
)

_INLINE_ET_AL_THRESHOLD = 3  # three or more authors → "et al." inline
_BIBLIOGRAPHY_ELLIPSIS_THRESHOLD = 20


def _format_author_bib(a: Author) -> str:
    if a.is_literal:
        return a.literal
    initials = a.given_initials
    return f"{a.family}, {initials}".strip(", ") if initials else a.family


def _format_authors_bib(authors: list[Author]) -> str:
    """Bibliography author list — 'Last, F. M., Last, F. M., & Last, F. M.'"""
    if not authors:
        return ""
    if len(authors) > _BIBLIOGRAPHY_ELLIPSIS_THRESHOLD:
        first_19 = [_format_author_bib(a) for a in authors[:19]]
        last = _format_author_bib(authors[-1])
        return ", ".join(first_19) + f", … {last}"
    parts = [_format_author_bib(a) for a in authors]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}, & {parts[1]}"
    return f"{', '.join(parts[:-1])}, & {parts[-1]}"


def _format_author_inline(a: Author) -> str:
    if a.is_literal:
        return a.literal
    return a.family


def _format_authors_inline(authors: list[Author]) -> str:
    """Inline author list — 'Smith', 'Smith & Jones', 'Smith et al.' for 3+."""
    if not authors:
        return ""
    if len(authors) >= _INLINE_ET_AL_THRESHOLD:
        return f"{_format_author_inline(authors[0])} et al."
    parts = [_format_author_inline(a) for a in authors]
    return " & ".join(parts)


class APAFormatter(CitationFormatter):
    """APA 7 — author-date, Publication Manual bibliography format."""

    name = "apa-7"
    citation_format = "author-date"

    def inline(self, item: CSLItem, number: int) -> str:
        del number
        authors = parse_authors(item.get("author"))
        year = parse_year(item.get("issued"))
        author_part = _format_authors_inline(authors) or "Anon"
        year_part = str(year) if year is not None else "n.d."
        return f"({author_part}, {year_part})"

    def bibliography(self, item: CSLItem, number: int) -> str:
        del number
        item_type = str(item.get("type", "article-journal"))
        dispatch = {
            "article-journal": self._article,
            "book": self._book,
            "chapter": self._chapter,
            "paper-conference": self._paper_conference,
            "thesis": self._thesis,
            "webpage": self._webpage,
            "report": self._report,
        }
        formatter = dispatch.get(item_type, self._article)
        return formatter(item)

    # --- Per-type bodies ------------------------------------------------------

    def _article(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        journal = get_str(item, "container-title")
        volume = get_str(item, "volume")
        issue = get_str(item, "issue")
        pages = format_page_range(get_str(item, "page"))
        doi = format_doi(get_str(item, "DOI"))

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors}")
        chunks.append(f"({year if year else 'n.d.'}).")
        if title:
            chunks.append(f"{title}.")
        if journal:
            journal_chunk = journal
            if volume:
                journal_chunk += f", {volume}"
                if issue:
                    journal_chunk += f"({issue})"
            if pages:
                journal_chunk += f", {pages}"
            chunks.append(f"{journal_chunk}.")
        if doi:
            chunks.append(doi)
        return self._join(chunks)

    def _book(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        publisher = get_str(item, "publisher")

        chunks: list[str] = []
        if authors:
            chunks.append(authors)
        chunks.append(f"({year if year else 'n.d.'}).")
        if title:
            chunks.append(f"{title}.")
        if publisher:
            chunks.append(f"{publisher}.")
        return self._join(chunks)

    def _chapter(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        book = get_str(item, "container-title")
        editors = parse_authors(item.get("editor"))
        pages = format_page_range(get_str(item, "page"))
        publisher = get_str(item, "publisher")

        chunks: list[str] = []
        if authors:
            chunks.append(authors)
        chunks.append(f"({year if year else 'n.d.'}).")
        if title:
            chunks.append(f"{title}.")
        in_parts: list[str] = ["In"]
        if editors:
            # APA chapter editor form: "In E. Editor (Ed.),"
            editor_str = ", ".join(
                f"{e.given_initials} {e.family}".strip() if not e.is_literal else e.literal
                for e in editors
            )
            ed_label = "Eds." if len(editors) > 1 else "Ed."
            in_parts.append(f"{editor_str} ({ed_label}),")
        if book:
            tail = book
            if pages:
                tail += f" (pp. {pages})"
            in_parts.append(f"{tail}.")
        if in_parts != ["In"]:
            chunks.append(" ".join(in_parts))
        if publisher:
            chunks.append(f"{publisher}.")
        return self._join(chunks)

    def _paper_conference(self, item: CSLItem) -> str:
        # APA conference paper looks much like an article but with "Paper
        # presented at …" / container-title as the conference name.
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        conf = get_str(item, "container-title")
        place = get_str(item, "publisher-place")

        chunks: list[str] = []
        if authors:
            chunks.append(authors)
        chunks.append(f"({year if year else 'n.d.'}).")
        if title:
            chunks.append(f"{title}.")
        tail_parts: list[str] = []
        if conf:
            tail_parts.append(conf)
        if place:
            tail_parts.append(place)
        if tail_parts:
            chunks.append(", ".join(tail_parts) + ".")
        return self._join(chunks)

    def _thesis(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        genre = get_str(item, "genre") or "Doctoral dissertation"
        publisher = get_str(item, "publisher")

        chunks: list[str] = []
        if authors:
            chunks.append(authors)
        chunks.append(f"({year if year else 'n.d.'}).")
        if title:
            bracket_tail = genre
            if publisher:
                bracket_tail = f"{genre}, {publisher}"
            chunks.append(f"{title} [{bracket_tail}].")
        elif publisher:
            chunks.append(f"{publisher}.")
        return self._join(chunks)

    def _webpage(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        site = get_str(item, "container-title")
        url = get_str(item, "URL")

        chunks: list[str] = []
        if authors:
            chunks.append(authors)
        chunks.append(f"({year if year else 'n.d.'}).")
        if title:
            chunks.append(f"{title}.")
        if site:
            chunks.append(f"{site}.")
        if url:
            chunks.append(url)
        return self._join(chunks)

    def _report(self, item: CSLItem) -> str:
        # APA report: like a book but with a genre / number in brackets.
        return self._book(item)

    @staticmethod
    def _join(chunks: list[str]) -> str:
        """Join with spaces and clean up a trailing standalone period."""
        return " ".join(c.strip() for c in chunks if c)
