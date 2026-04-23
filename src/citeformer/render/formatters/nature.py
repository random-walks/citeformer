"""Nature citation style.

Numeric inline markers (``1`` — no brackets; Nature renders them as
superscripts in typeset output, but our plain-text form keeps a bare number
for simplicity and downstream flexibility).

Bibliography shape:

- Article: ``1. Author, A. & Author, B. Title. Journal Volume, Pages (Year).``
- Book: ``1. Author, A. Book Title (Publisher, Year).``
- Chapter: ``1. Author, A. Chapter Title. In Book Title (eds Editor, E.) Pages (Publisher, Year).``

Authors: ``Last, F. M.``, ampersand before the final. For >5 authors, Nature
abbreviates to ``Last, F. et al.`` after the first.
"""

from __future__ import annotations

from citeformer.render.formatters._base import (
    Author,
    CitationFormatter,
    CSLItem,
    ensure_period,
    format_page_range,
    get_str,
    get_title,
    parse_authors,
    parse_year,
)

_ET_AL_THRESHOLD = 5


def _format_author(a: Author) -> str:
    if a.is_literal:
        return a.literal
    initials = a.given_initials
    return f"{a.family}, {initials}".strip(", ") if initials else a.family


def _format_authors(authors: list[Author]) -> str:
    if not authors:
        return ""
    if len(authors) > _ET_AL_THRESHOLD:
        return f"{_format_author(authors[0])} et al."
    parts = [_format_author(a) for a in authors]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} & {parts[1]}"
    return f"{', '.join(parts[:-1])} & {parts[-1]}"


class NatureFormatter(CitationFormatter):
    """Nature — numeric, journal-style bibliography."""

    name = "nature"
    citation_format = "numeric"

    def inline(self, item: CSLItem, number: int) -> str:
        del item
        return str(number)

    def bibliography(self, item: CSLItem, number: int) -> str:
        item_type = str(item.get("type", "article-journal"))
        dispatch = {
            "article-journal": self._article,
            "book": self._book,
            "chapter": self._chapter,
            "paper-conference": self._article,  # Nature folds proceedings into journals
            "thesis": self._thesis,
            "webpage": self._webpage,
            "report": self._book,
        }
        formatter = dispatch.get(item_type, self._article)
        body = formatter(item)
        return f"{number}. {body}"

    def _article(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        journal = get_str(item, "container-title")
        volume = get_str(item, "volume")
        pages = format_page_range(get_str(item, "page"))
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            chunks.append(ensure_period(title))
        if journal:
            journal_chunk = journal
            if volume:
                journal_chunk += f" {volume}"
            if pages:
                journal_chunk += f", {pages}"
            if year:
                journal_chunk += f" ({year})"
            chunks.append(f"{journal_chunk}.")
        elif year:
            chunks.append(f"({year}).")
        return " ".join(chunks)

    def _book(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        publisher = get_str(item, "publisher")
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            if publisher and year:
                chunks.append(f"{title} ({publisher}, {year}).")
            elif publisher:
                chunks.append(f"{title} ({publisher}).")
            elif year:
                chunks.append(f"{title} ({year}).")
            else:
                chunks.append(ensure_period(title))
        return " ".join(chunks)

    def _chapter(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        book = get_str(item, "container-title")
        editors = _format_authors(parse_authors(item.get("editor")))
        publisher = get_str(item, "publisher")
        year = parse_year(item.get("issued"))
        pages = format_page_range(get_str(item, "page"))

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            chunks.append(ensure_period(title))
        tail_parts: list[str] = []
        if book:
            tail_parts.append(f"In {book}")
        if editors:
            tail_parts.append(f"(eds {editors})")
        if pages:
            tail_parts.append(pages)
        if publisher and year:
            tail_parts.append(f"({publisher}, {year})")
        elif publisher:
            tail_parts.append(f"({publisher})")
        elif year:
            tail_parts.append(f"({year})")
        if tail_parts:
            chunks.append(" ".join(tail_parts) + ".")
        return " ".join(chunks)

    def _thesis(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        publisher = get_str(item, "publisher")
        genre = get_str(item, "genre") or "PhD thesis"
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            chunks.append(ensure_period(title))
        detail: list[str] = [genre]
        if publisher:
            detail.append(publisher)
        if year:
            detail.append(str(year))
        chunks.append("(" + ", ".join(detail) + ").")
        return " ".join(chunks)

    def _webpage(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        site = get_str(item, "container-title")
        year = parse_year(item.get("issued"))
        url = get_str(item, "URL")

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            chunks.append(ensure_period(title))
        if site:
            if year:
                chunks.append(f"{site} ({year}).")
            else:
                chunks.append(f"{site}.")
        elif year:
            chunks.append(f"({year}).")
        if url:
            chunks.append(url)
        return " ".join(chunks)
