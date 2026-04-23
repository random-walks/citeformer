"""Chicago author-date citation style.

Inline markers are parenthetical ``(Smith 2023)`` — no comma between author
and year, unlike APA. For 3+ authors we use ``(Smith et al. 2023)``.

Bibliography shape:

- Article: ``Smith, Alice. 2023. "Title." Journal Name 12 (3): 45–67.``
- Book: ``Smith, Alice. 2023. Book Title. City: Publisher.``
- Chapter: ``Smith, Alice. 2023. "Chapter Title." In Book Title, edited by Editor, 45–67. City: Publisher.``
- Thesis: ``Smith, Alice. 2023. "Title." PhD diss., University.``

Authors: first author ``Last, First``, subsequent authors ``First Last``.
For 4+ authors in the bibliography Chicago lists all (up to 10); we cap
at 10 and use ``et al.`` beyond.
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

_INLINE_ET_AL_THRESHOLD = 3
_BIBLIOGRAPHY_ET_AL_THRESHOLD = 10


def _format_first_author(a: Author) -> str:
    if a.is_literal:
        return a.literal
    return f"{a.family}, {a.given}".strip(", ") if a.given else a.family


def _format_subsequent_author(a: Author) -> str:
    if a.is_literal:
        return a.literal
    return f"{a.given} {a.family}".strip() if a.given else a.family


def _format_authors_bib(authors: list[Author]) -> str:
    if not authors:
        return ""
    if len(authors) > _BIBLIOGRAPHY_ET_AL_THRESHOLD:
        return f"{_format_first_author(authors[0])}, et al."
    first = _format_first_author(authors[0])
    rest = [_format_subsequent_author(a) for a in authors[1:]]
    if not rest:
        return first
    if len(rest) == 1:
        return f"{first}, and {rest[0]}"
    return f"{first}, {', '.join(rest[:-1])}, and {rest[-1]}"


def _format_authors_inline(authors: list[Author]) -> str:
    if not authors:
        return ""
    if len(authors) >= _INLINE_ET_AL_THRESHOLD:
        name = authors[0]
        return (name.literal if name.is_literal else name.family) + " et al."
    names = [a.literal if a.is_literal else a.family for a in authors]
    return " and ".join(names)


class ChicagoAuthorDateFormatter(CitationFormatter):
    """Chicago author-date — ``(Smith 2023)`` inline; bibliography per CMOS 17."""

    name = "chicago-author-date"
    citation_format = "author-date"

    def inline(self, item: CSLItem, number: int) -> str:
        del number
        authors = parse_authors(item.get("author"))
        year = parse_year(item.get("issued"))
        author_part = _format_authors_inline(authors) or "Anon."
        year_part = str(year) if year is not None else "n.d."
        return f"({author_part} {year_part})"

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
            "report": self._book,
        }
        formatter = dispatch.get(item_type, self._article)
        return formatter(item)

    def _article(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        journal = get_str(item, "container-title")
        volume = get_str(item, "volume")
        issue = get_str(item, "issue")
        pages = format_page_range(get_str(item, "page"))

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        chunks.append(f"{year if year is not None else 'n.d.'}.")
        if title:
            chunks.append(f'"{title}."')
        if journal:
            j = journal
            if volume:
                j += f" {volume}"
                if issue:
                    j += f" ({issue})"
            if pages:
                j += f": {pages}"
            chunks.append(j + ".")
        return " ".join(chunks)

    def _book(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        place = get_str(item, "publisher-place")
        publisher = get_str(item, "publisher")

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        chunks.append(f"{year if year is not None else 'n.d.'}.")
        if title:
            chunks.append(f"{title}.")
        parts = [p for p in (place, publisher) if p]
        if parts:
            chunks.append(": ".join(parts) + ".")
        return " ".join(chunks)

    def _chapter(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        book = get_str(item, "container-title")
        editors = parse_authors(item.get("editor"))
        pages = format_page_range(get_str(item, "page"))
        place = get_str(item, "publisher-place")
        publisher = get_str(item, "publisher")

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        chunks.append(f"{year if year is not None else 'n.d.'}.")
        if title:
            chunks.append(f'"{title}."')
        in_parts: list[str] = ["In"]
        if book:
            in_parts.append(f"{book},")
        if editors:
            editor_str = ", ".join(_format_subsequent_author(e) for e in editors)
            in_parts.append(f"edited by {editor_str},")
        if pages:
            in_parts.append(f"{pages}.")
        if in_parts != ["In"]:
            # Ensure the In-clause ends with a period if pages didn't already.
            joined = " ".join(in_parts).rstrip(",")
            if not joined.endswith("."):
                joined += "."
            chunks.append(joined)
        loc = [p for p in (place, publisher) if p]
        if loc:
            chunks.append(": ".join(loc) + ".")
        return " ".join(chunks)

    def _paper_conference(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        conf = get_str(item, "container-title")
        place = get_str(item, "publisher-place")

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        chunks.append(f"{year if year is not None else 'n.d.'}.")
        if title:
            chunks.append(f'"{title}."')
        if conf:
            tail = f"Paper presented at {conf}"
            if place:
                tail += f", {place}"
            chunks.append(tail + ".")
        return " ".join(chunks)

    def _thesis(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        genre = get_str(item, "genre") or "PhD diss."
        publisher = get_str(item, "publisher")

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        chunks.append(f"{year if year is not None else 'n.d.'}.")
        if title:
            chunks.append(f'"{title}."')
        tail_parts: list[str] = [genre]
        if publisher:
            tail_parts.append(publisher)
        chunks.append(", ".join(tail_parts) + ".")
        return " ".join(chunks)

    def _webpage(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        year = parse_year(item.get("issued"))
        title = get_title(item)
        site = get_str(item, "container-title")
        url = get_str(item, "URL")

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        chunks.append(f"{year if year is not None else 'n.d.'}.")
        if title:
            chunks.append(f'"{title}."')
        if site:
            chunks.append(f"{site}.")
        if url:
            chunks.append(url + ".")
        return " ".join(chunks)
