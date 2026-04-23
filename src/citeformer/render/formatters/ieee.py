"""IEEE citation style.

Numeric inline markers (``[1]``). Bibliography entries per IEEE's Author's
Kit conventions:

- Article: ``[1] A. Author, "Title," Journal Name, vol. X, no. Y, pp. Z, Month Year.``
- Book: ``[1] A. Author, Title. City: Publisher, Year.``
- Chapter: ``[1] A. Author, "Chapter Title," in Book Title, E. Editor, Ed. City: Publisher, Year, pp. X–Y.``
- Conference paper: ``[1] A. Author, "Title," in Proc. Conference, Year, pp. X–Y.``
- Thesis: ``[1] A. Author, "Title," Ph.D. dissertation, Dept., Univ., City, Year.``
- Webpage: ``[1] A. Author, "Title," Site, Year. [Online]. Available: URL``

Authors: ``F. M. Last``. For 3+ authors, IEEE allows ``et al.`` after the
first; we use it after six for consistency with common practice.
"""

from __future__ import annotations

from citeformer.render.formatters._base import (
    Author,
    CitationFormatter,
    CSLItem,
    format_page_range,
    get_str,
    get_title,
    parse_authors,
    parse_year,
)

_ET_AL_THRESHOLD = 6  # after this many, use "et al." after the first author


def _format_author(a: Author) -> str:
    if a.is_literal:
        return a.literal
    initials = a.given_initials
    return f"{initials} {a.family}".strip() if initials else a.family


def _format_authors(authors: list[Author]) -> str:
    if not authors:
        return ""
    if len(authors) > _ET_AL_THRESHOLD:
        return f"{_format_author(authors[0])} et al."
    parts = [_format_author(a) for a in authors]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


class IEEEFormatter(CitationFormatter):
    """IEEE — numeric, Author's-Kit bibliography format."""

    name = "ieee"
    citation_format = "numeric"

    def inline(self, item: CSLItem, number: int) -> str:
        del item
        return f"[{number}]"

    def bibliography(self, item: CSLItem, number: int) -> str:
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
        body = formatter(item)
        return f"[{number}] {body}"

    # -- Per-type formatters ---------------------------------------------------

    def _article(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        journal = get_str(item, "container-title")
        volume = get_str(item, "volume")
        issue = get_str(item, "issue")
        pages = format_page_range(get_str(item, "page"))
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(authors)
        if title:
            chunks.append(f'"{title},"')
        if journal:
            chunks.append(f"{journal},")
        if volume:
            chunks.append(f"vol. {volume},")
        if issue:
            chunks.append(f"no. {issue},")
        if pages:
            chunks.append(f"pp. {pages},")
        if year:
            chunks.append(f"{year}.")
        return " ".join(chunks).rstrip(",").rstrip() + (
            "" if chunks and chunks[-1].endswith(".") else "."
        )

    def _book(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        place = get_str(item, "publisher-place")
        publisher = get_str(item, "publisher")
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors},")
        if title:
            chunks.append(f"{title}.")
        location_parts: list[str] = []
        if place:
            location_parts.append(place)
        if publisher:
            location_parts.append(publisher)
        if location_parts:
            loc = ": ".join(location_parts)
            chunks.append(f"{loc},")
        if year:
            chunks.append(f"{year}.")
        return " ".join(chunks)

    def _chapter(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        book = get_str(item, "container-title")
        editors = _format_authors(parse_authors(item.get("editor")))
        place = get_str(item, "publisher-place")
        publisher = get_str(item, "publisher")
        year = parse_year(item.get("issued"))
        pages = format_page_range(get_str(item, "page"))

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors},")
        if title:
            chunks.append(f'"{title},"')
        if book:
            chunks.append(f"in {book},")
        if editors:
            chunks.append(f"{editors}, Ed.")
        location_parts = [p for p in (place, publisher) if p]
        if location_parts:
            chunks.append(": ".join(location_parts) + ",")
        if year:
            comma = "," if pages else "."
            chunks.append(f"{year}{comma}")
        if pages:
            chunks.append(f"pp. {pages}.")
        return " ".join(chunks)

    def _paper_conference(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        conf = get_str(item, "container-title")
        year = parse_year(item.get("issued"))
        pages = format_page_range(get_str(item, "page"))

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors},")
        if title:
            chunks.append(f'"{title},"')
        if conf:
            chunks.append(f"in Proc. {conf},")
        if year:
            comma = "," if pages else "."
            chunks.append(f"{year}{comma}")
        if pages:
            chunks.append(f"pp. {pages}.")
        return " ".join(chunks)

    def _thesis(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        publisher = get_str(item, "publisher")  # institution
        genre = get_str(item, "genre") or "Ph.D. dissertation"
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors},")
        if title:
            chunks.append(f'"{title},"')
        chunks.append(f"{genre},")
        if publisher:
            chunks.append(f"{publisher},")
        if year:
            chunks.append(f"{year}.")
        else:
            chunks[-1] = chunks[-1].rstrip(",") + "."
        return " ".join(chunks)

    def _webpage(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        site = get_str(item, "container-title")
        year = parse_year(item.get("issued"))
        url = get_str(item, "URL")

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors},")
        if title:
            chunks.append(f'"{title},"')
        if site:
            chunks.append(f"{site},")
        if year:
            chunks.append(f"{year}.")
        chunks.append("[Online].")
        if url:
            chunks.append(f"Available: {url}")
        return " ".join(chunks)

    def _report(self, item: CSLItem) -> str:
        # Treat reports as a book-like form.
        return self._book(item)
