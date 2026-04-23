"""MLA 9 citation style.

Inline markers are ``(Smith 45)`` (author + page) when a locator is known,
``(Smith)`` otherwise. Since our pipeline doesn't propagate per-cite
locators through ``Citation`` yet, the inline marker uses ``(Smith)`` /
``(Smith and Jones)`` / ``(Smith et al.)`` consistently.

Bibliography shape:

- Article: ``Smith, Alice. "Title." Journal Name, vol. 12, no. 3, 2023, pp. 45–67.``
- Book: ``Smith, Alice. Book Title. Publisher, 2023.``
- Chapter: ``Smith, Alice. "Chapter Title." Book Title, edited by Editor, Publisher, 2023, pp. 45–67.``
- Thesis: ``Smith, Alice. Title. 2023. University, PhD dissertation.``
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
    if len(authors) >= _INLINE_ET_AL_THRESHOLD:
        return f"{_format_first_author(authors[0])}, et al."
    first = _format_first_author(authors[0])
    rest = [_format_subsequent_author(a) for a in authors[1:]]
    if not rest:
        return first
    return f"{first}, and {rest[0]}"


def _format_authors_inline(authors: list[Author]) -> str:
    if not authors:
        return ""
    if len(authors) >= _INLINE_ET_AL_THRESHOLD:
        a = authors[0]
        return (a.literal if a.is_literal else a.family) + " et al."
    names = [a.literal if a.is_literal else a.family for a in authors]
    return " and ".join(names)


class MLAFormatter(CitationFormatter):
    """MLA 9 — ``(Smith)`` inline; Works Cited bibliography per MLA 9 handbook."""

    name = "mla-9"
    citation_format = "author"

    def inline(self, item: CSLItem, number: int) -> str:
        del number
        authors = parse_authors(item.get("author"))
        author_part = _format_authors_inline(authors) or "Anon."
        return f"({author_part})"

    def bibliography(self, item: CSLItem, number: int) -> str:
        del number
        item_type = str(item.get("type", "article-journal"))
        dispatch = {
            "article-journal": self._article,
            "book": self._book,
            "chapter": self._chapter,
            "paper-conference": self._article,
            "thesis": self._thesis,
            "webpage": self._webpage,
            "report": self._book,
        }
        formatter = dispatch.get(item_type, self._article)
        return formatter(item)

    def _article(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        title = get_title(item)
        journal = get_str(item, "container-title")
        volume = get_str(item, "volume")
        issue = get_str(item, "issue")
        year = parse_year(item.get("issued"))
        pages = format_page_range(get_str(item, "page"))

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            chunks.append(f'"{title}."')
        parts: list[str] = []
        if journal:
            parts.append(journal)
        if volume:
            parts.append(f"vol. {volume}")
        if issue:
            parts.append(f"no. {issue}")
        if year:
            parts.append(str(year))
        if pages:
            parts.append(f"pp. {pages}")
        if parts:
            chunks.append(", ".join(parts) + ".")
        return " ".join(chunks)

    def _book(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        title = get_title(item)
        publisher = get_str(item, "publisher")
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            chunks.append(f"{title}.")
        tail: list[str] = []
        if publisher:
            tail.append(publisher)
        if year:
            tail.append(str(year))
        if tail:
            chunks.append(", ".join(tail) + ".")
        return " ".join(chunks)

    def _chapter(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        title = get_title(item)
        book = get_str(item, "container-title")
        editors = parse_authors(item.get("editor"))
        publisher = get_str(item, "publisher")
        year = parse_year(item.get("issued"))
        pages = format_page_range(get_str(item, "page"))

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            chunks.append(f'"{title}."')
        tail_parts: list[str] = []
        if book:
            tail_parts.append(book)
        if editors:
            editor_str = ", ".join(_format_subsequent_author(e) for e in editors)
            tail_parts.append(f"edited by {editor_str}")
        if publisher:
            tail_parts.append(publisher)
        if year:
            tail_parts.append(str(year))
        if pages:
            tail_parts.append(f"pp. {pages}")
        if tail_parts:
            chunks.append(", ".join(tail_parts) + ".")
        return " ".join(chunks)

    def _thesis(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        title = get_title(item)
        year = parse_year(item.get("issued"))
        publisher = get_str(item, "publisher")
        genre = get_str(item, "genre") or "PhD dissertation"

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            chunks.append(f"{title}.")
        if year:
            chunks.append(f"{year}.")
        final_parts: list[str] = []
        if publisher:
            final_parts.append(publisher)
        final_parts.append(genre)
        chunks.append(", ".join(final_parts) + ".")
        return " ".join(chunks)

    def _webpage(self, item: CSLItem) -> str:
        authors = _format_authors_bib(parse_authors(item.get("author")))
        title = get_title(item)
        site = get_str(item, "container-title")
        year = parse_year(item.get("issued"))
        url = get_str(item, "URL")

        chunks: list[str] = []
        if authors:
            chunks.append(ensure_period(authors))
        if title:
            chunks.append(f'"{title}."')
        tail: list[str] = []
        if site:
            tail.append(site)
        if year:
            tail.append(str(year))
        if url:
            tail.append(url)
        if tail:
            chunks.append(", ".join(tail) + ".")
        return " ".join(chunks)
