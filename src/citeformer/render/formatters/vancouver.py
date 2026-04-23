"""Vancouver citation style (ICMJE / NLM recommended format).

Numeric inline markers, rendered as ``(1)`` in-text in some variants and
bare ``1`` in others — we emit ``[1]`` for consistency with our grammar
contract (§10.1: ``cite-id ::= "[" digits "]"``). Downstream users who need
parenthesised inline markers post-process via ``Reference.inline_marker``.

Bibliography shape:

- Article: ``1. Author AA, Author BB. Title. Journal. Year;Volume(Issue):Pages.``
- Book: ``1. Author AA. Title. City: Publisher; Year.``
- Chapter: ``1. Author AA. Chapter Title. In: Editor EE, ed. Book Title. City: Publisher; Year. p. X–Y.``

Authors: ``Last FF`` (no punctuation between family and initials). After
six authors, Vancouver abbreviates with ``et al.``.
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

_ET_AL_THRESHOLD = 6


def _format_author(a: Author) -> str:
    if a.is_literal:
        return a.literal
    # Vancouver: family + undotted initials, e.g. "Smith AB"
    initials = a.given_initials.replace(".", "").replace(" ", "").replace("-", "")
    return f"{a.family} {initials}".strip() if initials else a.family


def _format_authors(authors: list[Author]) -> str:
    if not authors:
        return ""
    if len(authors) > _ET_AL_THRESHOLD:
        return f"{_format_author(authors[0])} et al."
    parts = [_format_author(a) for a in authors]
    return ", ".join(parts)


class VancouverFormatter(CitationFormatter):
    """Vancouver — numeric, ICMJE biomedical bibliography format."""

    name = "vancouver"
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
        issue = get_str(item, "issue")
        pages = format_page_range(get_str(item, "page"))
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors}.")
        if title:
            chunks.append(f"{title}.")
        if journal:
            chunks.append(f"{journal}.")
        detail_parts: list[str] = []
        if year:
            detail_parts.append(str(year))
        vol_issue = volume or ""
        if volume and issue:
            vol_issue = f"{volume}({issue})"
        if vol_issue:
            if detail_parts:
                detail_parts[-1] += f";{vol_issue}"
            else:
                detail_parts.append(vol_issue)
        if pages:
            if detail_parts:
                detail_parts[-1] += f":{pages}"
            else:
                detail_parts.append(pages)
        if detail_parts:
            chunks.append("".join(detail_parts) + ".")
        return " ".join(chunks)

    def _book(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        place = get_str(item, "publisher-place")
        publisher = get_str(item, "publisher")
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors}.")
        if title:
            chunks.append(f"{title}.")
        location_parts: list[str] = [p for p in (place, publisher) if p]
        if location_parts and year:
            chunks.append(": ".join(location_parts) + f"; {year}.")
        elif location_parts:
            chunks.append(": ".join(location_parts) + ".")
        elif year:
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
            chunks.append(f"{authors}.")
        if title:
            chunks.append(f"{title}.")
        in_parts: list[str] = ["In:"]
        if editors:
            in_parts.append(f"{editors}, ed.")
        if book:
            in_parts.append(book + ".")
        chunks.append(" ".join(in_parts))
        location_parts: list[str] = [p for p in (place, publisher) if p]
        if location_parts and year:
            chunks.append(": ".join(location_parts) + f"; {year}.")
        elif location_parts:
            chunks.append(": ".join(location_parts) + ".")
        elif year:
            chunks.append(f"{year}.")
        if pages:
            chunks.append(f"p. {pages}.")
        return " ".join(chunks)

    def _paper_conference(self, item: CSLItem) -> str:
        # Vancouver treats conference papers similarly to book chapters; the
        # "container-title" is the proceedings name.
        return self._chapter(item)

    def _thesis(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        publisher = get_str(item, "publisher")
        genre = get_str(item, "genre") or "dissertation"
        year = parse_year(item.get("issued"))

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors}.")
        if title:
            chunks.append(f"{title} [{genre}].")
        if publisher and year:
            chunks.append(f"{publisher}; {year}.")
        elif publisher:
            chunks.append(f"{publisher}.")
        elif year:
            chunks.append(f"{year}.")
        return " ".join(chunks)

    def _webpage(self, item: CSLItem) -> str:
        authors = _format_authors(parse_authors(item.get("author")))
        title = get_title(item)
        site = get_str(item, "container-title")
        year = parse_year(item.get("issued"))
        url = get_str(item, "URL")

        chunks: list[str] = []
        if authors:
            chunks.append(f"{authors}.")
        if title:
            chunks.append(f"{title} [Internet].")
        if site:
            if year:
                chunks.append(f"{site}; {year}.")
            else:
                chunks.append(f"{site}.")
        elif year:
            chunks.append(f"{year}.")
        if url:
            chunks.append(f"Available from: {url}")
        return " ".join(chunks)
