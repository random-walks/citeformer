"""CitationFormatter ABC + shared CSL-JSON helpers.

Each concrete formatter (APA, MLA, Chicago, IEEE, Nature, Vancouver) subclasses
`CitationFormatter` and implements `inline()` + `bibliography()`. Shared
plumbing — author parsing, year extraction, page-range dashes, DOI/URL
rendering — lives here so styles stay focused on their specific formatting
quirks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

# CSL-JSON item shape. We treat it as a free-form dict throughout but accessors
# narrow it to the fields we actually use.
CSLItem = dict[str, Any]

# CSL citation-format categories recognised by the spec (+ "unknown" fallback).
CitationFormatKind = Literal["author-date", "author", "numeric", "note", "label"]


@dataclass(frozen=True)
class Author:
    """Normalized CSL-JSON author record.

    A CSL author is one of:
    - ``{"family": "...", "given": "..."}`` — personal name with family+given parts.
    - ``{"literal": "..."}`` — institutional / single-token / non-decomposable.

    We keep both forms in one dataclass so downstream formatters can pick the
    field they need without branching on dict keys.

    Attributes:
        family: Family / last name (empty for literal-only authors).
        given: Given / first name(s) (may be empty).
        literal: Non-decomposable name form (empty for personal names).
    """

    family: str = ""
    given: str = ""
    literal: str = ""

    @property
    def is_literal(self) -> bool:
        return bool(self.literal)

    @property
    def given_initials(self) -> str:
        """Return the given name(s) as initials: ``"Edgar Allan"`` → ``"E. A."``.

        Handles hyphenated names (``"Jean-Paul"`` → ``"J.-P."``) and
        existing dotted initials (``"E. A."`` → ``"E. A."``).
        """
        if not self.given:
            return ""
        parts: list[str] = []
        for token in self.given.replace(".", " ").split():
            # Hyphenated given names become dotted-initial pairs.
            if "-" in token:
                sub_initials = ".-".join(p[0] for p in token.split("-") if p)
                parts.append(sub_initials + ".")
            else:
                parts.append(token[0] + ".")
        return " ".join(parts)


def parse_authors(raw: list[dict[str, Any]] | None) -> list[Author]:
    """Convert CSL-JSON author dicts to `Author` records.

    Defensive: entries missing both ``family`` and ``literal`` are skipped.
    ``given`` alone (no family) is promoted to ``literal``.
    """
    if not raw:
        return []
    out: list[Author] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        literal = str(entry.get("literal", "") or "").strip()
        family = str(entry.get("family", "") or "").strip()
        given = str(entry.get("given", "") or "").strip()
        if literal:
            out.append(Author(literal=literal))
        elif family:
            out.append(Author(family=family, given=given))
        elif given:
            # Only a given name — treat as literal to preserve something.
            out.append(Author(literal=given))
    return out


def parse_year(issued: dict[str, Any] | None) -> int | None:
    """Extract a 4-digit year from a CSL ``issued`` field.

    Accepts ``{"date-parts": [[year, ...]]}`` (the canonical CSL form).
    """
    if not issued:
        return None
    date_parts = issued.get("date-parts")
    if not date_parts or not isinstance(date_parts, list) or not date_parts[0]:
        return None
    try:
        return int(date_parts[0][0])
    except (ValueError, TypeError, IndexError):
        return None


def get_str(item: CSLItem, key: str) -> str | None:
    """Return a stripped non-empty string value for ``key``, or None."""
    value = item.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def get_title(item: CSLItem) -> str:
    """Return the item's title, falling back to an empty string.

    CSL ``title`` can be either a string or (rarely) a list; we coerce.
    """
    raw = item.get("title", "")
    if isinstance(raw, list):
        return " ".join(str(t) for t in raw if t).strip()
    return str(raw).strip()


def format_page_range(pages: str | None, *, dash: str = "–") -> str | None:
    """Normalise a CSL ``page`` value, using the requested dash for ranges.

    ``"45-67"`` → ``"45–67"`` (en-dash by default; some styles want en-dash,
    some hyphen, some a special range indicator).
    """
    if not pages:
        return None
    s = str(pages).strip()
    if not s:
        return None
    # Replace plain hyphens between digits with the requested dash.
    parts = s.replace("—", "-").replace("–", "-").split("-")
    cleaned = [p.strip() for p in parts if p.strip()]
    if len(cleaned) == 1:
        return cleaned[0]
    return dash.join(cleaned)


def ensure_period(text: str) -> str:
    """Return ``text`` guaranteed to end in a period (no double-period).

    Used to terminate author-list chunks where a trailing initial may
    already end in ``.``. ``"Smith, E. A."`` stays as-is; ``"Smith"`` →
    ``"Smith."``.
    """
    if not text:
        return text
    return text if text.endswith((".", "!", "?")) else text + "."


def format_doi(doi: str | None) -> str | None:
    """Render a DOI as a full ``https://doi.org/…`` URL."""
    if not doi:
        return None
    d = str(doi).strip()
    if not d:
        return None
    if d.lower().startswith(("http://", "https://")):
        return d
    if d.lower().startswith("doi:"):
        d = d[4:]
    return f"https://doi.org/{d}"


class CitationFormatter(ABC):
    """Abstract base class for home-grown CSL style formatters.

    Subclasses must declare ``name`` + ``citation_format`` class variables and
    implement ``inline()`` + ``bibliography()``. Formatters are stateless by
    contract; one instance can render many items concurrently (though we
    typically instantiate per ``Citeformer`` call for clarity).

    Attributes:
        name: Canonical style identifier (e.g. ``"apa-7"``).
        citation_format: CSL citation-format classification. Drives the
            `Reference.inline_marker` user sees and informs downstream
            tooling about what kind of marker to expect.
    """

    name: str
    citation_format: CitationFormatKind

    @abstractmethod
    def inline(self, item: CSLItem, number: int) -> str:
        """Render the inline citation marker for a cited item.

        Args:
            item: The CSL-JSON item being cited.
            number: 1-indexed position in the bibliography.

        Returns:
            The style's native inline form — e.g. ``"(Smith, 2023)"`` for
            author-date styles, ``"[1]"`` for numeric styles.
        """

    @abstractmethod
    def bibliography(self, item: CSLItem, number: int) -> str:
        """Render the full bibliography entry for an item.

        Args:
            item: The CSL-JSON item being cited.
            number: 1-indexed position in the bibliography (used by numeric
                styles to render the leading ``"[1]"`` / ``"1."``).

        Returns:
            The full bibliography entry as a single plain-text string.
        """
