"""Minimal BibTeX → CSL-JSON parser.

Small, dependency-free BibTeX adapter for the common case: ``@article{...}``,
``@book{...}``, ``@inproceedings{...}``, etc., with a handful of familiar
fields (``author``, ``title``, ``year``, ``journal``, ``pages``, ``doi`` …).
Complex BibTeX corners — ``@string`` macro substitution, ``@preamble`` blocks,
LaTeX accent escapes, crossref-inheritance — are **out of scope**. Users with
a corner-heavy library should run the file through
`bibtexparser <https://pypi.org/project/bibtexparser/>`_ or convert via Zotero
and load via :func:`~citeformer.metadata.zotero.load_zotero_csl` instead.

What we do support:

- ``@type{key,`` followed by ``field = {value},`` or ``field = "value",``.
- Balanced braces inside values (``title = {The {B}ook}``).
- Author / editor splitting on ``" and "`` (case-insensitive) with
  ``Family, Given`` and ``Given Family`` conventions.
- A common-field and entry-type map to CSL 1.0.

Unknown fields are preserved under ``custom`` to avoid silent data loss.

Public API:

- :func:`parse_bibtex` — parse a BibTeX string → list of entries.
- :func:`bibtex_to_csl_json` — convert one parsed entry to CSL-JSON.
- :func:`load_bibtex` — parse a file path or string, return a list of
  CSL-JSON dicts ready to hand to :class:`~citeformer.core.Source`.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

__all__ = [
    "BIBTEX_TYPE_MAP",
    "bibtex_to_csl_json",
    "load_bibtex",
    "parse_bibtex",
]

#: BibTeX entry type → CSL 1.0 type. Unmapped types fall back to ``document``.
BIBTEX_TYPE_MAP: dict[str, str] = {
    "article": "article-journal",
    "book": "book",
    "booklet": "book",
    "collection": "book",
    "conference": "paper-conference",
    "inbook": "chapter",
    "incollection": "chapter",
    "inproceedings": "paper-conference",
    "manual": "report",
    "mastersthesis": "thesis",
    "misc": "document",
    "online": "webpage",
    "patent": "patent",
    "phdthesis": "thesis",
    "preprint": "article-journal",
    "proceedings": "book",
    "software": "software",
    "techreport": "report",
    "thesis": "thesis",
    "unpublished": "manuscript",
    "webpage": "webpage",
    "www": "webpage",
}

# BibTeX → CSL-JSON field map. Values are the destination CSL-JSON key. We
# pull the BibTeX-specific handling for author / year / month out of the
# scalar-map path into explicit code below.
_FIELD_MAP: dict[str, str] = {
    "title": "title",
    "journal": "container-title",
    "journaltitle": "container-title",
    "booktitle": "container-title",
    "series": "collection-title",
    "publisher": "publisher",
    "address": "publisher-place",
    "location": "publisher-place",
    "edition": "edition",
    "volume": "volume",
    "number": "issue",
    "pages": "page",
    "doi": "DOI",
    "url": "URL",
    "note": "note",
    "abstract": "abstract",
    "isbn": "ISBN",
    "issn": "ISSN",
    "pmid": "PMID",
    "school": "publisher",
    "institution": "publisher",
    "howpublished": "note",
    "organization": "publisher",
}

_MONTH_ABBREVS: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Regex-driven BibTeX lexing. We walk the input character-by-character for
# entry bodies (nested-brace accounting), but for the entry header we can
# skip to ``@type{key,`` with a regex.
_ENTRY_HEADER = re.compile(r"@(?P<type>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,", re.MULTILINE)


def load_bibtex(source: str | Path) -> list[dict[str, Any]]:
    """Parse a BibTeX file path or string, return a list of CSL-JSON dicts.

    Args:
        source: Either a path-like to a ``.bib`` file or the BibTeX text itself.
            Detection is by ``Path.is_file`` — a filesystem check, so a string
            that happens to resemble a path but doesn't exist is treated as
            BibTeX source.

    Returns:
        A list of CSL-JSON item dicts in document order. Each item has ``id``
        (the BibTeX cite key), ``type`` (mapped from the BibTeX entry type),
        and whichever fields we recognised.
    """
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)
    return [bibtex_to_csl_json(entry) for entry in parse_bibtex(text)]


def parse_bibtex(text: str) -> list[dict[str, Any]]:
    """Parse BibTeX text into a list of raw-field dicts.

    Each dict has keys ``__type`` (entry type, lowercased), ``__key`` (cite
    key), and one entry per field. Values are strings (curly braces and
    quotes stripped). Use :func:`bibtex_to_csl_json` to map to CSL-JSON.

    Args:
        text: Full BibTeX file content.

    Returns:
        List of entries in source order. Entries that fail to parse are
        skipped (no exception raised).
    """
    entries: list[dict[str, Any]] = []
    pos = 0
    length = len(text)
    while pos < length:
        match = _ENTRY_HEADER.search(text, pos)
        if match is None:
            break
        entry_type = match.group("type").lower()
        # @string / @preamble / @comment: skip to the matching close brace.
        if entry_type in {"string", "preamble", "comment"}:
            pos = _skip_balanced_braces(text, match.end() - 1)
            continue
        cite_key = match.group("key")
        body_start = match.end()
        body_end = _find_entry_close(text, body_start)
        if body_end == -1:
            break
        body = text[body_start:body_end]
        fields = _parse_fields(body)
        fields["__type"] = entry_type
        fields["__key"] = cite_key
        entries.append(fields)
        pos = body_end + 1
    return entries


def bibtex_to_csl_json(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert one parsed BibTeX entry (from :func:`parse_bibtex`) to CSL-JSON.

    The BibTeX cite key becomes the CSL ``id``; the BibTeX entry type is
    mapped via :data:`BIBTEX_TYPE_MAP` (unmapped types become
    ``document``). Known fields are renamed; unknown fields land in
    ``custom`` so round-tripping through the adapter is lossless.

    Args:
        entry: A dict from :func:`parse_bibtex` with ``__type`` and
            ``__key`` bookkeeping keys plus raw field values.

    Returns:
        A CSL-JSON item dict.
    """
    out: dict[str, Any] = {
        "id": entry.get("__key", "unknown"),
        "type": BIBTEX_TYPE_MAP.get(entry.get("__type", ""), "document"),
    }
    custom: dict[str, Any] = {}
    year: str | None = None
    month: str | None = None
    for key, value in entry.items():
        if key.startswith("__"):
            continue
        if key == "author":
            out["author"] = _parse_name_list(str(value))
        elif key == "editor":
            out["editor"] = _parse_name_list(str(value))
        elif key == "translator":
            out["translator"] = _parse_name_list(str(value))
        elif key == "year":
            year = str(value).strip()
        elif key == "month":
            month = str(value).strip().lower()
        elif key in _FIELD_MAP:
            out[_FIELD_MAP[key]] = str(value)
        else:
            custom[key] = str(value)

    issued = _compose_issued(year, month)
    if issued is not None:
        out["issued"] = issued

    # Drop empty custom to keep the CSL-JSON tidy.
    if custom:
        out["custom"] = custom
    return out


# --- Internal lexing helpers --------------------------------------------------


def _find_entry_close(text: str, start: int) -> int:
    """Return the index of the outer ``}`` that closes an entry body."""
    depth = 1
    i = start
    in_string = False
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _skip_balanced_braces(text: str, start: int) -> int:
    """Skip past a ``{...}`` block starting at ``text[start] == '{'``."""
    if start >= len(text) or text[start] != "{":
        return start + 1
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def _parse_fields(body: str) -> dict[str, str]:
    """Parse an entry body like ``title = {foo}, year = 2020,`` into a dict.

    The body is the text between the header's ``{key,`` and the closing
    ``}``. We walk key-value pairs, handling both ``{...}`` and ``"..."``
    delimiters with balanced-brace accounting on the former.
    """
    fields: dict[str, str] = {}
    i = 0
    length = len(body)
    while i < length:
        # Skip whitespace and commas.
        while i < length and body[i] in " \t\r\n,":
            i += 1
        if i >= length:
            break
        # Field name: letters / digits / dash / underscore.
        name_start = i
        while i < length and (body[i].isalnum() or body[i] in "-_"):
            i += 1
        name = body[name_start:i].lower()
        if not name:
            break
        # Expect ``=``.
        while i < length and body[i] in " \t\r\n":
            i += 1
        if i >= length or body[i] != "=":
            break
        i += 1
        while i < length and body[i] in " \t\r\n":
            i += 1
        # Value: {...}, "...", or bare token.
        if i >= length:
            break
        if body[i] == "{":
            depth = 0
            j = i
            while j < length:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            value = body[i + 1 : j]
            i = j + 1
        elif body[i] == '"':
            j = i + 1
            while j < length and body[j] != '"':
                if body[j] == "\\" and j + 1 < length:
                    j += 2
                    continue
                j += 1
            value = body[i + 1 : j]
            i = j + 1
        else:
            # Bare numeric or macro name (e.g. ``year = 2020`` or ``month = jan``).
            j = i
            while j < length and body[j] not in " \t\r\n,":
                j += 1
            value = body[i:j]
            i = j
        fields[name] = _clean_value(value)
    return fields


def _clean_value(raw: str) -> str:
    """Normalise a raw BibTeX value — collapse whitespace, strip outer braces."""
    # Collapse any run of whitespace to a single space, preserving brace
    # structure. Trim.
    text = re.sub(r"\s+", " ", raw).strip()
    # Strip one layer of outer braces on text-only values (``{Foo}`` → ``Foo``).
    # We keep braces inside so titles like ``The {XML} Handbook`` don't lose
    # casing hints.
    return text


def _parse_name_list(raw: str) -> list[dict[str, str]]:
    """Parse a BibTeX author/editor list into CSL-JSON name records.

    Handles both ``Family, Given`` and ``Given Family`` conventions; the
    former wins when a comma is present. Splits on ``" and "`` (any case)
    at brace-depth zero so ``{Van Der Berg, Jan}`` stays intact.
    """
    names: list[dict[str, str]] = []
    for raw_name in _split_by_and(raw):
        name = raw_name.strip().strip("{}").strip()
        if not name:
            continue
        if "," in name:
            family, given = (p.strip() for p in name.split(",", 1))
            entry: dict[str, str] = {"family": family}
            if given:
                entry["given"] = given
            names.append(entry)
        else:
            parts = name.split()
            if len(parts) == 1:
                # Sometimes a BibTeX author field has a single token — this
                # is conventionally either an organisation or a single-word
                # surname. Emit as a literal so rendering doesn't rearrange.
                names.append({"literal": parts[0]})
            else:
                family = parts[-1]
                given = " ".join(parts[:-1])
                names.append({"family": family, "given": given})
    return names


def _split_by_and(raw: str) -> list[str]:
    """Split ``Author A and Author B`` on brace-aware, case-insensitive ``and``."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 0
    text = raw.strip()
    lowered = text.lower()
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == "}":
            depth -= 1
            buf.append(ch)
            i += 1
            continue
        if depth == 0 and lowered.startswith(" and ", i):
            parts.append("".join(buf))
            buf = []
            i += len(" and ")
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))
    return parts


def _compose_issued(year: str | None, month: str | None) -> dict[str, list[list[int]]] | None:
    """Combine BibTeX ``year`` / ``month`` fields into CSL ``issued``."""
    if not year:
        return None
    parts: list[int] = []
    try:
        parts.append(int(year))
    except ValueError:
        # Accept "in press" / "forthcoming" by dropping to no issued.
        return None
    if month:
        month_clean = month.strip().lower()
        if month_clean in _MONTH_ABBREVS:
            parts.append(_MONTH_ABBREVS[month_clean])
        else:
            with contextlib.suppress(ValueError):
                parts.append(int(month_clean))
    return {"date-parts": [parts]}
