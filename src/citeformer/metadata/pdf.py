"""PDF metadata + content extraction via pypdf.

Best-effort: pypdf gives us the PDF-info metadata (``/Title``, ``/Author``,
``/CreationDate``, …) and per-page text. Academic PDFs often have these
set — when they don't, we return what we have and leave gaps for the caller
to fill in.

For rigorous scientific-paper parsing (author disambiguation, reference
list extraction, section hierarchies) consider wrapping an external tool
like GROBID or docling. pypdf is fast and dependency-light; the trade-off
is that it's not ML-driven so it can't rescue malformed PDFs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader


def _parse_pdf_date(raw: str) -> int | None:
    """Extract a 4-digit year from a PDF ``/CreationDate`` string.

    PDF dates look like ``D:YYYYMMDDHHmmSSOHH'mm'``. We only want the year.
    """
    raw = raw.strip()
    if raw.startswith("D:") and len(raw) >= 6:
        try:
            return int(raw[2:6])
        except ValueError:
            return None
    # Sometimes the date is plain ISO-ish; grab the first 4 digits.
    if len(raw) >= 4 and raw[:4].isdigit():
        return int(raw[:4])
    return None


def _parse_pdf_authors(raw: str) -> list[dict[str, str]]:
    """Split a PDF ``/Author`` string into a list of CSL-JSON author dicts.

    PDFs have no convention for author separator, so we try semicolons,
    commas, and the word "and". Authors are returned as CSL ``literal`` names
    rather than split family/given — the PDF string is too ambiguous.
    """
    raw = raw.strip()
    if not raw:
        return []
    for sep in [";", "\n", " and ", ","]:
        if sep in raw:
            return [{"literal": n.strip()} for n in raw.split(sep) if n.strip()]
    return [{"literal": raw}]


def extract_pdf(path: str | Path) -> tuple[dict[str, Any], str]:
    """Extract CSL-JSON metadata + full text from a PDF.

    Args:
        path: Filesystem path to the PDF.

    Returns:
        ``(metadata, content)``. ``metadata`` is a CSL-JSON dict with at
        least ``id``, ``type`` (``"report"`` as a safe default), and
        ``title``. ``author`` and ``issued`` are included when the PDF info
        dict provides them. ``content`` is the concatenated per-page text.

    Raises:
        FileNotFoundError: If ``path`` doesn't exist.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path!s}")

    reader = PdfReader(str(path))
    info: Any = reader.metadata or {}

    title_raw = info.get("/Title") if info else None
    title = title_raw.strip() if isinstance(title_raw, str) and title_raw.strip() else path.stem

    author_raw = info.get("/Author") if info else None
    authors = _parse_pdf_authors(author_raw) if isinstance(author_raw, str) and author_raw else []

    date_raw = info.get("/CreationDate") if info else None
    year = _parse_pdf_date(date_raw) if isinstance(date_raw, str) else None

    metadata: dict[str, Any] = {
        "id": f"pdf-{path.stem}",
        # "report" is a conservative CSL type when we don't know whether
        # it's an article / book / thesis. Users who know better should
        # override after calling `from_pdf`.
        "type": "report",
        "title": title,
    }
    if authors:
        metadata["author"] = authors
    if year is not None:
        metadata["issued"] = {"date-parts": [[year]]}

    content_parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            # pypdf throws a variety of exceptions on malformed pages;
            # skip rather than abort the whole extraction.
            continue
        if text:
            content_parts.append(text)
    content = "\n".join(content_parts)

    return metadata, content
