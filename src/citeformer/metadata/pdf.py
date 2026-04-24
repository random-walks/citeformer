"""PDF metadata + content extraction.

Two extractor backends:

- ``"pypdf"`` (default, zero-dep): reads the PDF-info metadata
  (``/Title``, ``/Author``, ``/CreationDate``, …) and per-page text.
  Fast and always available; quality depends on the producer of the
  PDF. Academic PDFs often have these fields set — when they don't, we
  return what we have and leave gaps for the caller to fill in.

- ``"grobid"`` (optional, ``pip install citeformer[grobid]``): wraps
  [GROBID](https://github.com/kermitt2/grobid), an ML-based scientific-
  paper parser that returns structured TEI-XML with clean
  author/title/abstract fields and section-level body text. Requires a
  GROBID server reachable at ``grobid_url`` (defaults to
  ``http://localhost:8070``). The typical dev setup is::

      docker run -p 8070:8070 grobid/grobid:0.8.0

  GROBID extraction is ~5-10× slower than pypdf on a first call but
  produces substantially cleaner output for downstream NLI scoring —
  see ``benchmarks/README.md`` Finding 3 for the quality gap.

``Source.from_pdf(path, extractor="grobid")`` forwards to this module;
direct callers can use :func:`extract_pdf` and pass ``extractor=``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pypdf import PdfReader

DEFAULT_GROBID_URL = "http://localhost:8070"


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


def _extract_with_pypdf(path: Path) -> tuple[dict[str, Any], str]:
    """Original zero-dep extractor. Returns CSL-JSON metadata + concatenated page text."""
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


def _extract_with_grobid(
    path: Path,
    *,
    grobid_url: str = DEFAULT_GROBID_URL,
    timeout: int = 60,
) -> tuple[dict[str, Any], str]:
    """ML-based extractor via GROBID.

    Uses the ``grobid_client_python`` library's ``Client.process_text_file``
    entry point (named ``process`` / ``process_pdf`` across versions — we
    probe which one is available to tolerate both). Parses the returned
    TEI-XML with ``xml.etree.ElementTree`` (stdlib — no lxml requirement)
    to pull out:

    - title → ``metadata["title"]``
    - authors (forename + surname) → ``metadata["author"]``
    - publication date → ``metadata["issued"]``
    - abstract → ``metadata["abstract"]``
    - body paragraphs → joined into ``content``

    Raises ``RuntimeError`` with the GROBID response code if the service
    is unreachable or returns a non-200. Callers should catch and fall
    back to pypdf if graceful degradation is desired (``Source.from_pdf``
    does not; it surfaces the error so the user sees what's happening).
    """
    try:
        # The package name on PyPI is ``grobid-client-python``, but the
        # import module is ``grobid_client.grobid_client``. We try both in
        # case downstream environments re-export.
        try:
            from grobid_client.grobid_client import GrobidClient
        except ImportError:
            from grobid_client import GrobidClient
    except ImportError as e:
        raise ImportError(
            "The GROBID extractor requires the `grobid` extra. "
            "Install with `pip install citeformer[grobid]`."
        ) from e

    client = GrobidClient(grobid_server=grobid_url, timeout=timeout)
    # grobid_client_python returns (path, status_code, xml_text) or similar
    # depending on version. Support the common shapes.
    result = client.process_pdf(
        service="processFulltextDocument",
        pdf_file=str(path),
        generateIDs=False,
        consolidate_header=True,
        consolidate_citations=False,
        include_raw_citations=False,
        include_raw_affiliations=False,
        tei_coordinates=False,
        segment_sentences=False,
    )
    # Normalise to (status, xml_text).
    if isinstance(result, tuple) and len(result) >= 3:
        _, status, xml = result[0], result[1], result[2]
    elif isinstance(result, tuple) and len(result) == 2:
        status, xml = result
    else:
        status, xml = 200, str(result)

    if status and int(status) != 200:
        raise RuntimeError(f"GROBID at {grobid_url} returned status {status} on {path.name}")
    if not xml:
        raise RuntimeError(f"GROBID returned empty body for {path.name}")

    return _tei_to_csl(str(xml), path=path)


# --- TEI parsing --------------------------------------------------------


_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _tei_to_csl(xml_text: str, *, path: Path) -> tuple[dict[str, Any], str]:
    """Convert GROBID TEI-XML into (CSL-JSON metadata, content string)."""
    from xml.etree import ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise RuntimeError(f"Could not parse GROBID TEI for {path.name}: {e}") from e

    metadata: dict[str, Any] = {
        "id": f"pdf-{path.stem}",
        "type": "article-journal",  # GROBID is scientific-paper-centric.
    }

    # Title
    title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", _TEI_NS)
    title = (title_el.text or "").strip() if title_el is not None and title_el.text else path.stem
    metadata["title"] = title

    # Authors: biblStruct/analytic/author or sourceDesc//author
    authors: list[dict[str, str]] = []
    for author_el in root.findall(".//tei:teiHeader//tei:sourceDesc//tei:author", _TEI_NS):
        pers = author_el.find("tei:persName", _TEI_NS)
        if pers is None:
            continue
        forename = pers.find("tei:forename", _TEI_NS)
        surname = pers.find("tei:surname", _TEI_NS)
        family = (surname.text or "").strip() if surname is not None and surname.text else ""
        given = (forename.text or "").strip() if forename is not None and forename.text else ""
        if family and given:
            authors.append({"family": family, "given": given})
        elif family:
            authors.append({"family": family})
    if authors:
        metadata["author"] = authors

    # Date: monogr/imprint/date or publicationStmt/date
    for xpath in (
        ".//tei:teiHeader//tei:monogr//tei:imprint/tei:date",
        ".//tei:teiHeader//tei:publicationStmt/tei:date",
    ):
        date_el = root.find(xpath, _TEI_NS)
        if date_el is not None:
            when = date_el.attrib.get("when") or (date_el.text or "")
            if when and when[:4].isdigit():
                metadata["issued"] = {"date-parts": [[int(when[:4])]]}
                break

    # Abstract (goes into metadata; body paragraphs go into content)
    abstract_el = root.find(".//tei:profileDesc/tei:abstract", _TEI_NS)
    if abstract_el is not None:
        abstract_text = " ".join(abstract_el.itertext()).strip()
        if abstract_text:
            metadata["abstract"] = abstract_text

    # Body: collect all paragraph text from <text><body>
    paragraphs: list[str] = []
    for p in root.findall(".//tei:text/tei:body//tei:p", _TEI_NS):
        text = " ".join(p.itertext()).strip()
        if text:
            paragraphs.append(text)
    content = "\n\n".join(paragraphs)

    return metadata, content


def extract_pdf(
    path: str | Path,
    *,
    extractor: Literal["pypdf", "grobid"] = "pypdf",
    grobid_url: str = DEFAULT_GROBID_URL,
    **kwargs: Any,
) -> tuple[dict[str, Any], str]:
    """Extract CSL-JSON metadata + body text from a PDF.

    Args:
        path: Filesystem path to the PDF.
        extractor: ``"pypdf"`` (default; fast, zero-dep) or ``"grobid"``
            (ML-based, needs a running GROBID server).
        grobid_url: GROBID server URL. Defaults to
            ``http://localhost:8070``. Ignored for ``extractor="pypdf"``.
        **kwargs: Extractor-specific options (currently ``timeout`` on
            GROBID).

    Returns:
        ``(metadata, content)``. ``metadata`` is a CSL-JSON dict with at
        least ``id``, ``type``, and ``title``. ``author``, ``issued``,
        and ``abstract`` (GROBID only) are included when extractable.
        ``content`` is the joined body text.

    Raises:
        FileNotFoundError: If ``path`` doesn't exist.
        ImportError: If ``extractor="grobid"`` and the ``grobid`` extra
            isn't installed.
        RuntimeError: If ``extractor="grobid"`` and the server is
            unreachable / returns non-200.
        ValueError: If ``extractor`` isn't one of the supported values.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path!s}")

    if extractor == "pypdf":
        return _extract_with_pypdf(path)
    if extractor == "grobid":
        return _extract_with_grobid(path, grobid_url=grobid_url, **kwargs)
    raise ValueError(f"Unknown extractor {extractor!r}; expected 'pypdf' or 'grobid'.")
