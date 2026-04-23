"""Tests for `citeformer.metadata.pdf` — pypdf-based extraction.

We generate a minimal PDF at test time (pypdf's `PdfWriter`) with predictable
metadata, then assert `extract_pdf` recovers it. Content extraction from a
blank page returns empty text — that's expected and asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from citeformer import Source
from citeformer.metadata import extract_pdf


@pytest.fixture
def pdf_with_full_metadata(tmp_path: Path) -> Path:
    """A minimal PDF with ``/Title``, ``/Author``, ``/CreationDate`` set."""
    path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.add_metadata(
        {
            "/Title": "A Study of Mock PDFs",
            "/Author": "Poe, Edgar A.; Melville, Herman",
            "/CreationDate": "D:20260423120000Z",
        }
    )
    with path.open("wb") as f:
        writer.write(f)
    return path


@pytest.fixture
def pdf_with_no_metadata(tmp_path: Path) -> Path:
    """A PDF without any info dict fields — worst case for extraction."""
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    with path.open("wb") as f:
        writer.write(f)
    return path


def test_extract_pdf_recovers_title_authors_year(pdf_with_full_metadata: Path) -> None:
    metadata, content = extract_pdf(pdf_with_full_metadata)
    assert metadata["title"] == "A Study of Mock PDFs"
    assert metadata["type"] == "report"
    assert metadata["id"] == "pdf-sample"
    assert metadata["author"] == [
        {"literal": "Poe, Edgar A."},
        {"literal": "Melville, Herman"},
    ]
    assert metadata["issued"] == {"date-parts": [[2026]]}
    # Blank page → no text; content is empty string.
    assert content == ""


def test_extract_pdf_falls_back_to_stem_when_no_title(pdf_with_no_metadata: Path) -> None:
    metadata, content = extract_pdf(pdf_with_no_metadata)
    assert metadata["title"] == "blank"
    assert metadata["id"] == "pdf-blank"
    assert "author" not in metadata
    assert "issued" not in metadata
    assert content == ""


def test_extract_pdf_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises(FileNotFoundError, match=r"does-not-exist\.pdf"):
        extract_pdf(missing)


def test_source_from_pdf_classmethod(pdf_with_full_metadata: Path) -> None:
    """`Source.from_pdf` is the public entry point."""
    src = Source.from_pdf(pdf_with_full_metadata)
    assert src.metadata["title"] == "A Study of Mock PDFs"
    assert src.content == ""
