"""Tests for the GROBID extractor path in `citeformer.metadata.pdf`.

Running an actual GROBID server inside CI is a docker-install-level
dependency we don't want to take on. Instead:

- TEI-XML parsing is tested with a minimal hand-crafted TEI document.
- The GROBID dispatch in :func:`extract_pdf` is tested by monkey-
  patching the `grobid_client` import to a stub that returns our
  hand-crafted XML — the same integration surface without needing
  a running server.

A real integration test against a local GROBID server is left as an
opt-in check behind a ``GROBID_URL`` env var; adding it is a one-liner
if the project ever wires up a docker step in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from citeformer import Source
from citeformer.metadata.pdf import _tei_to_csl, extract_pdf

_MINIMAL_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title level="a" type="main">Attention Is All You Need</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName><forename type="first">Ashish</forename><surname>Vaswani</surname></persName>
            </author>
            <author>
              <persName><forename type="first">Noam</forename><surname>Shazeer</surname></persName>
            </author>
            <author>
              <persName><surname>Parmar</surname></persName>
            </author>
          </analytic>
          <monogr>
            <imprint>
              <date type="published" when="2017-12-06"/>
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>We propose a new simple network architecture based solely on attention.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><p>The dominant sequence transduction models use recurrent networks.</p></div>
      <div><p>We propose the Transformer, a model architecture eschewing recurrence.</p></div>
    </body>
  </text>
</TEI>
"""


# --- Pure TEI parsing --------------------------------------------------------


def test_tei_to_csl_extracts_title(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    metadata, _ = _tei_to_csl(_MINIMAL_TEI, path=path)
    assert metadata["title"] == "Attention Is All You Need"


def test_tei_to_csl_extracts_all_three_authors(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    metadata, _ = _tei_to_csl(_MINIMAL_TEI, path=path)
    authors = metadata["author"]
    assert authors == [
        {"family": "Vaswani", "given": "Ashish"},
        {"family": "Shazeer", "given": "Noam"},
        {"family": "Parmar"},  # No given name → family-only
    ]


def test_tei_to_csl_extracts_issued_year(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    metadata, _ = _tei_to_csl(_MINIMAL_TEI, path=path)
    assert metadata["issued"] == {"date-parts": [[2017]]}


def test_tei_to_csl_extracts_abstract(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    metadata, _ = _tei_to_csl(_MINIMAL_TEI, path=path)
    assert "attention" in metadata["abstract"].lower()


def test_tei_to_csl_body_contains_both_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _, content = _tei_to_csl(_MINIMAL_TEI, path=path)
    assert "recurrent networks" in content
    assert "eschewing recurrence" in content


def test_tei_to_csl_falls_back_to_stem_when_no_title(tmp_path: Path) -> None:
    path = tmp_path / "some-paper.pdf"
    minimal_no_title = "<TEI xmlns='http://www.tei-c.org/ns/1.0'><teiHeader/></TEI>"
    metadata, _ = _tei_to_csl(minimal_no_title, path=path)
    assert metadata["title"] == "some-paper"


def test_tei_to_csl_raises_on_malformed_xml(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    with pytest.raises(RuntimeError, match="Could not parse GROBID TEI"):
        _tei_to_csl("<not-valid-xml", path=path)


# --- extract_pdf dispatch --------------------------------------------------


def test_extract_pdf_rejects_unknown_extractor(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    with pytest.raises(ValueError, match="Unknown extractor"):
        extract_pdf(pdf, extractor="docling")  # type: ignore[arg-type]


def test_extract_pdf_default_still_pypdf(tmp_path: Path) -> None:
    """Backwards-compat: calling without ``extractor=`` must go through pypdf.

    We fabricate a minimal PDF the pypdf reader will accept. Real content
    isn't the point — the test verifies the default route isn't GROBID.
    """
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(
        b"%PDF-1.1\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 1 1]>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000050 00000 n\n0000000090 00000 n\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n140\n%%EOF\n"
    )
    metadata, _ = extract_pdf(pdf)
    # pypdf-extracted PDFs get the "report" type; GROBID uses "article-journal".
    assert metadata["type"] == "report"


def test_extract_pdf_grobid_dispatches_to_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: `extract_pdf(path, extractor='grobid')` → TEI parse → CSL.

    We install a stub `grobid_client.grobid_client` module whose
    `GrobidClient.process_pdf` returns our fixture TEI. Then we call
    `extract_pdf` and verify the result is what the TEI encodes.
    """
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-fake%%EOF\n")

    captured: dict[str, Any] = {}

    class _StubClient:
        def __init__(self, grobid_server: str = "", timeout: int = 60) -> None:
            captured["server"] = grobid_server
            captured["timeout"] = timeout

        def process_pdf(self, **kwargs: Any) -> tuple[str, int, str]:
            captured["kwargs"] = kwargs
            return ("paper.pdf", 200, _MINIMAL_TEI)

    # Build the fake namespace: `grobid_client.grobid_client.GrobidClient`
    fake_pkg = ModuleType("grobid_client")
    fake_sub = ModuleType("grobid_client.grobid_client")
    fake_sub.GrobidClient = _StubClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "grobid_client", fake_pkg)
    monkeypatch.setitem(sys.modules, "grobid_client.grobid_client", fake_sub)

    metadata, content = extract_pdf(pdf, extractor="grobid", grobid_url="http://test:8070")
    assert metadata["title"] == "Attention Is All You Need"
    assert metadata["author"][0] == {"family": "Vaswani", "given": "Ashish"}
    assert metadata["issued"] == {"date-parts": [[2017]]}
    assert "recurrent networks" in content
    # Ensure we passed the URL through.
    assert captured["server"] == "http://test:8070"


def test_extract_pdf_grobid_raises_on_non_200(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-fake%%EOF\n")

    class _StubClient:
        def __init__(self, **_: Any) -> None:
            pass

        def process_pdf(self, **_: Any) -> tuple[str, int, str]:
            return ("paper.pdf", 503, "")

    fake_pkg = ModuleType("grobid_client")
    fake_sub = ModuleType("grobid_client.grobid_client")
    fake_sub.GrobidClient = _StubClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "grobid_client", fake_pkg)
    monkeypatch.setitem(sys.modules, "grobid_client.grobid_client", fake_sub)

    with pytest.raises(RuntimeError, match="status 503"):
        extract_pdf(pdf, extractor="grobid")


def test_extract_pdf_grobid_raises_when_extra_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `grobid_client` isn't installed, we should surface a helpful ImportError."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-fake%%EOF\n")

    # Prevent any installed stub from resolving.
    monkeypatch.setitem(sys.modules, "grobid_client", None)
    monkeypatch.setitem(sys.modules, "grobid_client.grobid_client", None)

    with pytest.raises(ImportError, match="citeformer\\[grobid\\]"):
        extract_pdf(pdf, extractor="grobid")


def test_source_from_pdf_forwards_extractor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Source.from_pdf(path, extractor='grobid')` threads through to extract_pdf."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-fake%%EOF\n")

    class _StubClient:
        def __init__(self, **_: Any) -> None:
            pass

        def process_pdf(self, **_: Any) -> tuple[str, int, str]:
            return ("paper.pdf", 200, _MINIMAL_TEI)

    fake_pkg = ModuleType("grobid_client")
    fake_sub = ModuleType("grobid_client.grobid_client")
    fake_sub.GrobidClient = _StubClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "grobid_client", fake_pkg)
    monkeypatch.setitem(sys.modules, "grobid_client.grobid_client", fake_sub)

    source = Source.from_pdf(pdf, extractor="grobid")
    assert source.metadata["title"] == "Attention Is All You Need"
    assert source.metadata["type"] == "article-journal"


def test_source_from_pdf_default_extractor_unchanged(tmp_path: Path) -> None:
    """No breaking change: Source.from_pdf() with no extractor arg still works."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(
        b"%PDF-1.1\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 1 1]>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000050 00000 n\n0000000090 00000 n\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n140\n%%EOF\n"
    )
    source = Source.from_pdf(pdf)
    assert source.metadata["type"] == "report"  # pypdf path signature
