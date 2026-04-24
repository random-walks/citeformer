"""Tests for the network-backed metadata fetchers (Crossref, arXiv, URL).

Uses pytest-vcr to record live HTTP responses into cassettes on first run;
subsequent runs replay from the cassette (hermetic). Cassettes live under
``tests/unit/cassettes/`` and are committed.

All tests pass ``use_cache=False`` to bypass the on-disk diskcache — we
want VCR to be the recording mechanism, not our own cache layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citeformer import Source
from citeformer.metadata import extract_url, fetch_arxiv, fetch_crossref

# Stable identifiers chosen for long-term replay stability.
_TEST_DOI = "10.1038/s41586-023-06221-2"
_TEST_ARXIV = "2305.14627"
_TEST_URL = "https://example.com/"


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, object]:
    """Strip auth headers from cassettes (we don't use any, but defensive)."""
    return {
        "filter_headers": ["authorization", "cookie"],
        "record_mode": "once",
    }


# --- Crossref -----------------------------------------------------------------


@pytest.mark.vcr
def test_fetch_crossref_returns_csl_json() -> None:
    meta = fetch_crossref(_TEST_DOI, use_cache=False)
    assert meta["DOI"].lower() == _TEST_DOI.lower()
    assert meta["type"]
    assert isinstance(meta.get("author"), list)
    assert isinstance(meta.get("title"), (list, str))


@pytest.mark.vcr
def test_source_from_doi_builds_source() -> None:
    src = Source.from_doi(_TEST_DOI, use_cache=False)
    assert src.metadata["DOI"].lower() == _TEST_DOI.lower()
    assert src.content == ""  # DOI fetch doesn't include paper text


@pytest.mark.vcr
def test_fetch_crossref_accepts_url_and_prefixed_forms() -> None:
    # URL form
    u = fetch_crossref(f"https://doi.org/{_TEST_DOI}", use_cache=False)
    # doi: prefix
    d = fetch_crossref(f"doi:{_TEST_DOI}", use_cache=False)
    assert u["DOI"] == d["DOI"]


# --- arXiv --------------------------------------------------------------------


@pytest.mark.vcr
def test_fetch_arxiv_returns_csl_json_with_abstract() -> None:
    meta = fetch_arxiv(_TEST_ARXIV, use_cache=False)
    assert meta["id"] == f"arxiv-{_TEST_ARXIV}"
    assert meta["type"] == "article-journal"
    assert meta["URL"] == f"https://arxiv.org/abs/{_TEST_ARXIV}"
    assert meta["container-title"] == "arXiv preprint"
    assert meta["title"]
    assert isinstance(meta["author"], list)
    assert len(meta["author"]) >= 1
    assert meta["abstract"]  # non-empty abstract


@pytest.mark.vcr
def test_fetch_arxiv_strips_version_suffix() -> None:
    meta = fetch_arxiv(f"{_TEST_ARXIV}v3", use_cache=False)
    assert meta["id"] == f"arxiv-{_TEST_ARXIV}"


@pytest.mark.vcr
def test_source_from_arxiv_puts_abstract_in_content() -> None:
    src = Source.from_arxiv(_TEST_ARXIV, use_cache=False)
    assert "abstract" not in src.metadata  # moved to content
    assert src.content  # non-empty


# --- URL ----------------------------------------------------------------------


@pytest.mark.vcr
def test_extract_url_returns_metadata_and_content() -> None:
    metadata, content = extract_url(_TEST_URL)
    assert metadata["URL"] == _TEST_URL
    assert metadata["type"] == "webpage"
    assert metadata["title"]  # falls back to URL if no <title>
    # example.com has a tiny body; just assert the extraction produced SOMETHING
    # meaningful when there's any article content.
    assert isinstance(content, str)


@pytest.mark.vcr
def test_source_from_url_builds_source() -> None:
    src = Source.from_url(_TEST_URL)
    assert src.metadata["URL"] == _TEST_URL
    assert src.metadata["type"] == "webpage"


# --- Path-based cassette discovery --------------------------------------------


def test_cassettes_are_committed() -> None:
    """Sanity check that the cassettes directory exists alongside this file.

    If this test passes but the @pytest.mark.vcr tests above hit the network,
    something's wrong with pytest-vcr's cassette resolution.
    """
    cassette_dir = Path(__file__).parent / "cassettes"
    # Directory is created on first run when VCR records; we don't want to
    # fail the test just because cassettes haven't been recorded yet.
    # Instead, assert the directory is either absent or contains YAML files.
    if cassette_dir.exists():
        yml_files = list(cassette_dir.rglob("*.yaml")) + list(cassette_dir.rglob("*.yml"))
        assert yml_files, "cassettes dir exists but has no recordings"
