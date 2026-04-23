"""Tests for `citeformer.render.styles` — bundled styles, loader, classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from citeformer.render.styles import (
    bundled_style_names,
    load_style,
    resolve_style_path,
    style_citation_format,
)

_EXPECTED_BUNDLED = ["apa-7", "mla-9", "chicago-author-date", "ieee", "nature"]


def test_bundled_style_names_are_stable_list() -> None:
    assert bundled_style_names() == _EXPECTED_BUNDLED


def test_resolve_bundled_names_returns_existing_files() -> None:
    for name in _EXPECTED_BUNDLED:
        path = resolve_style_path(name)
        assert path.exists()
        assert path.suffix == ".csl"


def test_resolve_bundled_aliases_share_the_same_file() -> None:
    assert resolve_style_path("apa") == resolve_style_path("apa-7")
    assert resolve_style_path("mla") == resolve_style_path("mla-9")
    assert resolve_style_path("chicago") == resolve_style_path("chicago-author-date")


def test_resolve_raises_on_unknown_name() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_style_path("nonexistent-style-that-will-never-exist")


def test_resolve_accepts_absolute_path(tmp_path: Path) -> None:
    # Build a minimal valid CSL file and resolve it.
    fake = tmp_path / "fake.csl"
    fake.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<style xmlns="http://purl.org/net/xbiblio/csl" class="in-text" '
        'version="1.0" default-locale="en-US">'
        "<info><id>http://fake</id><title>Fake</title>"
        '<category citation-format="numeric"/>'
        "<updated>2026-04-23T00:00:00+00:00</updated></info>"
        "<citation><layout/></citation></style>"
    )
    resolved = resolve_style_path(str(fake))
    assert resolved == fake.resolve()


def test_load_style_is_cached() -> None:
    style_a = load_style("apa-7")
    style_b = load_style("apa-7")
    # lru_cache → same object identity on repeat calls with the same key.
    assert style_a is style_b


def test_style_citation_format_classifies_bundled_styles() -> None:
    # APA and Chicago are author-date; MLA uses "author" (author + page,
    # not year); IEEE and Nature are numeric.
    assert style_citation_format("apa-7") == "author-date"
    assert style_citation_format("mla-9") == "author"
    assert style_citation_format("chicago-author-date") == "author-date"
    assert style_citation_format("ieee") == "numeric"
    assert style_citation_format("nature") == "numeric"
