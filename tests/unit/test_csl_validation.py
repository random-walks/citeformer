"""Tests for the opt-in CSL-JSON validator.

The validator is the §10.2 enforcement tool users can invoke before
constructing a ``Source`` when they want the schema checked up front.
Tests lock the boundary between "error" (hard contract violation) and
"warning" (soft / forward-compat signal) that downstream users will
rely on.
"""

from __future__ import annotations

import pytest

from citeformer import (
    CSLValidationError,
    validate_csl_json,
    validate_source_metadata,
)


def _minimal(**extra) -> dict:  # type: ignore[no-untyped-def]
    base = {"id": "x", "type": "article-journal", "title": "X"}
    base.update(extra)
    return base


def test_minimal_item_validates_clean() -> None:
    report = validate_csl_json(_minimal())
    assert report.ok
    assert report.errors == []
    assert report.warnings == []


def test_missing_id_is_error() -> None:
    report = validate_csl_json({"type": "book", "title": "x"})
    assert not report.ok
    assert any("'id'" in e for e in report.errors)


def test_missing_type_is_error() -> None:
    report = validate_csl_json({"id": "x", "title": "x"})
    assert not report.ok
    assert any("'type'" in e for e in report.errors)


def test_unknown_type_is_warning_by_default() -> None:
    report = validate_csl_json(_minimal(type="imaginary-kind"))
    assert report.ok  # still renders (dispatches to default formatter)
    assert any("imaginary-kind" in w for w in report.warnings)


def test_unknown_type_is_error_when_strict() -> None:
    report = validate_csl_json(_minimal(type="imaginary-kind"), strict_types=True)
    assert not report.ok
    assert any("imaginary-kind" in e for e in report.errors)


def test_wrong_type_for_known_field_is_error() -> None:
    report = validate_csl_json(_minimal(DOI=12345))
    assert not report.ok
    assert any("DOI" in e for e in report.errors)


def test_unknown_top_level_field_is_warning() -> None:
    report = validate_csl_json(_minimal(zoltar="pending"))
    assert report.ok
    assert any("zoltar" in w for w in report.warnings)


def test_author_must_be_list() -> None:
    report = validate_csl_json(_minimal(author="Vaswani"))
    assert not report.ok
    assert any("author" in e for e in report.errors)


def test_author_entry_missing_family_and_literal_is_warning() -> None:
    report = validate_csl_json(_minimal(author=[{"given": "A."}]))
    assert report.ok
    assert any("family" in w or "literal" in w for w in report.warnings)


def test_author_entry_with_family_validates() -> None:
    report = validate_csl_json(_minimal(author=[{"family": "Vaswani"}]))
    assert report.ok
    assert report.warnings == []


def test_author_entry_with_literal_validates() -> None:
    report = validate_csl_json(_minimal(author=[{"literal": "OpenAI"}]))
    assert report.ok


def test_issued_without_date_parts_warns() -> None:
    report = validate_csl_json(_minimal(issued={}))
    assert report.ok
    assert any("date-parts" in w for w in report.warnings)


def test_issued_wrong_type_errors() -> None:
    report = validate_csl_json(_minimal(issued="2023"))
    assert not report.ok


def test_raise_on_error_raises_on_errors_only() -> None:
    # Warnings shouldn't raise.
    report = validate_csl_json(_minimal(zoltar="x"), raise_on_error=True)
    assert report.ok
    # Missing id is an error — should raise.
    with pytest.raises(CSLValidationError, match="'id'"):
        validate_csl_json({"type": "book", "title": "x"}, raise_on_error=True)


def test_non_dict_input_errors_cleanly() -> None:
    report = validate_csl_json("not a dict")  # type: ignore[arg-type]
    assert not report.ok
    assert any("dict" in e for e in report.errors)


def test_validate_source_metadata_wraps() -> None:
    # Just sugar over validate_csl_json.
    report = validate_source_metadata(_minimal())
    assert report.ok


def test_integration_keys_are_known() -> None:
    """LangChain/LlamaIndex adapters stash extras under `_langchain_metadata`
    and `_llamaindex_metadata` — those shouldn't trigger warnings.
    """
    report = validate_csl_json(
        _minimal(_langchain_metadata={"chunk": 5}, _llamaindex_metadata={"score": 0.8})
    )
    assert report.ok
    assert report.warnings == []
