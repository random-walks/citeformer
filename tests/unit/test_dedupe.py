"""Tests for `citeformer.deduplicate_adjacent_cites`.

Locks the "collapse runs of `[N]` markers to unique-first-order" semantics
so downstream tools that want to clean up REQUIRED-policy stacking can
rely on the shape.
"""

from __future__ import annotations

import pytest

from citeformer import deduplicate_adjacent_cites


def test_dedupes_simple_stacking() -> None:
    assert (
        deduplicate_adjacent_cites("Foo [1] [2] [3] [1] [2] [3] [1].")
        == "Foo [1] [2] [3]."
    )


def test_single_cite_untouched() -> None:
    assert deduplicate_adjacent_cites("Foo [1].") == "Foo [1]."


def test_two_distinct_cites_in_a_row_kept() -> None:
    assert deduplicate_adjacent_cites("Foo [1] [2].") == "Foo [1] [2]."


def test_two_separate_runs_each_deduped() -> None:
    text = "Alpha [1] [2] [1]. Beta [3] [4] [3] [4]."
    assert deduplicate_adjacent_cites(text) == "Alpha [1] [2]. Beta [3] [4]."


def test_run_followed_by_non_cite_content() -> None:
    text = "Foo [1] [2] [3] [1] explains it."
    assert deduplicate_adjacent_cites(text) == "Foo [1] [2] [3] explains it."


def test_preserves_order_of_first_appearance() -> None:
    assert (
        deduplicate_adjacent_cites("[3] [1] [2] [3] [1].")
        == "[3] [1] [2]."
    )


def test_respects_non_adjacent_cites_separated_by_content() -> None:
    """Runs are only collapsed if cites are separated by whitespace only —
    intervening words should keep each cite isolated.
    """
    text = "Foo [1] and also Bar [2]. Baz [1] [2] [1]."
    assert (
        deduplicate_adjacent_cites(text)
        == "Foo [1] and also Bar [2]. Baz [1] [2]."
    )


def test_empty_string_roundtrips() -> None:
    assert deduplicate_adjacent_cites("") == ""


def test_no_cites_at_all_is_identity() -> None:
    assert deduplicate_adjacent_cites("The quick brown fox jumps.") == "The quick brown fox jumps."


@pytest.mark.parametrize(
    "before,after",
    [
        ("[1][2][3][1]", "[1] [2] [3]"),  # no spaces between markers
        ("[1]  [2]   [3]  [1]", "[1] [2] [3]"),  # extra whitespace collapses too
        ("[1]\n[2]\n[1]", "[1] [2]"),  # newline separator
    ],
)
def test_whitespace_normalization(before: str, after: str) -> None:
    assert deduplicate_adjacent_cites(before) == after
