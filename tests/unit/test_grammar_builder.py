"""Tests for `citeformer.grammar.builder` — §10.1 contract snapshot + semantics.

Grammar outputs are pinned via `pytest-regressions` so any change to the
`CITE_ID` terminal shape or the per-policy body is surfaced as a snapshot
diff — which forces the §10.1 ceremony in `docs/reference/contracts.md`.
"""

from __future__ import annotations

import pytest

from citeformer import Policy
from citeformer.grammar import Grammar, build_grammar, parse_ok

# --- Snapshots (pinned §10.1 contract outputs) --------------------------------


def test_grammar_required_snapshot(data_regression) -> None:  # type: ignore[no-untyped-def]
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED)
    data_regression.check(_grammar_dict(g))


def test_grammar_auto_snapshot(data_regression) -> None:  # type: ignore[no-untyped-def]
    g = build_grammar(n_sources=3, policy=Policy.AUTO)
    data_regression.check(_grammar_dict(g))


def test_grammar_quotes_only_snapshot(data_regression) -> None:  # type: ignore[no-untyped-def]
    g = build_grammar(n_sources=3, policy=Policy.QUOTES_ONLY)
    data_regression.check(_grammar_dict(g))


def test_grammar_scales_cite_id_with_n_sources(data_regression) -> None:  # type: ignore[no-untyped-def]
    """Dynamic CITE_ID enum — §10.1 requires `"1" | ... | "N"` per call."""
    g = build_grammar(n_sources=10, policy=Policy.REQUIRED)
    data_regression.check(_grammar_dict(g))


# --- CITE_ID terminal shape (the load-bearing §10.1 invariant) ----------------


def test_cite_id_terminal_is_bracket_wrapped_enum() -> None:
    """§10.1: `CITE_ID: "[" ("1" | "2" | ... | "N") "]"` — verbatim."""
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED)
    assert 'CITE_ID: "[" ("1" | "2" | "3") "]"' in g.ebnf


def test_cite_id_terminal_scales_to_large_n() -> None:
    g = build_grammar(n_sources=15, policy=Policy.AUTO)
    assert (
        'CITE_ID: "[" ("1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | '
        '"9" | "10" | "11" | "12" | "13" | "14" | "15") "]"'
    ) in g.ebnf


def test_cite_ids_metadata_matches_n_sources() -> None:
    g = build_grammar(n_sources=7, policy=Policy.AUTO)
    assert g.cite_ids == (1, 2, 3, 4, 5, 6, 7)


def test_grammar_policy_roundtrip() -> None:
    g = build_grammar(n_sources=3, policy=Policy.QUOTES_ONLY)
    assert g.policy is Policy.QUOTES_ONLY


# --- Input validation ---------------------------------------------------------


def test_build_grammar_rejects_zero_sources() -> None:
    with pytest.raises(ValueError, match="n_sources must be >= 1"):
        build_grammar(n_sources=0, policy=Policy.REQUIRED)


def test_build_grammar_rejects_negative_sources() -> None:
    with pytest.raises(ValueError, match="n_sources must be >= 1"):
        build_grammar(n_sources=-1, policy=Policy.AUTO)


# --- Lark parse_ok: grammars are well-formed and admit expected samples -------


def test_required_admits_cited_sentences() -> None:
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED)
    assert parse_ok(g, "Hello world [1].")
    assert parse_ok(g, "One thing [1] [2].")
    assert parse_ok(g, "First [1]. Second [2].")


def test_required_rejects_uncited_sentences() -> None:
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED)
    assert not parse_ok(g, "Hello world.")
    assert not parse_ok(g, "No cite here.")


def test_required_rejects_out_of_range_cite_id() -> None:
    """The whole point of the §10.1 contract: [N+k] is grammar-unreachable."""
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED)
    assert not parse_ok(g, "Hello [4].")
    assert not parse_ok(g, "Hello [99].")
    # Also zero and non-numeric variants.
    assert not parse_ok(g, "Hello [0].")
    assert not parse_ok(g, "Hello [a].")


def test_auto_admits_cited_and_uncited() -> None:
    g = build_grammar(n_sources=3, policy=Policy.AUTO)
    assert parse_ok(g, "Anything goes here.")
    assert parse_ok(g, "With a cite [2] in the middle.")
    assert parse_ok(g, "[1][3] cites alone also fine.")


def test_auto_rejects_out_of_range_cite_id() -> None:
    g = build_grammar(n_sources=3, policy=Policy.AUTO)
    assert not parse_ok(g, "Nope [4].")


def test_quotes_only_accepts_quoted_span_with_cite() -> None:
    g = build_grammar(n_sources=3, policy=Policy.QUOTES_ONLY)
    assert parse_ok(g, '"quoted"[1]')
    assert parse_ok(g, 'Intro "quoted"[1] tail')


def test_quotes_only_rejects_out_of_range_cite_id() -> None:
    g = build_grammar(n_sources=3, policy=Policy.QUOTES_ONLY)
    assert not parse_ok(g, '"quoted"[7]')


# --- Helpers ------------------------------------------------------------------


def _grammar_dict(g: Grammar) -> dict[str, object]:
    """Stable serialization for `data_regression`. Avoids the dataclass default repr."""
    return {
        "ebnf": g.ebnf,
        "cite_ids": list(g.cite_ids),
        "policy": g.policy.value,
    }
