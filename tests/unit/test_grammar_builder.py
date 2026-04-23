"""Tests for `citeformer.grammar.builder` — §10.1 contract snapshot + shape.

Grammar outputs are pinned via `pytest-regressions` so any change to the
`cite-id` rule shape or the per-policy body is surfaced as a snapshot diff —
which forces the §10.1 ceremony in `docs/reference/contracts.md`.

Semantic validity (does the grammar actually admit what we think it admits?)
is exercised in `tests/integration/test_hf_backend.py`, which compiles the
emitted string with xgrammar — the authoritative parser for our target
backends. Bundling a separate Lark validator here was considered and rejected:
it'd mean maintaining two grammar formats in lock-step, and the integration
tests already cover the semantic checks against real models.
"""

from __future__ import annotations

import pytest

from citeformer import Policy
from citeformer.grammar import Grammar, build_grammar

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
    """Dynamic cite-id enum — §10.1 requires `"1" | ... | "N"` per call."""
    g = build_grammar(n_sources=10, policy=Policy.REQUIRED)
    data_regression.check(_grammar_dict(g))


# --- cite-id rule shape (the load-bearing §10.1 invariant) --------------------


def test_cite_id_rule_is_bracket_wrapped_enum() -> None:
    """§10.1: `cite-id ::= "[" ("1" | "2" | ... | "N") "]"` — verbatim."""
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED)
    assert 'cite-id ::= "[" ("1" | "2" | "3") "]"' in g.gbnf


def test_cite_id_rule_scales_to_large_n() -> None:
    g = build_grammar(n_sources=15, policy=Policy.AUTO)
    assert (
        'cite-id ::= "[" ("1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | '
        '"9" | "10" | "11" | "12" | "13" | "14" | "15") "]"'
    ) in g.gbnf


def test_cite_ids_metadata_matches_n_sources() -> None:
    g = build_grammar(n_sources=7, policy=Policy.AUTO)
    assert g.cite_ids == (1, 2, 3, 4, 5, 6, 7)


def test_grammar_policy_roundtrip() -> None:
    g = build_grammar(n_sources=3, policy=Policy.QUOTES_ONLY)
    assert g.policy is Policy.QUOTES_ONLY


def test_grammar_default_root_rule_is_root() -> None:
    """GBNF convention + xgrammar default. Any change should be deliberate."""
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED)
    assert g.root_rule == "root"
    assert g.gbnf.lstrip().startswith("root ::=")


# --- Input validation ---------------------------------------------------------


def test_build_grammar_rejects_zero_sources() -> None:
    with pytest.raises(ValueError, match="n_sources must be >= 1"):
        build_grammar(n_sources=0, policy=Policy.REQUIRED)


def test_build_grammar_rejects_negative_sources() -> None:
    with pytest.raises(ValueError, match="n_sources must be >= 1"):
        build_grammar(n_sources=-1, policy=Policy.AUTO)


# --- Helpers ------------------------------------------------------------------


def _grammar_dict(g: Grammar) -> dict[str, object]:
    """Stable serialization for `data_regression`. Avoids the dataclass default repr."""
    return {
        "gbnf": g.gbnf,
        "cite_ids": list(g.cite_ids),
        "policy": g.policy.value,
        "root_rule": g.root_rule,
    }
