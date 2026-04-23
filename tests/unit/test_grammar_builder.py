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
from citeformer.grammar.builder import DEFAULT_MAX_CONTENT_CHARS

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


def test_grammar_required_unbounded_snapshot(data_regression) -> None:  # type: ignore[no-untyped-def]
    """ADR-009 escape hatch: `max_content_chars=None` keeps the legacy `+` body."""
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED, max_content_chars=None)
    data_regression.check(_grammar_dict(g))


def test_grammar_required_custom_bound_snapshot(data_regression) -> None:  # type: ignore[no-untyped-def]
    """Small explicit bound — used in tests to exercise the progression quickly."""
    g = build_grammar(n_sources=2, policy=Policy.REQUIRED, max_content_chars=16)
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


def test_build_grammar_rejects_zero_max_content_chars() -> None:
    with pytest.raises(ValueError, match="max_content_chars must be >= 1 or None"):
        build_grammar(n_sources=3, policy=Policy.REQUIRED, max_content_chars=0)


def test_build_grammar_rejects_negative_max_content_chars() -> None:
    with pytest.raises(ValueError, match="max_content_chars must be >= 1 or None"):
        build_grammar(n_sources=3, policy=Policy.REQUIRED, max_content_chars=-5)


# --- ADR-009 bounded content (REQUIRED progression fix) -----------------------


def test_required_default_content_bound_is_applied() -> None:
    """ADR-009: default REQUIRED body includes `{1, DEFAULT_MAX_CONTENT_CHARS}`."""
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED)
    assert f"content ::= [^\\[.!?]{{1, {DEFAULT_MAX_CONTENT_CHARS}}}" in g.gbnf
    assert g.max_content_chars == DEFAULT_MAX_CONTENT_CHARS


def test_required_unbounded_preserves_legacy_plus_quantifier() -> None:
    g = build_grammar(n_sources=3, policy=Policy.REQUIRED, max_content_chars=None)
    assert "content ::= [^\\[.!?]+" in g.gbnf
    assert g.max_content_chars is None


def test_auto_and_quotes_only_ignore_max_content_chars() -> None:
    """Bound only applies to REQUIRED — surfaced via `max_content_chars=None` metadata."""
    g_auto = build_grammar(n_sources=3, policy=Policy.AUTO, max_content_chars=50)
    g_q = build_grammar(n_sources=3, policy=Policy.QUOTES_ONLY, max_content_chars=50)
    assert g_auto.max_content_chars is None
    assert g_q.max_content_chars is None
    # Neither body should have gained a bounded repetition.
    assert "{1, 50}" not in g_auto.gbnf
    assert "{1, 50}" not in g_q.gbnf


# --- Helpers ------------------------------------------------------------------


def _grammar_dict(g: Grammar) -> dict[str, object]:
    """Stable serialization for `data_regression`. Avoids the dataclass default repr."""
    return {
        "gbnf": g.gbnf,
        "cite_ids": list(g.cite_ids),
        "policy": g.policy.value,
        "root_rule": g.root_rule,
    }
