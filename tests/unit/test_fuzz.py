"""Property-based fuzz tests using `hypothesis`.

The goal is coverage of shapes we don't hand-enumerate:

- Grammar builder across any valid (n_sources, policy, max_content_chars) triple
- Every formatter against any well-typed CSL-JSON item (all canonical types,
  random author lists, missing or extra fields)
- `build_rag_prompt` against random source sets + queries
- Parsing cycle (emit `[N]` markers up to N, regex should round-trip)

If any of these fail, we've got a real invariant violation. Hypothesis shrinks
failing inputs to the minimal reproducer; output goes to
`tests/unit/.hypothesis/examples/` (gitignored by default).
"""

from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from citeformer import Policy, Source, build_rag_prompt
from citeformer.grammar import build_grammar
from citeformer.render import render_single_reference
from citeformer.render.formatters import available_formatters

_CITE = re.compile(r"\[(\d+)\]")

# Keep examples small; correctness is about shape, not throughput.
_FUZZ_SETTINGS = settings(
    max_examples=50,
    deadline=None,  # some tests load xgrammar which can be slow on first call
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)


# --- Strategies ---------------------------------------------------------------


@st.composite
def csl_name(draw: st.DrawFn) -> dict:
    """A CSL-JSON author record. Either `family/given` or `literal`."""
    if draw(st.booleans()):
        family = draw(st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=1, max_size=15))
        entry: dict = {"family": family}
        if draw(st.booleans()):
            entry["given"] = draw(
                st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=1, max_size=15)
            )
        return entry
    return {
        "literal": draw(
            st.text(alphabet=st.characters(whitelist_categories=("L", "N", "Zs")), min_size=1, max_size=30)
        )
    }


@st.composite
def csl_date(draw: st.DrawFn) -> dict:
    """A CSL-JSON `issued`-style date. Sometimes just year; sometimes y/m/d."""
    year = draw(st.integers(min_value=1500, max_value=2030))
    parts: list[list[int]] = [[year]]
    if draw(st.booleans()):
        parts[0].append(draw(st.integers(min_value=1, max_value=12)))
        if draw(st.booleans()):
            parts[0].append(draw(st.integers(min_value=1, max_value=28)))
    return {"date-parts": parts}


@st.composite
def csl_item(draw: st.DrawFn) -> dict:
    """Generate a CSL-JSON item across the canonical types we format."""
    item_type = draw(
        st.sampled_from(
            ["book", "article-journal", "chapter", "thesis", "webpage", "paper-conference", "report"]
        )
    )
    ident = draw(st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20))
    title = draw(st.text(min_size=1, max_size=80).filter(lambda s: s.strip()))
    item: dict = {"id": ident, "type": item_type, "title": title}
    if draw(st.booleans()):
        item["author"] = draw(st.lists(csl_name(), min_size=0, max_size=12))
    if draw(st.booleans()):
        item["issued"] = draw(csl_date())
    # Optional structural fields — exercise formatter branches.
    if draw(st.booleans()):
        item["container-title"] = draw(st.text(min_size=1, max_size=60).filter(lambda s: s.strip()))
    if draw(st.booleans()):
        item["publisher"] = draw(st.text(min_size=1, max_size=40).filter(lambda s: s.strip()))
    if draw(st.booleans()):
        item["publisher-place"] = draw(st.text(min_size=1, max_size=30).filter(lambda s: s.strip()))
    if draw(st.booleans()):
        item["volume"] = str(draw(st.integers(min_value=1, max_value=999)))
    if draw(st.booleans()):
        item["issue"] = str(draw(st.integers(min_value=1, max_value=99)))
    if draw(st.booleans()):
        start = draw(st.integers(min_value=1, max_value=900))
        end = start + draw(st.integers(min_value=0, max_value=100))
        item["page"] = f"{start}-{end}"
    if draw(st.booleans()):
        item["DOI"] = f"10.{draw(st.integers(min_value=1000, max_value=9999))}/fuzz.{ident}"
    return item


@st.composite
def source_strategy(draw: st.DrawFn) -> Source:
    return Source(
        metadata=draw(csl_item()),
        content=draw(st.text(max_size=200)),
    )


_POLICY_ST = st.sampled_from(list(Policy))
_N_SOURCES_ST = st.integers(min_value=1, max_value=30)
_MAX_CONTENT_ST = st.one_of(
    st.none(),
    st.integers(min_value=1, max_value=500),
)


# --- Grammar builder ----------------------------------------------------------


@_FUZZ_SETTINGS
@given(n=_N_SOURCES_ST, policy=_POLICY_ST, mc=_MAX_CONTENT_ST)
def test_grammar_always_compiles_with_xgrammar(n: int, policy: Policy, mc: int | None) -> None:
    """Any valid (n_sources, policy, max_content_chars) → xgrammar accepts the GBNF."""
    import xgrammar as xgr

    g = build_grammar(n_sources=n, policy=policy, max_content_chars=mc)
    # Semantic validity — raises if the emitted body has a shape xgrammar rejects.
    parsed = xgr.Grammar.from_ebnf(g.gbnf)
    assert str(parsed)  # rendered form is non-empty


@_FUZZ_SETTINGS
@given(n=_N_SOURCES_ST, policy=_POLICY_ST, mc=_MAX_CONTENT_ST)
def test_grammar_cite_ids_match_n_sources(n: int, policy: Policy, mc: int | None) -> None:
    g = build_grammar(n_sources=n, policy=policy, max_content_chars=mc)
    assert g.cite_ids == tuple(range(1, n + 1))
    # cite-id rule must literally enumerate 1..n — no gaps, no extras.
    assert all(f'"{i}"' in g.gbnf for i in range(1, n + 1))
    assert f'"{n + 1}"' not in g.gbnf  # N+1 is NOT in the enum


# --- Formatters ---------------------------------------------------------------


@_FUZZ_SETTINGS
@given(item=csl_item(), style=st.sampled_from(available_formatters()))
def test_every_formatter_never_crashes_on_well_typed_csl(item: dict, style: str) -> None:
    """Formatters must handle any well-typed CSL-JSON item without raising."""
    source = Source(metadata=item, content="")
    ref = render_single_reference(source, style_name=style, number=1)
    assert isinstance(ref.rendered, str)
    assert isinstance(ref.inline_marker, str)


@_FUZZ_SETTINGS
@given(item=csl_item(), style=st.sampled_from(available_formatters()))
def test_formatter_outputs_never_contain_double_period(item: dict, style: str) -> None:
    """No bibliography entry should emit `..` or `et al..` (regression lock)."""
    source = Source(metadata=item, content="")
    ref = render_single_reference(source, style_name=style, number=1)
    assert ".." not in ref.rendered, (
        f"{style} emitted double-period: {ref.rendered!r} for item {item!r}"
    )


@_FUZZ_SETTINGS
@given(item=csl_item(), n=st.integers(min_value=1, max_value=20))
def test_inline_marker_for_numeric_styles_matches_number(item: dict, n: int) -> None:
    """Numeric styles must emit ``[N]`` / ``N`` with the argument's number verbatim."""
    source = Source(metadata=item, content="")
    for style in ("ieee", "vancouver"):
        ref = render_single_reference(source, style_name=style, number=n)
        assert ref.inline_marker == f"[{n}]"
    ref_nature = render_single_reference(source, style_name="nature", number=n)
    assert ref_nature.inline_marker == str(n)


# --- Prompt helper ------------------------------------------------------------


@_FUZZ_SETTINGS
@given(
    sources=st.lists(source_strategy(), min_size=1, max_size=10),
    query=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
)
def test_build_rag_prompt_always_numbers_sources_consistently(
    sources: list[Source], query: str
) -> None:
    """Every source gets a ``[i]`` marker in order; no extras."""
    out = build_rag_prompt(query=query, sources=sources)
    for i in range(1, len(sources) + 1):
        assert f"[{i}]" in out, f"missing [{i}] in prompt with {len(sources)} sources"
    # No marker for N+1.
    assert f"[{len(sources) + 1}]" not in out


@_FUZZ_SETTINGS
@given(
    sources=st.lists(source_strategy(), min_size=1, max_size=10),
    query=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
)
def test_build_rag_prompt_query_appears_verbatim(sources: list[Source], query: str) -> None:
    out = build_rag_prompt(query=query, sources=sources)
    assert query.strip() in out


# --- Marker parsing round-trip ------------------------------------------------


@_FUZZ_SETTINGS
@given(
    cite_ids=st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=30),
)
def test_cite_pattern_roundtrips_ids_in_order(cite_ids: list[int]) -> None:
    """If we render ``[N]`` markers and parse back, we must recover the same ids."""
    text = " ".join(f"Sentence [{i}]." for i in cite_ids)
    parsed = [int(m.group(1)) for m in _CITE.finditer(text)]
    assert parsed == cite_ids


# --- Source construction ------------------------------------------------------


@_FUZZ_SETTINGS
@given(item=csl_item(), content=st.text(max_size=500))
def test_source_is_frozen_and_roundtrips(item: dict, content: str) -> None:
    from pydantic import ValidationError

    source = Source(metadata=item, content=content)
    # Frozen — attribute assignment should raise a pydantic ValidationError.
    with pytest.raises(ValidationError):
        source.metadata = {"tampered": True}  # type: ignore[misc]
    # model_dump round-trip preserves shape.
    dumped = source.model_dump()
    assert dumped["metadata"] == item
    assert dumped["content"] == content
