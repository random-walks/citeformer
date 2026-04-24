"""Property-based fuzz tests using `hypothesis`.

The goal is coverage of shapes we don't hand-enumerate:

- Grammar builder across any valid (n_sources, policy, max_content_chars) triple
- Every formatter against any well-typed CSL-JSON item (all canonical types,
  random author lists, missing or extra fields)
- `build_rag_prompt` against random source sets + queries
- Parsing cycle (emit `[N]` markers up to N, regex should round-trip)
- CSL-JSON validator boundary conditions
- `deduplicate_adjacent_cites` algebraic properties
- `GenerationResult` / `Citation` / `Source` pydantic invariants

If any of these fail, we've got a real invariant violation. Hypothesis shrinks
failing inputs to the minimal reproducer; output goes to
`tests/unit/.hypothesis/examples/` (gitignored by default).
"""

from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from citeformer import (
    Citation,
    GenerationResult,
    Policy,
    Reference,
    Source,
    build_rag_prompt,
    deduplicate_adjacent_cites,
    validate_csl_json,
)
from citeformer.grammar import build_grammar
from citeformer.render import render_references, render_single_reference
from citeformer.render.formatters import available_formatters
from citeformer.render.styles import style_citation_format

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
        family = draw(
            st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=1, max_size=15)
        )
        entry: dict = {"family": family}
        if draw(st.booleans()):
            entry["given"] = draw(
                st.text(
                    alphabet=st.characters(whitelist_categories=("L",)), min_size=1, max_size=15
                )
            )
        return entry
    return {
        "literal": draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
                min_size=1,
                max_size=30,
            )
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
            [
                "book",
                "article-journal",
                "chapter",
                "thesis",
                "webpage",
                "paper-conference",
                "report",
            ]
        )
    )
    ident = draw(
        st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20)
    )
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
_STYLE_ST = st.sampled_from(available_formatters())


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


@_FUZZ_SETTINGS
@given(n=_N_SOURCES_ST, policy=_POLICY_ST, mc=_MAX_CONTENT_ST)
def test_grammar_has_cite_group_and_cite_id_rules(n: int, policy: Policy, mc: int | None) -> None:
    """Every emitted grammar must reference a cite-group / cite-id terminal."""
    g = build_grammar(n_sources=n, policy=policy, max_content_chars=mc)
    assert "cite-group" in g.gbnf
    assert "cite-id" in g.gbnf
    assert g.root_rule == "root"
    assert g.policy is policy


@_FUZZ_SETTINGS
@given(n=_N_SOURCES_ST, mc=_MAX_CONTENT_ST)
def test_grammar_required_bound_reflected_in_result(n: int, mc: int | None) -> None:
    """`max_content_chars` round-trips on REQUIRED; AUTO/QUOTES_ONLY always `None`."""
    g_req = build_grammar(n_sources=n, policy=Policy.REQUIRED, max_content_chars=mc)
    assert g_req.max_content_chars == mc
    g_auto = build_grammar(n_sources=n, policy=Policy.AUTO, max_content_chars=mc)
    assert g_auto.max_content_chars is None


@_FUZZ_SETTINGS
@given(n=st.integers(min_value=-5, max_value=0))
def test_grammar_rejects_nonpositive_n_sources(n: int) -> None:
    with pytest.raises(ValueError, match="n_sources"):
        build_grammar(n_sources=n, policy=Policy.AUTO)


@_FUZZ_SETTINGS
@given(mc=st.integers(max_value=0))
def test_grammar_rejects_nonpositive_max_content_chars(mc: int) -> None:
    with pytest.raises(ValueError, match="max_content_chars"):
        build_grammar(n_sources=3, policy=Policy.REQUIRED, max_content_chars=mc)


# --- Formatters ---------------------------------------------------------------


@_FUZZ_SETTINGS
@given(item=csl_item(), style=_STYLE_ST)
def test_every_formatter_never_crashes_on_well_typed_csl(item: dict, style: str) -> None:
    """Formatters must handle any well-typed CSL-JSON item without raising."""
    source = Source(metadata=item, content="")
    ref = render_single_reference(source, style_name=style, number=1)
    assert isinstance(ref.rendered, str)
    assert isinstance(ref.inline_marker, str)


@_FUZZ_SETTINGS
@given(item=csl_item(), style=_STYLE_ST)
def test_formatter_outputs_never_contain_double_period(item: dict, style: str) -> None:
    """No bibliography entry should emit `..` or `et al..` (regression lock)."""
    source = Source(metadata=item, content="")
    ref = render_single_reference(source, style_name=style, number=1)
    assert ".." not in ref.rendered, (
        f"{style} emitted double-period: {ref.rendered!r} for item {item!r}"
    )


@_FUZZ_SETTINGS
@given(item=csl_item(), style=_STYLE_ST)
def test_formatter_outputs_never_have_trailing_or_leading_whitespace(
    item: dict, style: str
) -> None:
    """Rendered bibliography entries should be left/right trimmed."""
    source = Source(metadata=item, content="")
    ref = render_single_reference(source, style_name=style, number=1)
    assert ref.rendered == ref.rendered.strip(), f"{style} leaked whitespace: {ref.rendered!r}"


@_FUZZ_SETTINGS
@given(item=csl_item(), style=_STYLE_ST)
def test_formatter_outputs_never_contain_double_spaces(item: dict, style: str) -> None:
    """Bibliography entries should not contain `  ` (double-space regression)."""
    source = Source(metadata=item, content="")
    ref = render_single_reference(source, style_name=style, number=1)
    assert "  " not in ref.rendered, f"{style} emitted double-space: {ref.rendered!r}"


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


@_FUZZ_SETTINGS
@given(item=csl_item(), style=_STYLE_ST)
def test_formatter_output_is_nonempty(item: dict, style: str) -> None:
    """Every render — even on minimal input — must produce at least one character."""
    source = Source(metadata=item, content="")
    ref = render_single_reference(source, style_name=style, number=1)
    assert ref.rendered.strip(), f"{style} produced empty output for item {item!r}"


@_FUZZ_SETTINGS
@given(
    item=csl_item().filter(lambda i: "issued" in i),
    style=st.sampled_from(
        [s for s in available_formatters() if style_citation_format(s) == "author-date"]
    ),
)
def test_author_date_styles_contain_year_when_issued_present(item: dict, style: str) -> None:
    """APA-7 / Chicago author-date must render the year when issued is set."""
    source = Source(metadata=item, content="")
    ref = render_single_reference(source, style_name=style, number=1)
    year = str(item["issued"]["date-parts"][0][0])
    assert year in ref.rendered, f"{style} dropped year {year} for item {item!r}: {ref.rendered!r}"


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


@_FUZZ_SETTINGS
@given(
    sources=st.lists(source_strategy(), min_size=2, max_size=8),
    query=st.text(min_size=1, max_size=80).filter(lambda s: s.strip()),
)
def test_build_rag_prompt_orders_sources_by_position(sources: list[Source], query: str) -> None:
    """The N source markers appear in ascending order inside the prompt."""
    out = build_rag_prompt(query=query, sources=sources)
    positions = [out.find(f"[{i}]") for i in range(1, len(sources) + 1)]
    assert all(p > 0 for p in positions), f"missing marker, positions={positions}"
    assert positions == sorted(positions), f"source markers out of order: {positions}"


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


# --- Source / GenerationResult invariants -------------------------------------


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


@_FUZZ_SETTINGS
@given(
    text=st.text(max_size=200),
    cite_ids=st.lists(st.integers(min_value=1, max_value=5), min_size=0, max_size=10),
)
def test_generation_result_is_frozen(text: str, cite_ids: list[int]) -> None:
    """GenerationResult fields must not be mutable after construction."""
    from pydantic import ValidationError

    citations = [Citation(span=(i, i + 3), source_id=src) for i, src in enumerate(cite_ids)]
    result = GenerationResult(text=text, citations=citations, references=[], sources=[])
    with pytest.raises(ValidationError):
        result.text = "mutated"  # type: ignore[misc]


@_FUZZ_SETTINGS
@given(text=st.text(max_size=50))
def test_generation_result_default_schema_version_is_2(text: str) -> None:
    """Default schema_version must stay 2 until an explicit bump."""
    result = GenerationResult(text=text)
    assert result.schema_version == 2


@_FUZZ_SETTINGS
@given(source_id=st.integers(min_value=1, max_value=100))
def test_citation_requires_positive_source_id(source_id: int) -> None:
    """source_id is constrained to >= 1."""
    c = Citation(span=(0, 3), source_id=source_id)
    assert c.source_id == source_id


@_FUZZ_SETTINGS
@given(source_id=st.integers(max_value=0))
def test_citation_rejects_nonpositive_source_id(source_id: int) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Citation(span=(0, 3), source_id=source_id)


# --- render_references invariants --------------------------------------------


@_FUZZ_SETTINGS
@given(
    items=st.lists(csl_item(), min_size=1, max_size=8),
    cite_ids=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=15),
    style=_STYLE_ST,
)
def test_render_references_emits_one_reference_per_unique_cited_id(
    items: list[dict], cite_ids: list[int], style: str
) -> None:
    """The reference list's ids must match the unique cited ids exactly."""
    sources = [Source(metadata=item, content="") for item in items]
    citations = [
        Citation(span=(i, i + 3), source_id=sid)
        for i, sid in enumerate(cite_ids)
        if 1 <= sid <= len(sources)
    ]
    refs = render_references(sources, citations, style)
    cited_ids = {c.source_id for c in citations}
    assert {r.source_id for r in refs} == cited_ids


@_FUZZ_SETTINGS
@given(
    items=st.lists(csl_item(), min_size=2, max_size=6),
    cite_ids=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=12),
    style=_STYLE_ST,
)
def test_render_references_are_in_ascending_source_id_order(
    items: list[dict], cite_ids: list[int], style: str
) -> None:
    """References are returned in ascending source-id order (numeric bibliography)."""
    sources = [Source(metadata=item, content="") for item in items]
    citations = [
        Citation(span=(i, i + 3), source_id=sid)
        for i, sid in enumerate(cite_ids)
        if 1 <= sid <= len(sources)
    ]
    refs = render_references(sources, citations, style)
    ids = [r.source_id for r in refs]
    assert ids == sorted(ids)
    # Out-of-range citations are dropped; in-range unique ids are retained.
    in_range_unique = sorted({c.source_id for c in citations if 1 <= c.source_id <= len(sources)})
    assert ids == in_range_unique


# --- CSL validator ------------------------------------------------------------


@_FUZZ_SETTINGS
@given(item=csl_item())
def test_validate_csl_json_accepts_well_typed_items(item: dict) -> None:
    """Generated CSL items are well-typed — validator must not emit errors."""
    report = validate_csl_json(item)
    assert not report.errors, f"unexpected errors {report.errors} for {item!r}"


@_FUZZ_SETTINGS
@given(item=csl_item())
def test_validate_csl_json_missing_id_is_error(item: dict) -> None:
    """Stripping `id` → error."""
    stripped = {k: v for k, v in item.items() if k != "id"}
    report = validate_csl_json(stripped)
    assert any("id" in e for e in report.errors)


@_FUZZ_SETTINGS
@given(item=csl_item())
def test_validate_csl_json_missing_type_is_error(item: dict) -> None:
    """Stripping `type` → error."""
    stripped = {k: v for k, v in item.items() if k != "type"}
    report = validate_csl_json(stripped)
    assert any("type" in e for e in report.errors)


@_FUZZ_SETTINGS
@given(
    item=csl_item(),
    bogus_key=st.text(
        alphabet=st.characters(whitelist_categories=("L",)), min_size=3, max_size=15
    ).filter(lambda s: s not in {"id", "type", "title", "author", "issued"}),
)
def test_validate_csl_json_unknown_field_is_warning_not_error(item: dict, bogus_key: str) -> None:
    """Unknown top-level fields are warnings (not errors) by design."""
    extended = {**item, bogus_key: "anything"}
    report = validate_csl_json(extended)
    # The bogus key produces a warning but never an error (unless the bogus
    # name happens to be a known field, which is filtered above). Errors
    # arise only if the generated item is otherwise malformed — which our
    # strategy doesn't do.
    assert not report.errors, f"unexpected errors for extended item {extended!r}: {report.errors}"


@_FUZZ_SETTINGS
@given(item=csl_item())
def test_validate_csl_json_bogus_author_type_is_error(item: dict) -> None:
    """`author` must be a list of name dicts — scalar strings must error."""
    bad = {**item, "author": "Just a string not a list"}
    report = validate_csl_json(bad)
    assert any("author" in e for e in report.errors)


# --- deduplicate_adjacent_cites algebraic properties -------------------------


@_FUZZ_SETTINGS
@given(
    text=st.text(max_size=200),
    cite_ids=st.lists(st.integers(min_value=1, max_value=9), min_size=0, max_size=15),
)
def test_dedupe_is_idempotent(text: str, cite_ids: list[int]) -> None:
    """Applying dedupe twice is the same as applying once."""
    body = text + " " + " ".join(f"[{i}]" for i in cite_ids)
    once = deduplicate_adjacent_cites(body)
    twice = deduplicate_adjacent_cites(once)
    assert once == twice


@_FUZZ_SETTINGS
@given(cite_id=st.integers(min_value=1, max_value=9))
def test_dedupe_single_marker_unchanged(cite_id: int) -> None:
    """A lone `[N]` with no neighbours is untouched."""
    body = f"Prose before [{cite_id}] and prose after."
    assert deduplicate_adjacent_cites(body) == body


@_FUZZ_SETTINGS
@given(cite_ids=st.lists(st.integers(min_value=1, max_value=9), min_size=3, max_size=8))
def test_dedupe_collapses_run_to_unique_ids(cite_ids: list[int]) -> None:
    """`[1] [2] [1] [2]` → `[1] [2]` (unique ids preserved in first-appearance order)."""
    run = " ".join(f"[{i}]" for i in cite_ids)
    result = deduplicate_adjacent_cites(run)
    # Unique ids in first-appearance order
    seen: list[int] = []
    for i in cite_ids:
        if i not in seen:
            seen.append(i)
    expected = " ".join(f"[{i}]" for i in seen)
    assert result == expected


# --- Reference model invariants ----------------------------------------------


@_FUZZ_SETTINGS
@given(source_id=st.integers(min_value=1, max_value=20), rendered=st.text(max_size=200))
def test_reference_model_is_frozen(source_id: int, rendered: str) -> None:
    from pydantic import ValidationError

    ref = Reference(source_id=source_id, inline_marker=f"[{source_id}]", rendered=rendered)
    with pytest.raises(ValidationError):
        ref.rendered = "mutated"  # type: ignore[misc]
