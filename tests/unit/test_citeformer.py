"""Unit tests for the `Citeformer` orchestrator, exercised with `MockBackend`."""

from __future__ import annotations

from citeformer import Citeformer, MockBackend, Policy, Source


def _sources(n: int = 3) -> list[Source]:
    return [
        Source(
            metadata={"id": f"s{i}", "type": "book", "title": f"Title {i}"},
            content=f"Content {i}",
        )
        for i in range(1, n + 1)
    ]


def test_citeformer_defaults() -> None:
    cf = Citeformer(backend=MockBackend())
    assert cf.style == "apa-7"
    assert cf.citation_policy is Policy.REQUIRED


def test_citeformer_parses_cite_markers() -> None:
    backend = MockBackend(responses={"q": "Text A [1] and text B [3]."})
    cf = Citeformer(backend=backend)
    result = cf.generate(prompt="q", sources=_sources())
    assert len(result.citations) == 2
    ids = [c.source_id for c in result.citations]
    assert ids == [1, 3]
    # Verify char spans actually point at marker substrings.
    for c in result.citations:
        start, end = c.span
        assert result.text[start:end] == f"[{c.source_id}]"


def test_citeformer_renders_stub_references_for_unique_cites() -> None:
    backend = MockBackend(responses={"q": "A [1] B [1] C [3]."})
    cf = Citeformer(backend=backend)
    result = cf.generate(prompt="q", sources=_sources())
    # Two unique cites (1 and 3), even though [1] appears twice.
    assert {ref.source_id for ref in result.references} == {1, 3}
    # Rendering is a stub until P3 — just check it resolves to the source title.
    ref1 = next(r for r in result.references if r.source_id == 1)
    assert "Title 1" in ref1.rendered


def test_citeformer_skips_out_of_range_cite_in_references() -> None:
    """Grammar-level enforcement prevents this in P2+; P1 skip is belt-and-suspenders."""
    backend = MockBackend(responses={"q": "Fabricated [99]."})
    cf = Citeformer(backend=backend)
    result = cf.generate(prompt="q", sources=_sources(3))
    # Citation still parsed (post-hoc, no grammar), but no Reference for the bogus id.
    assert [c.source_id for c in result.citations] == [99]
    assert result.references == []


def test_citeformer_policy_override() -> None:
    recorded: dict[str, object] = {}

    class RecordingBackend(MockBackend):
        def generate(self, prompt, sources, policy, **options):  # type: ignore[no-untyped-def,override]
            recorded["policy"] = policy
            return super().generate(prompt, sources, policy, **options)

    cf = Citeformer(backend=RecordingBackend(), citation_policy=Policy.REQUIRED)
    cf.generate(prompt="q", sources=_sources(1), policy=Policy.AUTO)
    assert recorded["policy"] is Policy.AUTO


def test_citeformer_respects_default_policy() -> None:
    recorded: dict[str, object] = {}

    class RecordingBackend(MockBackend):
        def generate(self, prompt, sources, policy, **options):  # type: ignore[no-untyped-def,override]
            recorded["policy"] = policy
            return super().generate(prompt, sources, policy, **options)

    cf = Citeformer(backend=RecordingBackend(), citation_policy=Policy.QUOTES_ONLY)
    cf.generate(prompt="q", sources=_sources(1))
    assert recorded["policy"] is Policy.QUOTES_ONLY
