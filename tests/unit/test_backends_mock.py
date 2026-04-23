"""Unit tests for the MockBackend.

Mock is the only backend available until P2 — these tests exercise the Backend ABC
contract, not any real decoding logic.
"""

from __future__ import annotations

from citeformer import MockBackend, Policy, Source


def _sources(n: int = 3) -> list[Source]:
    return [
        Source(
            metadata={"id": f"s{i}", "type": "book", "title": f"Title {i}"},
            content=f"Content chunk {i}",
        )
        for i in range(1, n + 1)
    ]


def test_mock_backend_fallback_emits_cite_marker_when_sources_present() -> None:
    backend = MockBackend()
    text = backend.generate(prompt="hello", sources=_sources(), policy=Policy.REQUIRED)
    assert "[1]" in text
    assert "hello" in text


def test_mock_backend_fallback_emits_no_marker_when_no_sources() -> None:
    backend = MockBackend()
    text = backend.generate(prompt="hi", sources=[], policy=Policy.AUTO)
    assert "[" not in text
    assert "hi" in text


def test_mock_backend_honors_scripted_response() -> None:
    backend = MockBackend(responses={"ping": "pong [2]."})
    text = backend.generate(prompt="ping", sources=_sources(), policy=Policy.REQUIRED)
    assert text == "pong [2]."


def test_mock_backend_scripted_response_overrides_fallback() -> None:
    backend = MockBackend(responses={"ping": "pong."})
    text = backend.generate(prompt="pong", sources=_sources(), policy=Policy.REQUIRED)
    # "pong" isn't in the responses map, so we fall back — distinct from "ping" above.
    assert text != "pong."
    assert "pong" in text  # echoed in the fallback


def test_mock_backend_ignores_options_kwargs() -> None:
    backend = MockBackend()
    # Shouldn't raise on arbitrary options.
    text = backend.generate(
        prompt="x",
        sources=_sources(1),
        policy=Policy.AUTO,
        max_tokens=50,
        temperature=0.7,
        unknown_knob="banana",
    )
    assert "x" in text
