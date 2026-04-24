"""Quickstart demo using MockBackend — no HF, no torch, no model download.

Exercises the full orchestration shape so you can see how a `Source`,
`Citation`, `Reference`, and `GenerationResult` fit together without any
ML noise. The `MockBackend` emits a canned response that respects the
grammar contract (every `[N]` is in range); swap it for an `HFBackend` in
production.

Run:

    uv run python examples/01_quickstart_mock.py

Installs needed: core only (`pip install citeformer` or `uv sync`).
"""

from __future__ import annotations

from citeformer import Citeformer, Policy, Source
from citeformer.backends import MockBackend


def main() -> None:
    sources = [
        Source(
            metadata={
                "id": "poe-raven",
                "type": "book",
                "title": "The Raven",
                "author": [{"family": "Poe", "given": "Edgar Allan"}],
                "issued": {"date-parts": [[1845]]},
            },
            content="Once upon a midnight dreary, while I pondered, weak and weary...",
        ),
        Source(
            metadata={
                "id": "melville-moby-dick",
                "type": "book",
                "title": "Moby-Dick",
                "author": [{"family": "Melville", "given": "Herman"}],
                "issued": {"date-parts": [[1851]]},
            },
            content="Call me Ishmael. Some years ago—never mind how long precisely...",
        ),
        Source(
            metadata={
                "id": "austen-pride",
                "type": "book",
                "title": "Pride and Prejudice",
                "author": [{"family": "Austen", "given": "Jane"}],
                "issued": {"date-parts": [[1813]]},
            },
            content="It is a truth universally acknowledged that a single man in possession...",
        ),
    ]

    cf = Citeformer(
        backend=MockBackend(),
        style="apa-7",
        citation_policy=Policy.AUTO,  # MockBackend doesn't honor grammar shape
    )
    result = cf.generate(
        prompt="Summarize the three works.",
        sources=sources,
    )

    print("=" * 60)
    print("Generated text")
    print("=" * 60)
    print(result.text)
    print()

    print("=" * 60)
    print(f"Parsed {len(result.citations)} citation(s)")
    print("=" * 60)
    for cite in result.citations:
        start, end = cite.span
        print(f"  source_id={cite.source_id}  span=[{start}:{end}]  ({result.text[start:end]!r})")
    print()

    print("=" * 60)
    print(f"Rendered {len(result.references)} reference(s) in {cf.style}")
    print("=" * 60)
    for ref in result.references:
        print(f"  [{ref.source_id}] {ref.inline_marker} — {ref.rendered}")
    print()

    # schema_version is part of the §10.3 contract; a real caller would
    # pin-compare this before deserializing cross-version.
    print(f"schema_version = {result.schema_version}  (§10.3 contract)")


if __name__ == "__main__":
    main()
