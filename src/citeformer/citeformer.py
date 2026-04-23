"""The `Citeformer` orchestrator — the public entry point for generation.

Composes a `Backend` with a citation policy and a CSL style. Calls the backend to
produce raw text with inline `[N]` markers, parses those markers into `Citation`
objects, renders the reference list, and packages the whole thing as a
`GenerationResult`.

Reference rendering is a stub in P1 and gains real citeproc-py backing in P3.
"""

from __future__ import annotations

import re
from typing import Any

from citeformer.backends.base import Backend
from citeformer.core import Citation, GenerationResult, Policy, Reference, Source

# Pattern for extracting [N] markers from generated text. The N is constrained at
# decode time (P2+) to the valid source id range — this pattern only does the
# post-hoc parsing, not enforcement.
_CITE_PATTERN = re.compile(r"\[(\d+)\]")


class Citeformer:
    """High-level orchestrator for generating citation-backed text.

    Wraps a `Backend` with a citation policy and a CSL style. In v0.1, references
    are rendered deterministically via citeproc-py (P3); until that phase ships,
    references carry a stub `rendered` string so the pipeline is end-to-end
    testable.

    Example:
        >>> from citeformer import Citeformer, Source
        >>> from citeformer.backends import MockBackend
        >>> sources = [Source(metadata={"id": "a", "type": "book"}, content="...")]
        >>> cf = Citeformer(backend=MockBackend())
        >>> result = cf.generate(prompt="hi", sources=sources)
        >>> "[1]" in result.text
        True

    Attributes:
        backend: The backend used to generate raw text.
        style: CSL style identifier (e.g. `"apa-7"`). Consumed in P3.
        citation_policy: Default citation enforcement policy.
    """

    def __init__(
        self,
        backend: Backend,
        style: str = "apa-7",
        citation_policy: Policy = Policy.REQUIRED,
    ) -> None:
        """Construct a Citeformer.

        Args:
            backend: Backend instance to delegate generation to.
            style: CSL style ID. Not yet consumed (P3).
            citation_policy: Default citation enforcement policy for `generate()`.
        """
        self.backend = backend
        self.style = style
        self.citation_policy = citation_policy

    def generate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy | None = None,
        **options: Any,
    ) -> GenerationResult:
        """Generate text with constrained citations.

        Args:
            prompt: User prompt. The orchestrator passes it through to the backend
                verbatim; retrieval-augmented stitching is the caller's job until
                we add a prompt-assembly helper in a later phase.
            sources: Evidence chunks in scope. Position (1-indexed) determines the
                citation id the model emits and the id used in `Citation.source_id`
                and `Reference.source_id`.
            policy: Override the default `citation_policy` for this call.
            **options: Backend-specific options forwarded to `Backend.generate()`.

        Returns:
            A `GenerationResult` with text, parsed citations, and rendered references.
        """
        effective_policy = policy if policy is not None else self.citation_policy
        text = self.backend.generate(
            prompt=prompt,
            sources=sources,
            policy=effective_policy,
            **options,
        )
        citations = self._parse_citations(text)
        references = self._render_references(sources, citations)
        return GenerationResult(
            text=text,
            citations=citations,
            references=references,
        )

    @staticmethod
    def _parse_citations(text: str) -> list[Citation]:
        """Extract `[N]` markers from `text` into `Citation` objects."""
        return [
            Citation(span=(m.start(), m.end()), source_id=int(m.group(1)))
            for m in _CITE_PATTERN.finditer(text)
        ]

    def _render_references(
        self,
        sources: list[Source],
        citations: list[Citation],
    ) -> list[Reference]:
        """Render the reference list for the unique cited sources.

        P1 emits a stub rendering — one `Reference` per unique cited source_id,
        using the source's `metadata["title"]` (or `"Untitled"` fallback). P3
        replaces this with a citeproc-py wrapper that renders the full CSL entry
        in `self.style`.
        """
        cited_ids = sorted({c.source_id for c in citations})
        references: list[Reference] = []
        for cid in cited_ids:
            if not (1 <= cid <= len(sources)):
                # Shouldn't happen under grammar-level enforcement, but the mock
                # backend and hand-constructed tests could produce out-of-range
                # ids. Silently skip rather than crash — verify() will surface it.
                continue
            source = sources[cid - 1]
            title = source.metadata.get("title", "Untitled")
            references.append(
                Reference(
                    source_id=cid,
                    inline_marker=f"[{cid}]",
                    rendered=(
                        f"[{cid}] {title} (stub rendering — citeproc-py CSL rendering lands in P3)"
                    ),
                )
            )
        return references
