"""The `Citeformer` orchestrator — the public entry point for generation.

Composes a `Backend` with a citation policy and a CSL style. Calls the backend to
produce raw text with inline `[N]` markers, parses those markers into `Citation`
objects, renders the reference list, and packages the whole thing as a
`GenerationResult`.

Reference rendering uses the home-grown formatters in
``citeformer.render.formatters`` — see ADR-004 for the rationale.

Streaming:

- `Citeformer.stream()` returns a `StreamingResult` — iterate to consume chunks
  as they're decoded, then call `.finalize()` to get the full `GenerationResult`.
  See the class docstring below for the usage pattern.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from citeformer.backends.base import Backend
from citeformer.core import Citation, GenerationResult, Policy, Reference, Source
from citeformer.render import render_references

# Pattern for extracting [N] markers from generated text. The grammar layer
# constrains N to the valid source id range at decode time; this pattern only
# does the post-hoc parsing, not enforcement.
_CITE_PATTERN = re.compile(r"\[(\d+)\]")

# Run-of-markers pattern used by `deduplicate_adjacent_cites`. Matches two or
# more `[N]` markers separated only by whitespace — the exact "stacking"
# shape the REQUIRED policy produces on small models that want to close a
# sentence but keep choosing in-range cite ids.
_CITE_RUN_PATTERN = re.compile(r"(?:\[\d+\]\s*){2,}\[\d+\]")


def deduplicate_adjacent_cites(text: str) -> str:
    """Collapse runs of adjacent ``[N]`` markers to the unique ids only.

    The REQUIRED policy's grammar allows ``cite-group ::= cite-id (ws
    cite-id)*`` — more than one citation between content and ``sent-end``.
    Small instruction-tuned models under REQUIRED often emit runs like
    ``[1] [2] [3] [1] [2] [3] [1]`` when closing a sentence where they
    *wanted* to cite something out-of-scope: the grammar forces progress,
    but the model fills the cite-group by cycling valid ids.

    This helper rewrites each such run to contain each cite id at most
    once, preserving order of first appearance. ``[1] [2] [3] [1] [2]`` →
    ``[1] [2] [3]``. Single markers are untouched.

    Args:
        text: The generated text (``GenerationResult.text``).

    Returns:
        The same string with adjacent-cite runs deduplicated. Non-citation
        content is copied verbatim.

    Example:
        >>> deduplicate_adjacent_cites("Foo [1] [2] [3] [1] [2]. Bar [4].")
        'Foo [1] [2] [3]. Bar [4].'
    """

    def _collapse(match: re.Match[str]) -> str:
        ids = _CITE_PATTERN.findall(match.group(0))
        # Dedupe preserving first-appearance order.
        seen: list[str] = []
        for i in ids:
            if i not in seen:
                seen.append(i)
        return " ".join(f"[{i}]" for i in seen)

    return _CITE_RUN_PATTERN.sub(_collapse, text)


class Citeformer:
    """High-level orchestrator for generating citation-backed text.

    Wraps a `Backend` with a citation policy and a CSL style. References are
    rendered deterministically by the home-grown formatters in
    ``citeformer.render.formatters`` — the model never emits bibliography text.

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
        style: CSL style identifier (e.g. `"apa-7"`) for the reference renderer.
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
            style: CSL style identifier for reference rendering (see
                ``bundled_style_names()`` for available styles).
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
            sources=list(sources),
        )

    def stream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy | None = None,
        **options: Any,
    ) -> StreamingResult:
        """Stream generation as text chunks while tracking state for finalization.

        Returns a `StreamingResult` which is both iterable (yielding decoded
        chunks in order) and finalizable (building a full `GenerationResult`
        from the accumulated text after the stream completes).

        Typical usage::

            stream = cf.stream(prompt="...", sources=sources)
            for chunk in stream:
                print(chunk, end="", flush=True)
            result = stream.finalize()  # full GenerationResult with refs + verify()

        If you call `.finalize()` without consuming the iterator first, it will
        exhaust the iterator internally so you get a complete result either way.

        Args:
            prompt: User prompt, same semantics as `generate()`.
            sources: Sources in scope (position → citation id).
            policy: Override the default `citation_policy` for this call.
            **options: Forwarded to `Backend.stream()`.

        Returns:
            A `StreamingResult` wrapping the backend's chunk iterator.
        """
        effective_policy = policy if policy is not None else self.citation_policy
        chunks = self.backend.stream(
            prompt=prompt,
            sources=sources,
            policy=effective_policy,
            **options,
        )
        return StreamingResult(chunks=chunks, sources=list(sources), style=self.style)

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
        """Render the reference list via the built-in formatters.

        Delegates to `citeformer.render.render_references`. Out-of-range cite
        ids are skipped silently (grammar-level enforcement prevents them at
        decode time; this is the belt-and-suspenders path for mock backends
        and hand-constructed tests).
        """
        return render_references(sources, citations, self.style)


class StreamingResult:
    """Iterable wrapper over a backend's streaming output.

    Acts as an `Iterator[str]`, yielding decoded chunks in order. Internally
    accumulates the full text so that after iteration completes, `finalize()`
    can parse citations + render references and return a `GenerationResult`
    identical to what `Citeformer.generate()` would have produced for the
    same (prompt, sources, policy).

    Idempotent: calling `finalize()` multiple times returns the same
    `GenerationResult` instance. Calling it before exhausting the iterator
    consumes the remaining chunks so the result is complete — partial
    finalize isn't supported by design; if you want the partial text at
    any point, use the `text` property.

    Attributes:
        sources: Sources passed to `Citeformer.stream()`.
        style: CSL style used to render references.
    """

    def __init__(
        self,
        *,
        chunks: Iterator[str],
        sources: list[Source],
        style: str,
    ) -> None:
        """Wrap a backend chunk iterator. Not for direct construction by users."""
        self._chunks = chunks
        self.sources = sources
        self.style = style
        self._accumulated: list[str] = []
        self._finalized: GenerationResult | None = None

    def __iter__(self) -> StreamingResult:
        return self

    def __next__(self) -> str:
        chunk = next(self._chunks)
        self._accumulated.append(chunk)
        return chunk

    @property
    def text(self) -> str:
        """The text consumed so far. Updates as iteration progresses."""
        return "".join(self._accumulated)

    def finalize(self) -> GenerationResult:
        """Exhaust the iterator (if needed) and build the full `GenerationResult`.

        Safe to call multiple times — the first call caches the result and
        subsequent calls return the same instance.
        """
        if self._finalized is not None:
            return self._finalized
        # Exhaust any remaining chunks so the result is complete.
        for _ in self:
            pass
        text = self.text
        citations = [
            Citation(span=(m.start(), m.end()), source_id=int(m.group(1)))
            for m in _CITE_PATTERN.finditer(text)
        ]
        references = render_references(self.sources, citations, self.style)
        self._finalized = GenerationResult(
            text=text,
            citations=citations,
            references=references,
            sources=self.sources,
        )
        return self._finalized
