"""Abstract backend interface.

Every concrete backend (`HFBackend`, `VLLMBackend`, `LlamaCppBackend`, `MockBackend`)
implements this ABC. The orchestration layer (`Citeformer`) is backend-agnostic and
delegates generation via this interface, keeping grammar-building and decoding logic
scoped to the backend that cares about them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from citeformer.core import Policy, Source


class Backend(ABC):
    """Abstract backend for citeformer.

    Subclasses implement `generate()` against a specific model runtime (HF transformers,
    vLLM, llama.cpp, plus the `MockBackend`). Constrained-decoding grammar construction
    is the backend's responsibility — each runtime has a different native format
    (XGrammar object, GBNF string, etc.), so the shared `grammar/builder.py` module
    emits a backend-agnostic intermediate representation that each backend converts as
    needed.

    Subclasses may optionally override `stream()` to yield chunks as the model decodes
    them. The default implementation falls back to `generate()` and emits the full
    text as a single chunk — any backend works with `Citeformer.stream()`, but only
    overriding backends deliver true token-by-token behavior.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> str:
        """Generate text with citation markers constrained to the given sources.

        The returned string contains inline `[N]` markers where `N` is a 1-indexed
        position into `sources`. On grammar-level-enforcing backends (HF, vLLM,
        llama.cpp via their constrained-decoding integration) emitting an `[N]` for
        `N > len(sources)` is token-impossible; `MockBackend` in tests just respects
        the contract by construction.

        Args:
            prompt: User prompt. The orchestration layer is responsible for
                constructing a retrieval-augmented prompt (stitching in source
                snippets); this method receives the final prompt string.
            sources: Sources in scope; position determines citation index.
            policy: Citation enforcement policy.
            **options: Backend-specific decoding options (e.g. `max_tokens`,
                `temperature`, `seed`). Unknown options are silently ignored.

        Returns:
            The generated text with inline markers. References are not part of the
            backend output — the orchestration layer renders them separately.
        """

    def stream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> Iterator[str]:
        """Yield generation output as a stream of text chunks.

        The default implementation is not a true stream: it calls `generate()` and
        yields the full text as a single chunk. Backends that can produce
        token-by-token output (HF, llama.cpp) should override this to yield chunks
        as they're decoded — the grammar constraints apply to every yielded chunk
        exactly as in `generate()`.

        Args:
            prompt: See `generate()`.
            sources: See `generate()`.
            policy: See `generate()`.
            **options: See `generate()`. Most backends also accept streaming-specific
                hints here (e.g. XGrammar's LogitsProcessor is already stateful, so
                no extra plumbing is needed).

        Yields:
            Text chunks in the order they're produced. Joining all yielded chunks
            reconstructs what `generate()` would have returned.
        """
        yield self.generate(prompt=prompt, sources=sources, policy=policy, **options)
