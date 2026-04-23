"""Abstract backend interface.

Every concrete backend (`HFBackend`, `VLLMBackend`, `LlamaCppBackend`, `MockBackend`)
implements this ABC. The orchestration layer (`Citeformer`) is backend-agnostic and
delegates generation via this interface, keeping grammar-building and decoding logic
scoped to the backend that cares about them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from citeformer.core import Policy, Source


class Backend(ABC):
    """Abstract backend for citeformer.

    Subclasses implement `generate()` against a specific model runtime (HF transformers
    in P2, vLLM and llama.cpp in P5, plus the `MockBackend` available since P1).
    Constrained-decoding grammar construction is the backend's responsibility — each
    runtime has a different native format (XGrammar object, GBNF string, etc.), so the
    shared `grammar/builder.py` module (P2) emits a backend-agnostic intermediate
    representation that each backend converts as needed.
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
            backend output — the orchestration layer renders them separately via
            citeproc-py (P3+).
        """
