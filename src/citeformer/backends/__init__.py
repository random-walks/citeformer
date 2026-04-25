"""Backend implementations for citeformer.

Each backend adapts a model runtime to the common `Backend` ABC (see `base.py`).
Only the `Backend` ABC and `MockBackend` are re-exported here because they
have no optional-extra dependencies. Import the real backends directly from
their submodules — they each require the matching extra.

**Local backends (logit-layer enforcement, in-process):**

- ``citeformer.backends.hf.HFBackend`` needs ``pip install citeformer[hf]``.
- ``citeformer.backends.llamacpp.LlamaCppBackend`` needs ``[llamacpp]``.
- ``citeformer.backends.vllm.VLLMBackend`` needs ``[vllm]`` (Linux/CUDA only).

**API backends.** As of late 2025 every modern provider's strict
structured-outputs mode is real token-level masking inside the provider's
runtime — not post-validation. The honest split is "where the masking
runs", not "logit vs schema":

- ``citeformer.backends.openai.OpenAIBackend`` needs ``[openai]``.
  Strict JSON schema → token-level masking inside OpenAI on
  ``gpt-4o-2024-08-06+`` and successors.
- ``citeformer.backends.anthropic.AnthropicBackend`` needs ``[anthropic]``.
  Adapter over Anthropic's native Citations API — provider enforces that
  every cite references a supplied document.
- ``citeformer.backends.gemini.GeminiBackend`` needs ``[gemini]``.
  Constrained generation via ``response_schema`` (OpenAPI-subset).
- ``citeformer.backends.mistral.MistralBackend`` needs ``[mistral]``.
  Strict JSON schema (``mistral-large-2411+``).
- ``citeformer.backends.openrouter.OpenRouterBackend`` needs ``[openrouter]``.
  Multi-provider routing on the OpenAI wire format with
  ``provider.require_parameters`` so requests refuse to land on
  upstreams that can't honour strict mode.

Per-provider tier discussion lives in
``docs/reference/architecture.md``.
"""

from __future__ import annotations

from citeformer.backends.base import Backend
from citeformer.backends.mock import MockBackend

__all__ = ["Backend", "MockBackend"]
