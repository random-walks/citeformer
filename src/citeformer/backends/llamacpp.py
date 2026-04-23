"""llama.cpp backend via ``llama-cpp-python``.

llama.cpp has native GBNF support, which is exactly what our grammar builder
emits — so the integration is a one-liner: hand the grammar string to
``LlamaGrammar.from_string`` and pass the result to ``Llama.__call__``.

Requires the ``llamacpp`` extra: ``pip install citeformer[llamacpp]``. You
also need a GGUF model file on disk (llama-cpp-python consumes GGUF, not
HuggingFace weights directly — use ``huggingface_hub`` or the ``convert-hf-
to-gguf`` script in the llama.cpp repo to produce one).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from citeformer.backends.base import Backend
from citeformer.core import Policy, Source
from citeformer.grammar import DEFAULT_MAX_CONTENT_CHARS, build_grammar

_LOG = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 256
_DEFAULT_TEMPERATURE = 0.7


class LlamaCppBackend(Backend):
    """llama.cpp backend with grammar-level citation enforcement.

    Wraps ``llama_cpp.Llama``. Citation markers are constrained at decode time
    by ``LlamaGrammar.from_string(grammar.gbnf)`` — same GBNF string our
    builder emits for XGrammar, no translation layer needed.

    Attributes:
        model_path: Local filesystem path to a GGUF model.
        n_ctx: Context window size.
        n_gpu_layers: How many layers to offload to GPU (Metal/CUDA). ``-1``
            for all, ``0`` for CPU-only.
        llm: The loaded ``llama_cpp.Llama`` instance.
    """

    model_path: str
    n_ctx: int
    n_gpu_layers: int
    llm: Any

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ) -> None:
        """Load a GGUF model via ``llama-cpp-python``.

        Args:
            model_path: Filesystem path to a GGUF file.
            n_ctx: Context window size (tokens). Larger = more memory.
            n_gpu_layers: Layers to offload to GPU. ``-1`` offloads all
                (fastest on Metal / CUDA); ``0`` runs on CPU only.
            verbose: Whether to print llama.cpp's decoding diagnostics.

        Raises:
            ImportError: If ``citeformer[llamacpp]`` extras aren't installed.
        """
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "LlamaCppBackend requires the `llamacpp` extra. "
                "Install with `pip install citeformer[llamacpp]`."
            ) from e

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

        _LOG.info("Loading GGUF model %s", model_path)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def generate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> str:
        """Generate text with llama.cpp's GBNF-constrained decoder.

        Args:
            prompt: User prompt. Caller assembles any RAG context.
            sources: Sources in scope (must be non-empty).
            policy: Citation enforcement policy.
            **options: Sampling + grammar overrides — ``max_new_tokens``
                (default 256), ``temperature`` (default 0.7),
                ``max_content_chars`` (REQUIRED-policy progression bound; see
                ADR-009). Unknown keys ignored.

        Returns:
            Generated text with only valid ``[N]`` markers.
        """
        max_new_tokens, temperature, llama_grammar = self._prepare(sources, policy, options)

        result: Any = self.llm(
            prompt,
            grammar=llama_grammar,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return str(result["choices"][0]["text"])

    def stream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> Iterator[str]:
        """Stream text chunks from llama.cpp's native streaming mode.

        `llama-cpp-python`'s `Llama.__call__(..., stream=True)` yields dicts
        shaped like the non-streaming result: each has
        ``["choices"][0]["text"]`` with the newly-decoded piece. Grammar
        enforcement is identical to `generate()`.
        """
        max_new_tokens, temperature, llama_grammar = self._prepare(sources, policy, options)
        stream_iter: Any = self.llm(
            prompt,
            grammar=llama_grammar,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream_iter:
            piece = chunk.get("choices", [{}])[0].get("text", "")
            if piece:
                yield str(piece)

    def _prepare(
        self,
        sources: list[Source],
        policy: Policy,
        options: dict[str, Any],
    ) -> tuple[int, float, Any]:
        """Shared setup for generate/stream: validate, build + compile grammar."""
        from llama_cpp import LlamaGrammar  # type: ignore[attr-defined,unused-ignore]

        if len(sources) < 1:
            raise ValueError("LlamaCppBackend requires at least 1 source")

        max_new_tokens = int(options.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS))
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))
        max_content_chars = options.get("max_content_chars", DEFAULT_MAX_CONTENT_CHARS)

        grammar = build_grammar(
            n_sources=len(sources),
            policy=policy,
            max_content_chars=max_content_chars,
        )
        llama_grammar = LlamaGrammar.from_string(grammar.gbnf, verbose=False)
        return max_new_tokens, temperature, llama_grammar
