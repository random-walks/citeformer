"""vLLM backend with grammar-level citation enforcement.

vLLM supports multiple guided-decoding backends (``xgrammar``, ``outlines``,
``lm-format-enforcer``, ``llguidance``). We pick XGrammar by default because
(a) it's vLLM's default in 2026, (b) it's what our HF backend already uses,
so a user running the same grammar through both gets identical decode-time
semantics.

Requires the ``vllm`` extra: ``pip install citeformer[vllm]``. **Linux with
CUDA only.** vLLM doesn't ship macOS or Windows wheels as of April 2026, so
this backend is excluded from the ``all`` extra and from the integration
tests that run on non-Linux hosts.
"""

from __future__ import annotations

import logging
from typing import Any

from citeformer.backends.base import Backend
from citeformer.core import Policy, Source
from citeformer.grammar import build_grammar

_LOG = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 256
_DEFAULT_TEMPERATURE = 0.7


class VLLMBackend(Backend):
    """vLLM backend with grammar-level citation enforcement.

    Wraps ``vllm.LLM`` for offline batched generation. Uses XGrammar as the
    constrained-decoding backend by default; override via the
    ``guided_decoding_backend`` constructor kwarg (``"llguidance"`` is the
    next-best choice for fast TTFT on simple grammars).

    Attributes:
        model_name: HuggingFace model identifier.
        guided_decoding_backend: vLLM's guided-decoding backend selector.
        llm: The loaded ``vllm.LLM`` instance.
    """

    model_name: str
    guided_decoding_backend: str
    llm: Any

    def __init__(
        self,
        model: str,
        *,
        guided_decoding_backend: str = "xgrammar",
        **llm_kwargs: Any,
    ) -> None:
        """Load a model with vLLM.

        Args:
            model: HuggingFace model identifier (or a local path vLLM can load).
            guided_decoding_backend: Constrained-decoding backend. Common
                choices: ``"xgrammar"`` (default), ``"llguidance"``,
                ``"outlines"``, ``"lm-format-enforcer"``.
            **llm_kwargs: Forwarded to ``vllm.LLM``. Useful ones: ``dtype``,
                ``tensor_parallel_size``, ``gpu_memory_utilization``,
                ``max_model_len``, ``enforce_eager``.

        Raises:
            ImportError: If ``citeformer[vllm]`` extras aren't installed (or
                not available on this platform — vLLM is Linux/CUDA only).
        """
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "VLLMBackend requires the `vllm` extra (Linux/CUDA only). "
                "Install with `pip install citeformer[vllm]`."
            ) from e

        self.model_name = model
        self.guided_decoding_backend = guided_decoding_backend

        _LOG.info("Loading vLLM model %s", model)
        self.llm = LLM(model=model, **llm_kwargs)

    def generate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> str:
        """Generate text with vLLM + grammar-constrained decoding.

        Args:
            prompt: User prompt. Caller assembles any RAG context.
            sources: Sources in scope (must be non-empty).
            policy: Citation enforcement policy.
            **options: Sampling overrides — ``max_new_tokens`` (default 256),
                ``temperature`` (default 0.7). Unknown keys ignored.

        Returns:
            Generated text with only valid ``[N]`` markers.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        if len(sources) < 1:
            raise ValueError("VLLMBackend.generate requires at least 1 source")

        max_new_tokens = int(options.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS))
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))

        grammar = build_grammar(n_sources=len(sources), policy=policy)
        guided = GuidedDecodingParams(
            grammar=grammar.gbnf,
            backend=self.guided_decoding_backend,
        )
        sampling = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            guided_decoding=guided,
        )
        outputs: Any = self.llm.generate([prompt], sampling)
        # vLLM returns a list[RequestOutput]; each has .outputs[0].text.
        return str(outputs[0].outputs[0].text)
