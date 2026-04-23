"""HuggingFace transformers backend with grammar-level citation enforcement.

This is the flagship P2 backend: loads a transformers causal LM, builds the
§10.1 citation grammar for the in-scope sources, compiles it with XGrammar's
tokenizer-aware compiler, and masks logits at every decode step so the model
**cannot** emit a citation marker that refers to a non-existent source.

Requires the `hf` extra: ``pip install citeformer[hf]``.

The integration surface is narrow on purpose — xgrammar does the masking,
transformers does the sampling, we just wire them up via
``xgrammar.contrib.hf.LogitsProcessor``. Note that the LogitsProcessor is
stateful per-generation: we construct a new one for every ``generate()`` call
(that's an xgrammar constraint, not ours).
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


class HFBackend(Backend):
    """Transformers + XGrammar backend with logit-level citation enforcement.

    Citation markers are structurally unforgeable on this backend: at every decode
    step XGrammar masks tokens that would produce an ``[N]`` where ``N`` is not a
    valid source index, so sampling simply cannot traverse a path that emits a
    fabricated citation.

    Example:
        >>> from citeformer import Citeformer, Source
        >>> from citeformer.backends.hf import HFBackend
        >>> sources = [Source(metadata={"id": "a", "type": "book"}, content="...")]
        >>> backend = HFBackend(model="gpt2")
        >>> cf = Citeformer(backend=backend)
        >>> result = cf.generate(prompt="Describe the book.", sources=sources)

    Attributes:
        model_name: HF model identifier passed to ``from_pretrained``.
        device: Torch device the model lives on (``cuda`` / ``mps`` / ``cpu``).
        tokenizer: The loaded tokenizer.
        model: The loaded causal LM.
    """

    # Attribute types declared at class level so mypy doesn't try to infer from
    # the transformers imports (which are module-level overrides set to `Any`
    # but can still surface weird torch-side types via the `.to()` chain).
    model_name: str
    device: str
    tokenizer: Any
    model: Any

    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        dtype: str = "auto",
    ) -> None:
        """Load the model + tokenizer and prepare the XGrammar compiler.

        Args:
            model: HuggingFace model identifier (e.g. ``"gpt2"``,
                ``"microsoft/Phi-3.5-mini-instruct"``, or a local path).
            device: Torch device. If ``None``, auto-detect CUDA > MPS > CPU.
            dtype: ``"auto"`` picks bfloat16 on CUDA, float16 on MPS, float32 on
                CPU. Explicit options: ``"bf16"``, ``"fp16"``, ``"fp32"``.

        Raises:
            ImportError: If ``citeformer[hf]`` extras aren't installed.
            ValueError: If ``dtype`` is unrecognized.
        """
        try:
            import torch
            import xgrammar as xgr
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HFBackend requires the `hf` extra. Install with `pip install citeformer[hf]`."
            ) from e

        self.model_name = model
        self.device = device if device is not None else self._autodetect_device(torch)

        _LOG.info("Loading tokenizer + config for %s", model)
        # Intermediate `Any` bindings keep torch's `.to()` overloads from
        # confusing mypy when chained off `AutoModelForCausalLM.from_pretrained`.
        tokenizer_obj: Any = AutoTokenizer.from_pretrained(model)
        self.tokenizer = tokenizer_obj
        config = AutoConfig.from_pretrained(model)

        torch_dtype = self._resolve_dtype(torch, dtype, self.device)
        _LOG.info("Loading model %s on %s (dtype=%s)", model, self.device, torch_dtype)
        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch_dtype,
        )
        self.model = loaded_model.to(self.device)
        self.model.eval()

        # xgr compiler is reused across generate() calls — internal cache keys on
        # the EBNF string, so we only pay grammar compilation once per unique
        # (n_sources, policy) combo.
        self._xgr = xgr
        self._torch = torch
        self._tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=config.vocab_size,  # use config vocab_size, not tokenizer's
        )
        self._compiler = xgr.GrammarCompiler(self._tokenizer_info)

    def generate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> str:
        """Generate text with grammar-masked decoding.

        Args:
            prompt: User prompt. Caller is responsible for assembling any RAG
                context from ``sources`` into the prompt string (no implicit
                stitching — that's a later-phase helper concern).
            sources: Sources in scope. Position (1-indexed) determines the
                citation id the model can emit. Must be non-empty.
            policy: Citation enforcement policy.
            **options: Sampling overrides — ``max_new_tokens`` (default 256),
                ``temperature`` (default 0.7). Unknown options are silently
                ignored.

        Returns:
            The generated text as a string, containing only ``[N]`` markers where
            ``1 <= N <= len(sources)``. Fabrication is structurally impossible.

        Raises:
            ValueError: If ``sources`` is empty.
        """
        if len(sources) < 1:
            raise ValueError("HFBackend.generate requires at least 1 source")

        max_new_tokens = int(options.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS))
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))

        # Build + compile the citation grammar. Compiler cache means compilation
        # is near-free after the first call with a given (n_sources, policy).
        grammar = build_grammar(n_sources=len(sources), policy=policy)
        compiled = self._compiler.compile_grammar(
            grammar.gbnf,
            root_rule_name=grammar.root_rule,
        )

        # LogitsProcessor is stateful per-generation — a fresh instance per call
        # is mandatory (documented xgrammar constraint, not a citeformer choice).
        processor = self._xgr.contrib.hf.LogitsProcessor(compiled)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                logits_processor=[processor],
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
            )

        generated = output_ids[0][inputs.input_ids.shape[1] :]
        decoded: Any = self.tokenizer.decode(generated, skip_special_tokens=True)
        return str(decoded)

    @staticmethod
    def _autodetect_device(torch_module: Any) -> str:
        """Pick the best available device: CUDA > MPS > CPU."""
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(torch_module: Any, dtype: str, device: str) -> Any:
        """Translate the dtype string to a torch dtype object."""
        if dtype == "auto":
            if device == "cuda":
                return torch_module.bfloat16
            if device == "mps":
                return torch_module.float16  # bf16 is flaky on MPS
            return torch_module.float32
        aliases = {
            "bf16": torch_module.bfloat16,
            "fp16": torch_module.float16,
            "fp32": torch_module.float32,
        }
        if dtype in aliases:
            return aliases[dtype]
        raise ValueError(f"Unknown dtype: {dtype!r}. Use 'auto'|'bf16'|'fp16'|'fp32'.")
