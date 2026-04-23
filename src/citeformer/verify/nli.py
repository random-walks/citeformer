"""Natural-language-inference backend for verification.

We wrap a DeBERTa-v3 MNLI model via ``transformers``. The model is lazy-loaded
on first `entail()` call, cached globally per (model_name, device) so multiple
`Verifier` instances share weights. Batched scoring is the common path —
single-pair calls funnel through the batched API with a one-element batch.

Default model: ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``
(~850 MB; well-tested on scientific claims). Override via the ``nli_model``
kwarg on `Verifier`. A smaller / faster default can be swapped in at build
time by setting the ``CITEFORMER_NLI_MODEL`` env var.

Requires the ``verify`` extra: ``pip install citeformer[verify]``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

_LOG = logging.getLogger(__name__)

# Default NLI model — overridable via CITEFORMER_NLI_MODEL.
DEFAULT_NLI_MODEL = os.environ.get(
    "CITEFORMER_NLI_MODEL",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
)

# Fallback probability assignments when the model's label set doesn't match
# the canonical ``entailment`` / ``neutral`` / ``contradiction`` triple.
_LABEL_ALIASES = {
    "entailment": "entailment",
    "entail": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "contradict": "contradiction",
    "not_entailment": "contradiction",
    "label_0": "entailment",  # some stock MNLI heads use indices
    "label_1": "neutral",
    "label_2": "contradiction",
}


@dataclass(frozen=True)
class NLIResult:
    """One NLI scoring outcome for a (premise, hypothesis) pair.

    Attributes:
        entailment: Probability of the ``entailment`` class in [0, 1].
        neutral: Probability of the ``neutral`` class.
        contradiction: Probability of the ``contradiction`` class.
    """

    entailment: float
    neutral: float
    contradiction: float

    @property
    def supports(self) -> bool:
        """True if the entailment class is the predicted label.

        Equivalent to ``entailment > max(neutral, contradiction)``; thresholded
        use sites that want a hard cutoff should compare ``entailment`` to
        a configured threshold directly.
        """
        return self.entailment > self.neutral and self.entailment > self.contradiction


class NLIModel:
    """DeBERTa-v3-MNLI (or drop-in compatible) NLI scorer.

    Instances are cheap to construct; weights are loaded on first `entail()`.
    The transformers model + tokenizer are cached globally per
    (model_name, device) via `functools.lru_cache` so multiple `NLIModel`
    instances with identical config share a single GPU residence.

    Attributes:
        model_name: HuggingFace model identifier.
        device: Torch device (``cuda`` / ``mps`` / ``cpu``) resolved at
            construction.
        batch_size: Max pairs to score in a single forward pass.
    """

    model_name: str
    device: str
    batch_size: int

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        """Construct an `NLIModel`.

        Args:
            model_name: HF identifier (e.g. ``"MoritzLaurer/DeBERTa-…"``).
            device: ``None`` auto-detects CUDA > MPS > CPU.
            batch_size: Max pairs per forward pass; adjust down on low-VRAM
                hardware.

        Raises:
            ImportError: If ``citeformer[verify]`` extras aren't installed.
        """
        try:
            import torch  # noqa: F401 — probed for device detection
        except ImportError as e:
            raise ImportError(
                "NLIModel requires the `verify` extra. "
                "Install with `pip install citeformer[verify]`."
            ) from e
        self.model_name = model_name
        self.device = device if device is not None else self._autodetect_device()
        self.batch_size = batch_size

    @staticmethod
    def _autodetect_device() -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def entail(self, premise: str, hypothesis: str) -> NLIResult:
        """Score a single (premise, hypothesis) pair.

        Args:
            premise: The evidence / source text.
            hypothesis: The claim being checked against the premise.

        Returns:
            An `NLIResult` with per-class probabilities.
        """
        results = self.entail_batch([(premise, hypothesis)])
        return results[0]

    def entail_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Score a list of (premise, hypothesis) pairs in batches.

        Empty input returns an empty list.

        Args:
            pairs: A list of ``(premise, hypothesis)`` tuples.

        Returns:
            Results in the same order as input.
        """
        if not pairs:
            return []

        import torch
        from torch.nn import functional as torch_functional

        tokenizer, model, label_map = _load_nli(self.model_name, self.device)

        outputs: list[NLIResult] = []
        for chunk_start in range(0, len(pairs), self.batch_size):
            chunk = pairs[chunk_start : chunk_start + self.batch_size]
            premises = [p for p, _ in chunk]
            hypotheses = [h for _, h in chunk]
            encoded = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = model(**encoded).logits
            probs = torch_functional.softmax(logits, dim=-1)

            for row in probs:
                entail = float(row[label_map["entailment"]])
                neutral = float(row[label_map["neutral"]])
                contra = float(row[label_map["contradiction"]])
                outputs.append(NLIResult(entailment=entail, neutral=neutral, contradiction=contra))
        return outputs


@lru_cache(maxsize=4)
def _load_nli(model_name: str, device: str) -> tuple[Any, Any, dict[str, int]]:
    """Load an NLI tokenizer + model and resolve its label map.

    Returns:
        (tokenizer, model, label_map) where ``label_map`` maps canonical
        class names (``"entailment"``, ``"neutral"``, ``"contradiction"``)
        to the row index in the model's logits tensor.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _LOG.info("Loading NLI model %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_model: Any = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = raw_model.to(device)
    model.eval()

    # Build canonical-name → logit-index map. DeBERTa-v3-MNLI puts entailment
    # at index 0, neutral 1, contradiction 2 — but some models use different
    # orderings, and label names vary ("ENTAILMENT" / "entailment" /
    # "label_0"). Walk id2label with alias fallback.
    id2label = getattr(model.config, "id2label", {})
    label_map: dict[str, int] = {}
    for idx, raw in id2label.items():
        canonical = _LABEL_ALIASES.get(str(raw).lower())
        if canonical is not None:
            label_map[canonical] = int(idx)

    if set(label_map) != {"entailment", "neutral", "contradiction"}:
        # Fall back to canonical MNLI ordering; log a warning so the user
        # can swap models if scoring looks weird.
        _LOG.warning(
            "NLI model %s has unexpected id2label=%s; falling back to "
            "entailment=0, neutral=1, contradiction=2 ordering.",
            model_name,
            id2label,
        )
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    return tokenizer, model, label_map
