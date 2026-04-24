"""Natural-language-inference backend for verification.

We wrap a DeBERTa-v3 MNLI model via ``transformers``. The model is lazy-loaded
on first `entail()` call, cached globally per (model_name, device) so multiple
`Verifier` instances share weights. Batched scoring is the common path —
single-pair calls funnel through the batched API with a one-element batch.

Default model: ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``
(~850 MB; well-tested on scientific claims). Override via the ``nli_model``
kwarg on `Verifier`. A smaller / faster default can be swapped in at build
time by setting the ``CITEFORMER_NLI_MODEL`` env var.

Long premises (>512 tokens) can be **chunked** (opt-in): we slide a
fixed-size window over the premise, score each chunk against the
hypothesis, and take the maximum entailment as the pair's result. That
surfaces claim-to-source entailment that lives past the first 512
tokens — useful when scoring against full PDF body text. But max-over-
windows also inflates false positives on unrelated claims (each extra
window is another chance for noise to cross the threshold), so we keep
it **off by default** for score stability and enable it explicitly via
``chunk_premise=True`` when the caller wants long-document scoring.
When chunking is on, consider raising ``threshold`` on the Verifier
(0.7–0.8 rather than 0.5) to compensate for the max-reduction bias.

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

# DeBERTa-v3 has a 512-token input limit; the joint (premise, hypothesis)
# encoding needs room for CLS/SEP specials + the hypothesis. 400 leaves
# comfortable headroom for typical hypotheses (20-60 tokens).
_DEFAULT_MAX_PREMISE_TOKENS = 400
# Stride = max_tokens - overlap. Default stride of 300 gives ~100 tokens of
# overlap, so a claim that straddles a chunk boundary has a chance to land
# in one of the two neighboring windows.
_DEFAULT_CHUNK_STRIDE = 300


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
        chunk_premise: When ``True``, long premises are split into
            overlapping windows; max entailment across windows is the
            pair's result. Default is ``False`` — max-over-windows
            inflates false positives on unrelated claims. Enable for
            long-document scoring with a bumped ``threshold`` on the
            Verifier (0.7+) to compensate.
        max_premise_tokens: Window size in tokens. Default 400 (leaves room
            for the hypothesis + special tokens inside DeBERTa's 512 cap).
        chunk_stride: Token stride between windows. Default 300; overlap =
            max_premise_tokens - stride.
    """

    model_name: str
    device: str
    batch_size: int
    chunk_premise: bool
    max_premise_tokens: int
    chunk_stride: int

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 8,
        chunk_premise: bool = False,
        max_premise_tokens: int = _DEFAULT_MAX_PREMISE_TOKENS,
        chunk_stride: int = _DEFAULT_CHUNK_STRIDE,
    ) -> None:
        """Construct an `NLIModel`.

        Args:
            model_name: HF identifier (e.g. ``"MoritzLaurer/DeBERTa-…"``).
            device: ``None`` auto-detects CUDA > MPS > CPU.
            batch_size: Max pairs per forward pass; adjust down on low-VRAM
                hardware.
            chunk_premise: If ``True`` (default), long premises are chunked
                and scored with max-entailment reduction. Set to ``False``
                for raw truncation at ``max_premise_tokens + hypothesis``.
            max_premise_tokens: Window size when chunking. 400 is a safe
                default under DeBERTa's 512-token limit.
            chunk_stride: Stride between windows. Lower = more overlap =
                slower but more thorough.

        Raises:
            ImportError: If ``citeformer[verify]`` extras aren't installed.
            ValueError: If `chunk_stride >= max_premise_tokens` (would make
                windows non-overlapping or skip content).
        """
        try:
            import torch  # noqa: F401 — probed for device detection
        except ImportError as e:
            raise ImportError(
                "NLIModel requires the `verify` extra. "
                "Install with `pip install citeformer[verify]`."
            ) from e
        if chunk_stride <= 0 or max_premise_tokens <= 0:
            raise ValueError("chunk_stride and max_premise_tokens must be > 0")
        if chunk_stride > max_premise_tokens:
            raise ValueError(
                f"chunk_stride ({chunk_stride}) must not exceed "
                f"max_premise_tokens ({max_premise_tokens}); that would skip content."
            )
        self.model_name = model_name
        self.device = device if device is not None else self._autodetect_device()
        self.batch_size = batch_size
        self.chunk_premise = chunk_premise
        self.max_premise_tokens = max_premise_tokens
        self.chunk_stride = chunk_stride

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

        Uses chunked scoring when ``chunk_premise`` is enabled and the
        premise is long enough to benefit.

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

        Empty input returns an empty list. Uses chunked scoring when the
        model's ``chunk_premise`` is True; otherwise falls back to the
        naive 512-token truncation path.

        Args:
            pairs: A list of ``(premise, hypothesis)`` tuples.

        Returns:
            Results in the same order as input.
        """
        if not pairs:
            return []
        if self.chunk_premise:
            return self._entail_batch_chunked(pairs)
        return self._entail_batch_truncated(pairs)

    def _entail_batch_truncated(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Naive path: tokenizer handles truncation at max_length=512."""
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

    def _entail_batch_chunked(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Chunked path: window each premise, score all windows, take max entailment.

        For each (premise, hypothesis) pair we split the premise into one or
        more overlapping token-windows (size `max_premise_tokens`, stride
        `chunk_stride`), score every (window, hypothesis) sub-pair via the
        truncated path, and reduce by taking the `NLIResult` with the
        highest `entailment` probability.

        Pairs whose premises fit in a single window (i.e. short premises
        like abstracts) take exactly one forward pass — no extra cost.
        """
        tokenizer, _, _ = _load_nli(self.model_name, self.device)

        expanded: list[tuple[str, str]] = []
        # For each original pair, track [start, end) in the expanded list.
        group_bounds: list[tuple[int, int]] = []
        for premise, hypothesis in pairs:
            windows = _split_premise_into_windows(
                premise,
                tokenizer,
                max_tokens=self.max_premise_tokens,
                stride=self.chunk_stride,
            )
            start = len(expanded)
            for window_text in windows:
                expanded.append((window_text, hypothesis))
            group_bounds.append((start, len(expanded)))

        flat = self._entail_batch_truncated(expanded)

        reduced: list[NLIResult] = []
        for start, end in group_bounds:
            group = flat[start:end]
            if not group:  # shouldn't happen — every input yields ≥1 window
                reduced.append(NLIResult(0.0, 0.0, 0.0))
                continue
            best = max(group, key=lambda r: r.entailment)
            reduced.append(best)
        return reduced


def _split_premise_into_windows(
    premise: str,
    tokenizer: Any,
    *,
    max_tokens: int,
    stride: int,
) -> list[str]:
    """Split a premise string into overlapping token-windows.

    Returns a list of decoded text strings, one per window. Each window
    is at most ``max_tokens`` tokens long; consecutive windows start
    ``stride`` tokens apart (so overlap = max_tokens - stride).

    Short premises (<=max_tokens when tokenized) return a single-element
    list containing the original string — no decode round-trip cost.

    Implementation note: we decode each window back to text so the
    downstream joint (premise, hypothesis) tokenization handles special
    tokens + token type ids correctly. Slight overhead vs. manual
    tensor construction but far more robust across model variants.
    """
    if max_tokens <= 0 or stride <= 0:
        raise ValueError("max_tokens and stride must be > 0")

    # Tokenize without special tokens so length reflects just the content.
    encoded = tokenizer(premise, add_special_tokens=False)
    ids = list(encoded["input_ids"])
    if len(ids) <= max_tokens:
        return [premise]

    windows: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        window_ids = ids[start:end]
        decoded = tokenizer.decode(window_ids, skip_special_tokens=True)
        windows.append(decoded)
        if end == len(ids):
            break
        start += stride
    return windows


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
