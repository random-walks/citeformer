"""Verification module for citeformer.

Three checks compose into one `VerificationReport`:

1. **Existence** — do cited source_ids resolve to real sources? Trivially
   true under Tier 1 grammar enforcement; valuable on other backends.
2. **Entailment** — NLI-score each citation against its cited source.
3. **Coverage** — for uncited sentences, ask whether any source would
   entail them.

Public API:

- `Verifier` — orchestrator. Instantiate with a threshold + optional
  preloaded `NLIModel`.
- `NLIModel` — the transformers-based NLI scorer (DeBERTa-v3-MNLI by
  default).
- Data shapes: `VerificationReport`, `CitationSupport`, `UncitedClaim`.

`GenerationResult.verify(sources=...)` wraps `Verifier` with sensible
defaults and is the usual entry point for callers.
"""

from __future__ import annotations

from citeformer.verify.coverage import find_uncited_but_entailed
from citeformer.verify.entailment import score_entailment
from citeformer.verify.existence import ExistenceResult, check_existence
from citeformer.verify.nli import DEFAULT_NLI_MODEL, NLIModel, NLIResult
from citeformer.verify.report import CitationSupport, UncitedClaim, VerificationReport
from citeformer.verify.sentences import SentenceSpan, sentence_containing, split_sentences
from citeformer.verify.verifier import Verifier

__all__ = [
    "DEFAULT_NLI_MODEL",
    "CitationSupport",
    "ExistenceResult",
    "NLIModel",
    "NLIResult",
    "SentenceSpan",
    "UncitedClaim",
    "VerificationReport",
    "Verifier",
    "check_existence",
    "find_uncited_but_entailed",
    "score_entailment",
    "sentence_containing",
    "split_sentences",
]
