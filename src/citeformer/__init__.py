"""citeformer — a bulletproof way to generate verifiably cited text from language models.

The public API surface:

- `Citeformer` — the orchestrator. Wrap a `Backend`, call `generate()`.
- Types: `Source`, `Citation`, `Reference`, `GenerationResult`, `Policy`.
- Verification: `VerificationReport`, `CitationSupport` (schema in P1; behavior in P6).
- Backends: `Backend` (ABC), `MockBackend` (P1). `HFBackend` lands in P2;
  `VLLMBackend` and `LlamaCppBackend` in P5.

See https://citeformer.readthedocs.io for documentation.
"""

from __future__ import annotations

from citeformer._version import __version__
from citeformer.backends import Backend, MockBackend
from citeformer.citeformer import Citeformer
from citeformer.core import Citation, GenerationResult, Policy, Reference, Source
from citeformer.verify import CitationSupport, VerificationReport

__all__ = [
    "Backend",
    "Citation",
    "CitationSupport",
    "Citeformer",
    "GenerationResult",
    "MockBackend",
    "Policy",
    "Reference",
    "Source",
    "VerificationReport",
    "__version__",
]
