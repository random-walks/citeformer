"""citeformer — a bulletproof way to generate verifiably cited text from language models.

Public API (populated across phases):

- `Source`, `Citation`, `Reference`, `GenerationResult`, `Policy`   — P1
- `Citeformer` orchestrator                                          — P1/P2
- Backend classes (`HFBackend`, `VLLMBackend`, `LlamaCppBackend`)    — P2/P5
- `verify()` entry points                                            — P6

See https://citeformer.readthedocs.io for the full documentation.
"""

from __future__ import annotations

from citeformer._version import __version__

__all__ = ["__version__"]
