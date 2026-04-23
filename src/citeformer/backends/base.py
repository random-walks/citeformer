"""Abstract backend interface (populated in P1).

Every concrete backend (`HFBackend`, `VLLMBackend`, `LlamaCppBackend`, `MockBackend`)
implements this ABC so the rest of citeformer can stay runtime-agnostic.
"""

from __future__ import annotations
