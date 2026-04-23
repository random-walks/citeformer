"""Backend implementations for citeformer.

Each backend adapts a model runtime (HF transformers, vLLM, llama.cpp) to a common
`Backend` ABC (see `base.py`). Backends are populated across phases — `MockBackend`
ships in P1, `HFBackend` in P2, `VLLMBackend` and `LlamaCppBackend` in P5.
"""

from __future__ import annotations

from citeformer.backends.base import Backend
from citeformer.backends.mock import MockBackend

__all__ = ["Backend", "MockBackend"]
