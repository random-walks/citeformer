"""Backend implementations for citeformer.

Each backend adapts a model runtime to the common `Backend` ABC (see `base.py`).
Only the `Backend` ABC and `MockBackend` are re-exported here because they
have no optional-extra dependencies. Import the real backends directly from
their submodules — they each require the matching extra:

- ``citeformer.backends.hf.HFBackend`` needs ``pip install citeformer[hf]``.
- ``citeformer.backends.llamacpp.LlamaCppBackend`` needs ``[llamacpp]``.
- ``citeformer.backends.vllm.VLLMBackend`` needs ``[vllm]`` (Linux/CUDA only).
"""

from __future__ import annotations

from citeformer.backends.base import Backend
from citeformer.backends.mock import MockBackend

__all__ = ["Backend", "MockBackend"]
