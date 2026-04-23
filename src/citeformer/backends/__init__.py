"""Backend implementations for citeformer.

Each backend adapts a model runtime (HF transformers, vLLM, llama.cpp) to a common
`Backend` ABC (see `base.py`). Backends are populated across phases — see the plan file
at `/Users/blaise/.claude/plans/ok-i-setup-this-frolicking-graham.md` for phase details.
"""

from __future__ import annotations
