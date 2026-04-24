"""Integration tests for ``VLLMBackend``.

vLLM is Linux + CUDA only — these tests skip on other platforms. When they
run, they load a tiny model (Qwen 2.5 0.5B Instruct) and verify the same
"can't fabricate citations" guarantee the HF and llama.cpp backends provide,
now via vLLM's native guided-decoding with XGrammar.
"""

from __future__ import annotations

import re
import sys

import pytest

from citeformer import Citeformer, Policy, Source

_CITE = re.compile(r"\[(\d+)\]")
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _sources(n: int) -> list[Source]:
    return [
        Source(
            metadata={"id": f"src-{i}", "type": "book", "title": f"Book {i}"},
            content=f"Content chunk {i}",
        )
        for i in range(1, n + 1)
    ]


def _vllm_runnable() -> bool:
    """Return True iff vLLM is installed AND we're on Linux with CUDA."""
    if sys.platform != "linux":
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False
    return True


vllm_required = pytest.mark.skipif(
    not _vllm_runnable(),
    reason="vLLM integration requires Linux + CUDA + `citeformer[vllm]`.",
)


@pytest.fixture(scope="module")
def vllm_backend():  # type: ignore[no-untyped-def]
    from citeformer.backends.vllm import VLLMBackend

    return VLLMBackend(
        model=_MODEL,
        max_model_len=1024,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
    )


@pytest.mark.integration
@vllm_required
def test_vllm_cannot_fabricate_citations(vllm_backend) -> None:  # type: ignore[no-untyped-def]
    """P5 flagship: no `[N+k]` with N sources, via vLLM's guided decoding."""
    sources = _sources(3)
    cf = Citeformer(backend=vllm_backend, citation_policy=Policy.REQUIRED)
    result = cf.generate(
        prompt="Write one short sentence mentioning the books.",
        sources=sources,
        max_new_tokens=60,
        temperature=0.3,
    )
    emitted = [int(m.group(1)) for m in _CITE.finditer(result.text)]
    for cid in emitted:
        assert 1 <= cid <= 3, f"FABRICATED cite {cid} in text: {result.text!r}"


@pytest.mark.integration
@vllm_required
def test_vllm_rejects_empty_sources(vllm_backend) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="at least 1 source"):
        vllm_backend.generate(
            prompt="x",
            sources=[],
            policy=Policy.AUTO,
        )
