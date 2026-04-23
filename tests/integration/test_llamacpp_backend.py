"""Integration tests for ``LlamaCppBackend``.

Auto-downloads a small Qwen 2.5 0.5B GGUF on first run (~370 MB, one-time)
and caches under the HF hub cache. On subsequent runs, tests replay against
the cached model. Marked ``integration`` so the default ``pytest`` run
skips these.
"""

from __future__ import annotations

import re

import pytest

from citeformer import Citeformer, Policy, Source

_CITE = re.compile(r"\[(\d+)\]")

# Tiny chat-tuned GGUF — publicly available, Apache-2.0, ~370MB Q4_K_M.
# Works on Mac CPU / Apple Silicon Metal + Linux CPU / CUDA.
_GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
_GGUF_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def _sources(n: int) -> list[Source]:
    return [
        Source(
            metadata={"id": f"src-{i}", "type": "book", "title": f"Book {i}"},
            content=f"Content chunk {i}",
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture(scope="module")
def gguf_path() -> str:
    """Download (or cache-hit) a tiny GGUF model."""
    pytest.importorskip("llama_cpp")
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import hf_hub_download

    return str(hf_hub_download(repo_id=_GGUF_REPO, filename=_GGUF_FILE))


@pytest.fixture(scope="module")
def llamacpp_backend(gguf_path):  # type: ignore[no-untyped-def]
    from citeformer.backends.llamacpp import LlamaCppBackend

    return LlamaCppBackend(model_path=gguf_path, n_ctx=512, verbose=False)


@pytest.mark.integration
def test_llamacpp_grammar_compiles_against_llama_cpp_parser() -> None:
    """§10.1 grammar compiles with llama.cpp's native GBNF parser.

    This test only needs ``llama_cpp`` installed — no model load — so it's
    the fastest smoke check that our emitted GBNF is wire-compatible with
    the third local backend (not just xgrammar).
    """
    pytest.importorskip("llama_cpp")
    from llama_cpp import LlamaGrammar

    from citeformer.grammar import build_grammar

    for policy in (Policy.REQUIRED, Policy.AUTO, Policy.QUOTES_ONLY):
        g = build_grammar(n_sources=5, policy=policy)
        grammar = LlamaGrammar.from_string(g.gbnf, verbose=False)
        assert grammar is not None


@pytest.mark.integration
def test_llamacpp_cannot_fabricate_citations(llamacpp_backend) -> None:  # type: ignore[no-untyped-def]
    """The P5 flagship assertion: no `[N+k]` with N sources in scope."""
    sources = _sources(3)
    cf = Citeformer(backend=llamacpp_backend, citation_policy=Policy.REQUIRED)
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
def test_llamacpp_cannot_fabricate_citations_auto(llamacpp_backend) -> None:  # type: ignore[no-untyped-def]
    sources = _sources(2)
    cf = Citeformer(backend=llamacpp_backend, citation_policy=Policy.AUTO)
    result = cf.generate(
        prompt="Describe the books briefly.",
        sources=sources,
        max_new_tokens=50,
        temperature=0.3,
    )
    emitted = [int(m.group(1)) for m in _CITE.finditer(result.text)]
    for cid in emitted:
        assert 1 <= cid <= 2


@pytest.mark.integration
def test_llamacpp_rejects_empty_sources(llamacpp_backend) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="at least 1 source"):
        llamacpp_backend.generate(
            prompt="x",
            sources=[],
            policy=Policy.AUTO,
        )
