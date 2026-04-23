"""citeformer — a bulletproof way to generate verifiably cited text from language models.

The public API surface:

- `Citeformer` — the orchestrator. Wrap a `Backend`, call `generate()`.
- Types: `Source`, `Citation`, `Reference`, `GenerationResult`, `Policy`.
- Verification: `VerificationReport`, `CitationSupport`.
- Backends: `Backend` (ABC) and `MockBackend` live here. Concrete backends
  live on submodules that you import only if you install their extras:
  ``citeformer.backends.hf.HFBackend``,
  ``citeformer.backends.llamacpp.LlamaCppBackend``,
  ``citeformer.backends.vllm.VLLMBackend``.
- Prompt helper: `build_rag_prompt` — assembles the canonical RAG prompt.
- Metadata helpers: `fetch_crossref`, `fetch_arxiv`, `extract_pdf`, `extract_url`.

See https://citeformer.readthedocs.io for documentation.
"""

from __future__ import annotations

from citeformer._version import __version__
from citeformer.backends import Backend, MockBackend
from citeformer.citeformer import Citeformer, StreamingResult
from citeformer.core import Citation, GenerationResult, Policy, Reference, Source
from citeformer.metadata import extract_pdf, extract_url, fetch_arxiv, fetch_crossref
from citeformer.prompts import build_rag_prompt
from citeformer.render import bundled_style_names, render_references
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
    "StreamingResult",
    "VerificationReport",
    "__version__",
    "build_rag_prompt",
    "bundled_style_names",
    "extract_pdf",
    "extract_url",
    "fetch_arxiv",
    "fetch_crossref",
    "render_references",
]
