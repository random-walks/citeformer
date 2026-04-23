"""Third-party framework integrations.

Submodules adapt between citeformer's ``Source`` / ``GenerationResult``
types and the corresponding document / response types of popular RAG
frameworks. They're duck-typed — we don't import LangChain / LlamaIndex
at module load, so you can import ``citeformer.integrations.langchain``
without either library installed, as long as you pass in objects with
the expected attribute shapes.

Submodules:

- :mod:`citeformer.integrations.langchain` — LangChain ``Document`` →
  ``Source``, plus helpers to feed a LangChain retriever output straight
  into ``Citeformer.generate``.
- :mod:`citeformer.integrations.llamaindex` — LlamaIndex ``TextNode`` /
  ``NodeWithScore`` → ``Source``, same pattern.
"""

from __future__ import annotations
