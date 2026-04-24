"""Metadata adapters — fetch CSL-JSON from DOIs, arXiv, PDFs, URLs.

Each adapter is a plain function that takes the identifier and returns either
a CSL-JSON metadata dict (for DOI / arXiv) or a `(metadata, content)` tuple
(for PDF / URL, where content is the paper text). The `Source.from_*`
classmethods on `citeformer.core.Source` are thin wrappers around these.

Results are cached by default in ``~/.cache/citeformer/metadata/`` via
diskcache. Pass ``use_cache=False`` to bypass. Override the location with
the ``CITEFORMER_CACHE_DIR`` environment variable.
"""

from __future__ import annotations

from citeformer.metadata.arxiv import fetch_arxiv
from citeformer.metadata.bibtex import (
    BIBTEX_TYPE_MAP,
    bibtex_to_csl_json,
    load_bibtex,
    parse_bibtex,
)
from citeformer.metadata.cache import get_metadata_cache
from citeformer.metadata.crossref import fetch_crossref
from citeformer.metadata.pdf import extract_pdf
from citeformer.metadata.url import extract_url
from citeformer.metadata.zotero import load_zotero_csl

__all__ = [
    "BIBTEX_TYPE_MAP",
    "bibtex_to_csl_json",
    "extract_pdf",
    "extract_url",
    "fetch_arxiv",
    "fetch_crossref",
    "get_metadata_cache",
    "load_bibtex",
    "load_zotero_csl",
    "parse_bibtex",
]
