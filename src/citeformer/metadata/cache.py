"""Metadata cache via diskcache.

Default path: ``~/.cache/citeformer/metadata/``. Override with the
``CITEFORMER_CACHE_DIR`` env var (which becomes the parent — the metadata
cache lives under ``<CITEFORMER_CACHE_DIR>/metadata/``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import diskcache


def _resolve_cache_dir() -> Path:
    """Resolve the metadata cache directory.

    Honors ``CITEFORMER_CACHE_DIR`` env var if set; otherwise falls back to
    ``~/.cache/citeformer/``. The returned path is always the ``metadata``
    subdirectory under that root — one metadata cache per root.
    """
    env = os.environ.get("CITEFORMER_CACHE_DIR")
    base = Path(env).expanduser().resolve() if env else Path.home() / ".cache" / "citeformer"
    return base / "metadata"


@lru_cache(maxsize=1)
def get_metadata_cache() -> diskcache.Cache:
    """Return the shared on-disk metadata cache (singleton per process).

    The cache directory is created on first access. Safe to call from
    multiple threads; diskcache is its own lock-based synchronizer.
    """
    cache_dir = _resolve_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(cache_dir))
