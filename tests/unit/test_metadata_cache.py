"""Tests for `citeformer.metadata.cache` — diskcache path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from citeformer.metadata import cache as cache_module


def test_default_cache_dir_is_under_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CITEFORMER_CACHE_DIR", raising=False)
    # Bust the lru_cache so _resolve_cache_dir re-reads the env.
    resolved = cache_module._resolve_cache_dir()
    assert resolved == Path.home() / ".cache" / "citeformer" / "metadata"


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CITEFORMER_CACHE_DIR", str(tmp_path))
    resolved = cache_module._resolve_cache_dir()
    assert resolved == tmp_path.resolve() / "metadata"


def test_get_metadata_cache_creates_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CITEFORMER_CACHE_DIR", str(tmp_path))
    cache_module.get_metadata_cache.cache_clear()  # reset the lru_cache wrapper
    cache = cache_module.get_metadata_cache()
    try:
        cache.set("k", "v")
        assert cache.get("k") == "v"
        assert (tmp_path.resolve() / "metadata").exists()
    finally:
        cache.close()
        cache_module.get_metadata_cache.cache_clear()
