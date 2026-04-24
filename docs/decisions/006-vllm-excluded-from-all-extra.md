# ADR-006 — `vllm` excluded from the `all` extra

- **Status**: Accepted (P0, re-confirmed P5, 2026-04-23).

## Context

vLLM is Linux + CUDA only. The upstream project has no macOS or Windows wheels as of April 2026, and attempting `pip install vllm` on macOS fails with build errors.

If `vllm` were in the `all` extra, then `uv sync --all-extras` (or `pip install citeformer[all]`) would fail on non-Linux hosts — even for users who never intend to use the vLLM backend. That's a papercut for every Mac / Windows developer.

## Decision

Keep `vllm` as its own extra (`citeformer[vllm]`) but **exclude it from `all`**. Users on compatible hosts install it explicitly. The `all` extra remains `citeformer[hf, llamacpp, verify]` — cross-platform safe.

## Consequences

- `uv sync --all-extras` works on macOS/Linux/Windows alike.
- `make test-integration` syncs `dev + hf + llamacpp` — no vLLM. Linux/CUDA users add `--extra vllm` themselves.
- The `VLLMBackend` class still imports cleanly on any platform (lazy imports in `__init__`); only instantiating it requires `vllm` to be installed.
- `test_vllm_backend.py` has a `_vllm_runnable()` helper that short-circuits to `pytest.skip` unless the runtime is Linux + CUDA + `vllm` installed. This means a Linux CI job that wants to exercise vLLM just needs to install `citeformer[vllm]` and run `pytest -m integration`; everything else auto-gates.
- Documentation must be explicit: our README and docs call out the Linux/CUDA gate for vLLM. Currently in `docs/decisions/006-vllm-excluded-from-all-extra.md` and the `VLLMBackend` docstring.
