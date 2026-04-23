# ADR-005 — Metadata-fetch and render deps go in main, not extras

- **Status**: Accepted (P3/P4, 2026-04-23).

## Context

The P0 scaffolding stratified optional dependencies aggressively: `hf`, `render`, `meta`, `pdf`, `url`, `vllm`, `llamacpp`, `verify`. Rationale was "let users pick what they need." In P3/P4 we reassessed:

- Rendering (`citeproc-py`, or eventually our own — see [ADR-004](004-citeproc-rewrite.md)) is a *core value prop* of citeformer. You don't use this library without wanting rendered references.
- Metadata adapters (`httpx`, `diskcache`, `readability-lxml`, `lxml`, `pypdf`) are small, pure-Python-or-wheel, and the most common user workflow goes through at least one of them.
- The heavy ML deps (`torch`, `transformers`, `xgrammar`, `accelerate`) are genuinely optional — users who only ever call API-provider backends (future v0.2+) don't need them. And even today, a user who brings their own HF-loaded model + tokenizer could hypothetically skip the `hf` extra.

## Decision

Move `httpx`, `diskcache`, `readability-lxml`, `lxml`, and `pypdf` into main `dependencies`. Remove the now-redundant `render`, `pdf`, `url`, and `meta` extras. Keep:

(Originally `citeproc-py` was also promoted here, but the [home-grown render rewrite](004-citeproc-rewrite.md) removed it entirely; it's now in the opt-in `citeproc-compat` extra.)


- `hf` (torch + transformers + xgrammar + llguidance + accelerate) — heavy ML stack.
- `vllm` (Linux/CUDA only; see [ADR-006](006-vllm-excluded-from-all-extra.md)).
- `llamacpp` (the llama-cpp-python wheel).
- `verify` (NLI — lands in P6).
- `docs`, `examples`, `all`, `dev` — developer / user-convenience bundles.

## Consequences

- `pip install citeformer` is ~30 MB of deps (vs. the original ~10 MB of just pydantic/typer/rich). Tolerable trade for a working out-of-the-box render + metadata pipeline.
- Users who genuinely want the minimal base can't anymore. Feedback so far suggests no one wants that.
- CI install shrinks — `dev` extra stopped pulling in `hf`-level ML deps in P3, so `uv sync --extra dev` installs only docs + test tooling + main deps. Fast.
- When we do the citeproc-py rewrite ([ADR-004](004-citeproc-rewrite.md)), `citeproc-py` moves to a `citeproc-compat` optional extra. At that point main deps shrink again — the rewrite is self-contained.
