# ADR-012 — `GenerationResult` bumped to schema_version=3 with optional `usage`

- **Status**: Accepted and implemented (2026-04-25).

## Context

Every API backend (`OpenAIBackend`, `AnthropicBackend`, `GeminiBackend`,
`MistralBackend`, plus the new `OpenRouterBackend`) receives a per-call
`usage` payload from the provider — input tokens, output tokens, and
(for Anthropic) cache-creation / cache-read tokens, plus (for OpenRouter)
the per-call USD cost. citeformer was discarding all of it. Users running
real RAG pipelines on API backends had no way to get token / cost
visibility without poking around in private SDK objects.

The natural home is `GenerationResult`. It's already the canonical typed
output of `Citeformer.generate()`. Adding a `usage: TokenUsage | None`
field makes the data accessible without changing any other shape; local
backends leave it `None` (they don't bill per token).

## Decision

1. Introduce `citeformer.core.TokenUsage` — a frozen pydantic model with
   `input_tokens`, `output_tokens`, optional `cache_creation_input_tokens`,
   `cache_read_input_tokens`, and `cost_usd` fields. Top-level export
   alongside the other types.
2. Add `GenerationResult.usage: TokenUsage | None = None`.
3. Bump `GenerationResult.schema_version` from **2 → 3**.
4. Each API backend exposes a `last_usage: TokenUsage | None` instance
   attribute populated at the end of every `generate()` /
   `stream()` call. The `Citeformer` orchestrator reads it via
   `getattr(backend, "last_usage", None)` and threads it onto
   `result.usage`.

The `getattr` (rather than a typed `Backend.last_usage` property on the
ABC) keeps local backends from having to implement a no-op getter and
keeps the `Backend` ABC unchanged — useful since out-of-tree backends
people have written against the v0.1 ABC keep working untouched.

## Consequences

- `GenerationResult` snapshot in `tests/integration/test_schemas.py`
  regenerated. The schema_version assertion was bumped from 2 to 3.
- Pre-bump serialised results (schema_version=2) deserialise cleanly into
  the new model — `usage` defaults to `None`. No migration shim needed.
- Users hand-constructing `GenerationResult` (custom backends, tests,
  demo scripts) get `usage=None` for free; they can opt-in to filling
  it themselves when they care.
- Backends that newly populate `last_usage`:
  - `OpenAIBackend` — from `completion.usage.{prompt,completion}_tokens`.
  - `AnthropicBackend` — from `message.usage.{input,output,cache_creation_input,cache_read_input}_tokens`.
  - `GeminiBackend` — from `response.usage_metadata.{prompt,candidates}_token_count`.
  - `MistralBackend` — from `response.usage.{prompt,completion}_tokens`.
  - `OpenRouterBackend` — same as OpenAI plus `usage.cost` (USD).
- Local backends (`HFBackend`, `VLLMBackend`, `LlamaCppBackend`,
  `MockBackend`) do not set `last_usage`; the orchestrator surfaces
  `usage = None` for those.
- Non-`Citeformer` consumers calling `Backend.generate()` directly still
  see `str` returned — they pull `backend.last_usage` themselves if they
  want it.
- CHANGELOG documents the bump under `Contracts (§10)`.
