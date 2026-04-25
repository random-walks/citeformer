# ADR-014 — Async surface (`agenerate` / `astream`) on Backend + Citeformer

- **Status**: Accepted and implemented (2026-04-25).

## Context

LangChain (FastAPI middleware), LlamaIndex (`async_query`), and most
modern RAG framework code is async-first. Wrapping every citeformer
call in `asyncio.to_thread` works but is awkward — and worse, callers
can't compose multiple concurrent generations cleanly because the sync
SDK clients underneath block the executor thread for the whole request
duration.

Two ways to ship async:

1. **Add `async def` parallels alongside the sync surface** — the
   bigger codebase, but lets backends with native async clients
   (OpenAI's `AsyncOpenAI`, Anthropic's `AsyncAnthropic`, Mistral's
   `AsyncMistral`) actually scale concurrent calls without thread
   exhaustion.
2. **Provide a sync-only library and tell users to wrap with
   `asyncio.to_thread`** — smaller surface, fine for occasional
   calls, terrible for high-concurrency RAG pipelines.

The local backends (`HFBackend`, `VLLMBackend`, `LlamaCppBackend`) are
GPU-bound — there's no concurrency win from async there; they'd just
get the to_thread fallback. The API backends are where the real win
lives.

## Decision

Adopt option (1): add a parallel async surface end-to-end.

1. **`Backend` ABC** gains `agenerate()` and `astream()` with sensible
   defaults — both delegate to the sync methods via `asyncio.to_thread`,
   so every existing backend works in async code without modification.
   Backends with native async clients override for genuine concurrency.

2. **`Citeformer` orchestrator** gains `agenerate()` (async, returns
   `GenerationResult`) and `astream()` (sync call returning a new
   `AsyncStreamingResult` that is async-iterable + has `await
   stream.finalize()`). Same parsing / rendering / usage / rich-citation
   threading as the sync path — the orchestrator is mostly the same code
   awaiting the backend instead of calling it.

3. **Native overrides land in this PR for `OpenAIBackend` and
   `AnthropicBackend`** — the two most-used API backends. Both lazy-build
   their async client on first async call (so existing sync-only users
   don't pay the `AsyncOpenAI()` instantiation cost). The async client
   is constructed from the same `client_kwargs` the sync client used —
   one configuration, two clients.

4. **`OpenRouterBackend`, `FireworksBackend`, `TogetherBackend`
   inherit `OpenAIBackend.agenerate()` / `astream()` for free** — they
   subclass without touching the async path, so the cascade pulls them
   along.

5. **`GeminiBackend` and `MistralBackend` use the to_thread default**
   for now. Their SDKs both support async (`google-genai` async mode,
   `mistralai.AsyncMistral`) — flagged as a follow-up TODO in their
   module docstrings; not load-bearing because most users with
   high-concurrency needs reach for OpenAI/Anthropic anyway.

6. **Local backends (`HFBackend`, `VLLMBackend`, `LlamaCppBackend`)
   keep the to_thread default forever.** GPU-bound; concurrent
   generation contends for the same hardware. Async is for I/O
   concurrency, not compute concurrency.

## Consequences

- The `Backend` ABC grows two methods. Out-of-tree backends written
  against the v0.1 ABC still work — both new methods have concrete
  defaults, no abstract requirement.
- `AsyncStreamingResult` is a new public class symmetric with
  `StreamingResult`. Same surface (`text` property, `finalize()`,
  iterable) but async — `__aiter__` / `__anext__` instead of
  `__iter__` / `__next__`, and `finalize()` is `async def`.
- `OpenAIBackend.async_client` becomes a public attribute (lazy
  property) so subclasses and advanced users can introspect it.
- No §10 contract changed. The new methods produce identical
  `GenerationResult` shape to the sync methods.
- Adds zero new test-time dependencies. `pytest-asyncio` is already in
  the dev extras (was already used for the few existing async tests),
  and `asyncio_mode = "auto"` in `pyproject.toml` means async tests
  don't need decoration.

## What this does NOT include

- **Native async for Gemini / Mistral** — easy follow-ups, but
  separate PRs to keep this one focused.
- **Async metadata fetchers** (`fetch_crossref` etc.) — they're
  already cached and called once per source, low-leverage.
- **Async `verify()`** — NLI inference is GPU-bound; same
  argument as the local backends.

## Why not async-everywhere with a sync fallback?

Considered: drop the sync surface entirely, recommend
`asyncio.run(cf.agenerate(...))` for sync callers. Rejected because
- Most existing RAG code is sync; the breaking-change cost is large.
- `asyncio.run` in a sync function inside an event loop (jupyter,
  some test runners) explodes — sync callers expect a sync surface.
- The dual surface is ~80 lines of orchestrator + per-backend, much
  cheaper than the migration cost.

The sync surface is the right default; the async surface is the right
addition.
