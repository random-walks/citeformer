# ADR-017 — Defer per-model cost tables; surface tokens, let consumers price

- **Status**: Decided (2026-04-25); not planned.

## Context

`TokenUsage.cost_credits` (added in ADR-012) is populated only when the
provider returns a per-call cost directly. Today that's only OpenRouter
(`usage.cost`, in OpenRouter credits). For OpenAI / Anthropic / Gemini /
Mistral / Fireworks / Together, `cost_credits` stays `None` — consumers
have to multiply tokens by their own pricing knowledge to get a dollar
figure.

A natural-feeling addition would be a **per-model pricing table** built
into citeformer — `PRICING["gpt-4o-mini"] = {"input": 0.15e-6, "output":
0.60e-6}` style — so we could compute and surface `cost_credits` (in
inferred USD) for every backend.

## Decision

**Don't ship a pricing table.** Reasons:

1. **Pricing changes constantly.** OpenAI alone re-priced gpt-4o twice
   in 2024-2025 and added/deprecated tiers (cached input, batch
   discount, structured-output surcharge). A baked-in table would be
   stale within months and create a maintenance burden disproportionate
   to its value.
2. **The data is one multiply away from `usage.input_tokens` and
   `usage.output_tokens`.** Consumers who care about cost already know
   their pricing (it's on their bill). A library shipping inferred USD
   pricing is just a convenience that's wrong half the time and right
   the other half.
3. **OpenRouter solved it correctly by reporting cost server-side.**
   The right shape is the provider returning the cost, not the client
   guessing it. We should expose `cost_credits` when the provider
   gives us one, and stay silent otherwise.
4. **Cached / batch pricing tiers compound the error.** Anthropic's
   `cache_creation_input_tokens` / `cache_read_input_tokens` are
   priced 1.25× and 0.1× of fresh input respectively. OpenAI's batch
   API is 50% off. A pricing table that doesn't account for these
   would systematically over- or under-count.

## What we ship instead

- **Token counts on `TokenUsage`** — accurate, provider-reported,
  composes with whatever pricing model the consumer wants.
- **`cost_credits` populated when the provider returns it.** Today
  that's OpenRouter's `usage.cost` field. As other providers add
  cost-in-response (Anthropic announced this is coming; Bedrock's
  CloudWatch metrics already expose it) we extract it and surface
  it on the same field.
- **Documentation** in the `TokenUsage` docstring noting that
  consumers needing dollar figures should consult the provider's
  pricing page or use a third-party calculator (`tokencost`, `tokonomy`,
  etc.) rather than expecting a table from us.

## When we'd reconsider

- **A vendor-neutral pricing source** with a stable API and refresh
  guarantee (LiteLLM ships one; if it stays maintained and accurate,
  pulling it in as an optional integration could be reasonable —
  separate PR).
- **A specific user requirement** for in-library pricing that can't
  be met by computing from token counts.

Until then: tokens in, tokens out, cost when the provider cares to
tell us.

## Consequences

- `TokenUsage` stays as defined in ADR-012. No new fields.
- No `pricing.py` module, no per-model dict.
- `cost_credits` docstring already notes the unit and that "OpenRouter
  is the only provider exposing this today."
- A `pricing` extra is intentionally NOT created.
