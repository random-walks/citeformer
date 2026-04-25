# ADR-015 — Defer Bedrock + Vertex AI backends to a future release

- **Status**: Decided (2026-04-25); deferred to v0.4+ (no committed timeline).

## Context

AWS Bedrock and Google Vertex AI are the two biggest cloud-native LLM
proxies — they front the same models we already support directly
(Claude family on Bedrock, Gemini family on Vertex) but with cloud
account billing, IAM/SigV4 auth, regional endpoints, and enterprise
controls. Adding both would round out the "if you have the models, we
support them" story.

The PR description for the OpenRouter / Fireworks / Together /
Anthropic-revamp branch listed both as deferred. This ADR makes the
defer explicit so the next contributor doesn't quietly pick them up
without weighing the tradeoffs.

## Decision

**Defer both to a future release.** Reasons:

1. **Auth complexity is non-trivial.** Bedrock requires AWS SigV4
   request signing (typically via `boto3` or `aws-sigv4-requests`).
   Vertex requires GCP service-account credentials and project/region
   configuration. Neither slots into the existing `client_kwargs +
   API_KEY env var` shape — both need a real auth integration. Bedrock
   alone would add a 2-3 MB `boto3` dependency footprint.
2. **Both proxy to providers we already support directly.** Bedrock's
   Claude models work via `AnthropicBackend` if you point it at
   `bedrock-runtime.us-east-1.amazonaws.com` — the message shape is
   identical. Vertex's Gemini works via `GeminiBackend` with
   `vertexai=True` already. The value-add of dedicated backends is
   mostly enterprise auth + cloud billing — both orthogonal to
   citeformer's main use case (verifiable RAG citations).
3. **Bedrock's strict structured-outputs story is murky.** As of
   April 2026, AWS Bedrock proxies the Anthropic Citations API
   transparently, so Claude-via-Bedrock would work for our needs. But
   for non-Anthropic models on Bedrock (Llama, Titan), it's unclear
   whether strict mode passes through cleanly — would require a
   per-model capability matrix to maintain.
4. **Vertex's `responseSchema` is mostly a copy of the Gemini API
   surface.** A `VertexGeminiBackend` would be ~80% duplicated code
   with `GeminiBackend`, distinguished mainly by the auth path.

## When we'd revisit

Concrete signals that would justify the work:

- **A real user request** for either backend with a specific use case
  (enterprise RAG that can't leave AWS / GCP).
- **A non-Anthropic Bedrock model** demonstrating strict-structured-
  output support that we couldn't reach from the existing backends.
- **Vertex AI exposing capabilities the public Gemini API doesn't**
  (e.g., higher rate limits or larger context windows that warrant a
  separate backend with different defaults).

In the meantime, users with a hard requirement can today:

- Point `AnthropicBackend(client=anthropic.AnthropicBedrock(...))` at
  Bedrock — the `AnthropicBedrock` client from the official SDK is
  drop-in compatible.
- Point `GeminiBackend(client=genai.Client(vertexai=True, project="…",
  location="…"))` at Vertex — the `google-genai` SDK supports both
  with the same client object.

We document both paths in the architecture doc as "supported via
existing backends" rather than as separate citeformer backends.

## Consequences

- Architecture doc gains a "Cloud-proxy support" subsection pointing
  to the in-existing-backend paths above.
- No `boto3` / Vertex-specific deps in `pyproject.toml`.
- Future Bedrock / Vertex backend ADRs (016+) will cite this one if
  they revisit.
