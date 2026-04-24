---
description: Audit a diff against citeformer's three §10 contracts (grammar, CSL metadata, output schemas).
---

Review the pending changes (compared to `main`) for §10 contract compliance. Delegates the heavy lifting to the `contract-reviewer` agent.

## Steps

1. Run `git diff main...HEAD` (or `git diff --cached` if the user specified pre-commit).
2. Launch the `contract-reviewer` agent with the diff as context. Ask it to classify each contract touch as Untouched / Additive / Breaking and to verify the required ceremony happened.
3. Surface the agent's report verbatim to the user.
4. If the report says BLOCKING on any contract, do not suggest moving to `/bump` yet. If it's all OK / additive, print the recommended bump level (patch / minor / major) the agent concluded.

## What counts as a contract touch

- §10.1 — any file under `src/citeformer/grammar/`, or the `Policy` enum in `src/citeformer/core.py`.
- §10.2 — the CSL-JSON consumer in `src/citeformer/render/csl.py` or the `Source.metadata` type annotation in `src/citeformer/core.py`.
- §10.3 — `GenerationResult` or `VerificationReport` pydantic models (search for `schema_version` and nearby fields).

See [docs/reference/contracts.md](docs/reference/contracts.md) for the full ceremony.
