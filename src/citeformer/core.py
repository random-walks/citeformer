"""Core types for citeformer (populated in P1).

This module will hold:

- `Source` — an LLM-facing piece of evidence: id, CSL-JSON metadata, raw content.
- `Citation` — a span of generated text tagged with its cited source id.
- `Reference` — a rendered bibliography entry paired with its inline marker.
- `GenerationResult` — full output of a generation call (pydantic, `schema_version: 1`).
- `Policy` — enum of citation policies: `required`, `quotes_only`, `auto`.

The `schema_version: 1` on `GenerationResult` and `VerificationReport` is a §10 contract
(see docs/reference/contracts.md). Breaking changes to the shape bump to a major release.
"""

from __future__ import annotations
