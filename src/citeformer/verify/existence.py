"""Existence check — do cited IDs resolve to real sources?

Under Tier 1 (grammar-enforced backends, all current local backends) this
check is trivially always `True` because the decoder cannot produce a
citation to an out-of-range id. It's still valuable to run:

- Backwards-compatibility: a future API-provider backend (schema-level
  enforcement rather than grammar-level) may let an ID slip through.
- Robustness: if a user hand-constructs a `GenerationResult` or swaps
  backends, existence becomes a non-trivial assertion.
- Diagnostics: citeformer-internal bugs that emit malformed citations
  surface here.

No ML required; pure set arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from citeformer.core import Citation, Source


@dataclass(frozen=True)
class ExistenceResult:
    """Outcome of the existence check.

    Attributes:
        all_exist: True iff every `Citation.source_id` is in ``[1, len(sources)]``.
        missing: The source_ids that were emitted but don't resolve — empty
            if `all_exist` is True.
    """

    all_exist: bool
    missing: tuple[int, ...]


def check_existence(
    citations: list[Citation],
    sources: list[Source],
) -> ExistenceResult:
    """Verify every citation's `source_id` points to a real source.

    Args:
        citations: The citations emitted in generation.
        sources: The sources that were in scope for the generation call.

    Returns:
        An `ExistenceResult` carrying the verdict + any missing ids.
    """
    max_id = len(sources)
    missing_ids: set[int] = set()
    for c in citations:
        if not (1 <= c.source_id <= max_id):
            missing_ids.add(c.source_id)
    return ExistenceResult(
        all_exist=len(missing_ids) == 0,
        missing=tuple(sorted(missing_ids)),
    )
