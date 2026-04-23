"""Core types for citeformer.

Contains the §10.2 (`Source.metadata` CSL-JSON shape) and §10.3 (`GenerationResult`
output schema) contracts — both are pinned by snapshot tests in
`tests/integration/test_schemas.py`. Touching any of these models requires
the ceremony documented in `docs/reference/contracts.md`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from citeformer.verify.report import VerificationReport


class Policy(StrEnum):
    """Citation enforcement policy.

    - `REQUIRED`: every sentence must end with at least one citation (strictest; default).
    - `QUOTES_ONLY`: only quoted spans require a citation; narrative sentences can stand alone.
    - `AUTO`: citations are optional at every position; `verify()` surfaces missing citations
      via the coverage check instead of rejecting them at decode time.
    """

    REQUIRED = "required"
    QUOTES_ONLY = "quotes_only"
    AUTO = "auto"


class Source(BaseModel):
    """A piece of evidence made available to the model.

    Position in the `sources` list passed to `Citeformer.generate()` determines the
    citation index used by the model and echoed back in `Citation.source_id` and
    `Reference.source_id` — it is always 1-indexed.

    §10.2 contract: `metadata` must be a CSL-JSON item (the shape `citeproc-py`
    consumes). See https://github.com/citation-style-language/schema for the spec.

    Attributes:
        metadata: CSL-JSON item with at least `id`, `type`, and whatever fields the
            selected CSL style needs to render the entry (`author`, `title`, `issued`,
            `container-title`, `DOI`, `URL`, ...).
        content: Raw chunk text the model may cite from. Passed into the prompt; also
            used by `verify()` for NLI entailment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: dict[str, Any] = Field(
        description="CSL-JSON item consumed by citeproc-py.",
    )
    content: str = Field(
        description="Raw text the model may cite from.",
    )


class Citation(BaseModel):
    """A single inline citation marker emitted by the model.

    Attributes:
        span: `(start, end)` character offsets of the marker inside
            `GenerationResult.text`.
        source_id: 1-indexed position of the cited source inside the `sources` list
            that was passed to `Citeformer.generate()`.
        verified: Populated by `GenerationResult.verify()`; `False` until then. `True`
            iff the cited source entails the citing claim with score above threshold.
        entailment_score: Populated by `GenerationResult.verify()`; `None` until then.
            Value in [0, 1] indicating NLI entailment confidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    span: tuple[int, int] = Field(
        description="(start, end) char offsets of the marker inside GenerationResult.text.",
    )
    source_id: int = Field(
        ge=1,
        description="1-indexed position of the cited source in the sources list.",
    )
    verified: bool = Field(
        default=False,
        description="True iff verify() found the claim entailed by the source.",
    )
    entailment_score: float | None = Field(
        default=None,
        description="NLI entailment probability in [0, 1]; set by verify().",
    )


class Reference(BaseModel):
    """A rendered bibliography entry paired with its inline marker.

    Every cited `source_id` has exactly one `Reference` in `GenerationResult.references`.
    Rendering is deterministic via `citeproc-py` — **the model never touches this**.

    Attributes:
        source_id: The 1-indexed source this reference describes. Matches the
            `source_id` of every `Citation` that points at this reference.
        inline_marker: How the marker appears in prose. For numeric styles this is
            `"[1]"`; for author-year styles `"(Poe 1845)"`; for footnote styles
            `"¹"`. The renderer chooses based on the selected CSL style.
        rendered: Full bibliography entry, rendered by `citeproc-py` in the chosen
            CSL style. E.g. `"Poe, E. A. (1845). The Raven. ..."`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: int = Field(
        ge=1,
        description="1-indexed source this reference corresponds to.",
    )
    inline_marker: str = Field(
        description="How the marker appears in prose, e.g. '[1]' or '(Poe 1845)'.",
    )
    rendered: str = Field(
        description="Full bibliography entry, rendered by citeproc-py.",
    )


class GenerationResult(BaseModel):
    """Full output of a `Citeformer.generate()` call.

    §10.3 contract: `schema_version` is pinned by `tests/integration/test_schemas.py`.
    Any shape change requires bumping `schema_version` and following the ceremony in
    `docs/reference/contracts.md`.

    Attributes:
        schema_version: Contract version. Bump on any field add/rename/removal.
        text: The generated prose with inline `[N]` markers.
        citations: One entry per `[N]` marker, with its char span and `source_id`.
        references: Deterministically rendered bibliography, one entry per unique
            cited `source_id`. Rendered by `citeproc-py` (P3+) — never by the LLM.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(
        default=1,
        description="§10.3 contract version. Bumped on any shape change.",
    )
    text: str = Field(
        description="Generated prose with inline citation markers.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="One entry per [N] marker in text.",
    )
    references: list[Reference] = Field(
        default_factory=list,
        description="Deterministically rendered references, one per unique cited source_id.",
    )

    def verify(self, **_options: Any) -> VerificationReport:
        """Run NLI-based verification against the cited sources.

        Populated in P6. Until then, raises `NotImplementedError` so callers who
        wire up the full flow get a clear error rather than silently-incorrect stubs.

        Returns:
            A `VerificationReport` with per-citation entailment scores, an overall
            support rate, and uncited-but-entailed flags for missing citations.

        Raises:
            NotImplementedError: Always, in P1–P5. Landing in P6.
        """
        raise NotImplementedError(
            "GenerationResult.verify() lands in P6 (NLI verification). "
            "Track progress at https://github.com/random-walks/citeformer."
        )
