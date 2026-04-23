"""Core types for citeformer.

Contains the §10.2 (`Source.metadata` CSL-JSON shape) and §10.3 (`GenerationResult`
output schema) contracts — both are pinned by snapshot tests in
`tests/integration/test_schemas.py`. Touching any of these models requires
the ceremony documented in `docs/reference/contracts.md`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

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

    @classmethod
    def from_doi(cls, doi: str, **kwargs: Any) -> Self:
        """Build a `Source` from a Crossref DOI lookup.

        The returned `content` field is empty — DOI metadata alone doesn't
        ship the paper text. If you have the PDF, use `Source.from_pdf` to
        get the text and merge with `metadata=source.metadata | pdf_meta`,
        or construct the combined `Source` directly.

        Args:
            doi: DOI in bare, URL, or ``doi:`` form.
            **kwargs: Forwarded to `citeformer.metadata.fetch_crossref`
                (``timeout``, ``use_cache``).

        Returns:
            A `Source` with `metadata` = CSL-JSON from Crossref and empty
            `content`.
        """
        from citeformer.metadata import fetch_crossref

        metadata = fetch_crossref(doi, **kwargs)
        return cls(metadata=metadata, content="")

    @classmethod
    def from_arxiv(cls, arxiv_id: str, **kwargs: Any) -> Self:
        """Build a `Source` from an arXiv API lookup.

        The abstract becomes `content` so the model has something concrete
        to cite. For the full paper body, fetch the PDF and use
        `Source.from_pdf` separately.

        Args:
            arxiv_id: arXiv id (bare, URL, or ``arxiv:`` form; version
                suffix is stripped).
            **kwargs: Forwarded to `citeformer.metadata.fetch_arxiv`.

        Returns:
            A `Source` with the arXiv CSL-JSON and the abstract in `content`.
        """
        from citeformer.metadata import fetch_arxiv

        metadata = dict(fetch_arxiv(arxiv_id, **kwargs))
        # abstract lives in the fetcher output but doesn't belong in CSL
        # metadata — pull it into content.
        abstract = str(metadata.pop("abstract", ""))
        return cls(metadata=metadata, content=abstract)

    @classmethod
    def from_pdf(cls, path: str | Any, **kwargs: Any) -> Self:
        """Build a `Source` from a local PDF via pypdf.

        Args:
            path: Filesystem path to the PDF.
            **kwargs: Reserved for future fetcher options (none currently).

        Returns:
            A `Source` with best-effort CSL metadata (``title``, ``author``,
            ``issued`` when the PDF info dict has them) and the concatenated
            page text as `content`.
        """
        from citeformer.metadata import extract_pdf

        metadata, content = extract_pdf(path, **kwargs)
        return cls(metadata=metadata, content=content)

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> Self:
        """Build a `Source` from an HTTP(S) URL.

        Uses readability-lxml for the article body and meta-tag parsing
        (OpenGraph / Twitter / article) for title / author / date / site.

        Args:
            url: HTTP(S) URL.
            **kwargs: Forwarded to `citeformer.metadata.extract_url`.

        Returns:
            A `Source` with webpage CSL metadata and the article body in
            `content`.
        """
        from citeformer.metadata import extract_url

        metadata, content = extract_url(url, **kwargs)
        return cls(metadata=metadata, content=content)


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
    `docs/reference/contracts.md`. Current version: **2** (P6 added `sources`
    so `verify()` is self-contained).

    Attributes:
        schema_version: Contract version. Bump on any field add/rename/removal.
        text: The generated prose with inline `[N]` markers.
        citations: One entry per `[N]` marker, with its char span and `source_id`.
        references: Deterministically rendered bibliography, one entry per unique
            cited `source_id`. Rendered by the `citeformer.render` formatters —
            never by the LLM.
        sources: The sources that were in scope for this generation call. Carried
            on the result so `verify()` can run NLI against them without the
            caller having to pass them separately.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(
        default=2,
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
    sources: list[Source] = Field(
        default_factory=list,
        description="The sources that were in scope when this result was generated.",
    )

    def verify(
        self,
        *,
        threshold: float = 0.5,
        nli: Any | None = None,
        run_coverage: bool = True,
        **_options: Any,
    ) -> VerificationReport:
        """Run NLI-based verification against the cited sources.

        Requires the ``verify`` extra (``pip install citeformer[verify]``) —
        the NLI backend is imported lazily on first call.

        Args:
            threshold: Entailment probability above which a citation is
                ``supported`` and an uncited sentence is flagged as needing a
                citation.
            nli: Optional pre-constructed `citeformer.verify.NLIModel`. If
                ``None``, the default model (DeBERTa-v3-large-MNLI, or whatever
                ``CITEFORMER_NLI_MODEL`` is set to) is loaded on first use and
                cached.
            run_coverage: If False, skip the NLI coverage check (per-sentence
                "should this have been cited?" scan). Useful under REQUIRED
                policy where the grammar guarantees every sentence has a cite.

        Returns:
            A `VerificationReport` with per-citation entailment scores, an
            overall support rate, and uncited-but-entailed flags.

        Raises:
            ImportError: If ``citeformer[verify]`` extras aren't installed.
            ValueError: If this result was constructed without `sources` (e.g.
                a pre-P6 serialization that predates the schema_version=2
                shape).
        """
        from citeformer.verify import Verifier

        if not self.sources:
            raise ValueError(
                "GenerationResult.verify() needs `sources` populated. "
                "Results from Citeformer.generate() carry this automatically; "
                "hand-constructed results must pass `sources=...` at build time."
            )

        verifier = Verifier(threshold=threshold, nli=nli)
        return verifier.verify(
            text=self.text,
            citations=self.citations,
            sources=self.sources,
            run_coverage=run_coverage,
        )
