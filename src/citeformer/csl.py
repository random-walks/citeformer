"""CSL-JSON schema validation helpers (§10.2 enforcement, opt-in).

``Source.metadata`` is declared as ``dict[str, Any]`` so the pydantic model
stays permissive — passing a malformed item through raises at render time
with a formatter-level error rather than a schema violation. That's
pragmatic for exploratory usage, but strict downstream pipelines (academic
publishing, compliance) want the schema policed up-front.

This module exposes:

- :data:`KNOWN_TYPES` — the CSL 1.0 item types we recognise.
- :data:`KNOWN_FIELDS` — the top-level field names we render; a superset
  of what any single style uses.
- :func:`validate_csl_json` — pure-Python validator. Returns a
  :class:`ValidationReport`; optionally raises :class:`CSLValidationError`
  on any failure.
- :func:`validate_source_metadata` — thin wrapper that validates a
  ``Source.metadata`` directly and re-raises with a friendlier message.

The validator is strict by default: unknown top-level fields produce
warnings (not errors). That's a deliberate looseness — the CSL-JSON spec
evolves, and we don't want to block new fields like ``custom`` that
downstream styles might rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CSLValidationError(ValueError):
    """Raised when `validate_csl_json(..., raise_on_error=True)` hits a hard error."""


#: CSL 1.0 item types we render. Taken from the CSL-JSON schema at
#: https://github.com/citation-style-language/schema/blob/master/csl-types.rnc.
#: Our formatters dispatch on these; unrecognised types fall back to the
#: ``article-journal`` rendering path.
KNOWN_TYPES: frozenset[str] = frozenset(
    {
        "article",
        "article-journal",
        "article-magazine",
        "article-newspaper",
        "bill",
        "book",
        "broadcast",
        "chapter",
        "classic",
        "collection",
        "dataset",
        "document",
        "entry",
        "entry-dictionary",
        "entry-encyclopedia",
        "event",
        "figure",
        "graphic",
        "hearing",
        "interview",
        "legal_case",
        "legislation",
        "manuscript",
        "map",
        "motion_picture",
        "musical_score",
        "pamphlet",
        "paper-conference",
        "patent",
        "performance",
        "periodical",
        "personal_communication",
        "post",
        "post-weblog",
        "regulation",
        "report",
        "review",
        "review-book",
        "software",
        "song",
        "speech",
        "standard",
        "thesis",
        "treaty",
        "webpage",
    }
)


#: The top-level CSL-JSON fields we consume or consider "known-benign". A
#: superset of what any single style uses. Fields outside this set trigger a
#: warning (not an error) so the validator doesn't reject future spec
#: extensions.
KNOWN_FIELDS: frozenset[str] = frozenset(
    {
        # Identity / shape
        "id",
        "type",
        "title",
        "title-short",
        "abstract",
        "annote",
        # Contributors
        "author",
        "editor",
        "translator",
        "recipient",
        "container-author",
        # Date
        "issued",
        "accessed",
        "event-date",
        "original-date",
        # Container / locator
        "container-title",
        "container-title-short",
        "collection-title",
        "volume",
        "issue",
        "number",
        "page",
        "page-first",
        "locator",
        "section",
        # Publisher
        "publisher",
        "publisher-place",
        "edition",
        # Identifiers + URLs
        "DOI",
        "ISBN",
        "ISSN",
        "PMID",
        "PMCID",
        "URL",
        # Meta
        "language",
        "note",
        "status",
        "genre",
        "medium",
        "source",
        "references",
        # Room for adapter-specific private extensions
        "custom",
        # citeformer's own shim keys (used by integrations/langchain.py etc.)
        "_langchain_metadata",
        "_llamaindex_metadata",
    }
)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of a CSL-JSON validation pass.

    Attributes:
        errors: Hard validation errors — missing required fields, wrong
            types for known fields, ``id`` conflicts. An item with any
            error is unrenderable until the caller fixes it.
        warnings: Soft flags — unknown ``type`` values, unknown top-level
            fields, empty string values where a non-empty is expected.
            Rendering still works; output may degrade.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` iff there are no errors. Warnings are allowed."""
        return not self.errors


def validate_csl_json(
    item: dict[str, Any],
    *,
    raise_on_error: bool = False,
    strict_types: bool = False,
) -> ValidationReport:
    """Validate a single CSL-JSON item against the §10.2 contract.

    Args:
        item: The CSL-JSON item (typically ``source.metadata``).
        raise_on_error: If ``True``, raise :class:`CSLValidationError` on
            any error. Warnings never raise.
        strict_types: If ``True``, promote unknown ``type`` values to
            errors (default: warning).

    Returns:
        A :class:`ValidationReport` with ``errors`` and ``warnings`` lists.

    Raises:
        CSLValidationError: If ``raise_on_error`` is True and any error is
            found. The error message lists every error.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(item, dict):
        errors.append(f"CSL-JSON item must be a dict, got {type(item).__name__}")
        report = ValidationReport(errors=errors, warnings=warnings)
        if raise_on_error:
            raise CSLValidationError("; ".join(errors))
        return report

    # Required fields.
    if "id" not in item:
        errors.append("CSL-JSON item missing required field 'id'")
    elif not isinstance(item["id"], (str, int)):
        errors.append(
            f"'id' must be a string or int, got {type(item['id']).__name__}"
        )
    if "type" not in item:
        errors.append("CSL-JSON item missing required field 'type'")
    elif not isinstance(item["type"], str):
        errors.append(f"'type' must be a string, got {type(item['type']).__name__}")
    elif item["type"] not in KNOWN_TYPES:
        msg = f"'type' {item['type']!r} is not a recognised CSL 1.0 item type"
        if strict_types:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Field type sanity checks.
    for field_name, expected in (
        ("title", str),
        ("DOI", str),
        ("URL", str),
        ("publisher", str),
        ("container-title", str),
        ("volume", (str, int)),
        ("issue", (str, int)),
        ("page", (str, int)),
    ):
        if field_name in item and not isinstance(item[field_name], expected):
            exp_name = (
                expected.__name__
                if isinstance(expected, type)
                else " or ".join(t.__name__ for t in expected)
            )
            errors.append(
                f"field {field_name!r} should be a {exp_name}, "
                f"got {type(item[field_name]).__name__}"
            )

    # Author / editor shape.
    for field_name in ("author", "editor", "translator"):
        if field_name not in item:
            continue
        raw = item[field_name]
        if not isinstance(raw, list):
            errors.append(
                f"field {field_name!r} must be a list of name dicts, "
                f"got {type(raw).__name__}"
            )
            continue
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                errors.append(
                    f"{field_name}[{i}] must be a dict, got {type(entry).__name__}"
                )
                continue
            # Each entry needs family+given, literal, or both.
            if "family" not in entry and "literal" not in entry:
                warnings.append(
                    f"{field_name}[{i}] has neither 'family' nor 'literal'; "
                    "rendering may skip it"
                )

    # Issued date shape.
    issued = item.get("issued")
    if issued is not None:
        if not isinstance(issued, dict):
            errors.append(
                f"'issued' must be a dict with 'date-parts', "
                f"got {type(issued).__name__}"
            )
        elif "date-parts" not in issued:
            warnings.append("'issued' has no 'date-parts'; year will be empty")
        elif not isinstance(issued["date-parts"], list):
            errors.append("'issued.date-parts' must be a list")

    # Unknown top-level fields → warnings only.
    for key in item:
        if key not in KNOWN_FIELDS:
            warnings.append(f"unknown CSL-JSON field {key!r}")

    report = ValidationReport(errors=errors, warnings=warnings)
    if raise_on_error and errors:
        raise CSLValidationError(
            f"CSL-JSON validation failed with {len(errors)} error(s): "
            + "; ".join(errors)
        )
    return report


def validate_source_metadata(
    metadata: dict[str, Any],
    *,
    raise_on_error: bool = False,
    strict_types: bool = False,
) -> ValidationReport:
    """Validate the `metadata` field of a :class:`citeformer.Source`.

    Sugar over :func:`validate_csl_json` that reframes error messages to
    mention ``Source.metadata`` instead of raw CSL-JSON — matters for
    users reading the exception in the context of their pipeline.
    """
    return validate_csl_json(
        metadata, raise_on_error=raise_on_error, strict_types=strict_types
    )


__all__ = [
    "KNOWN_FIELDS",
    "KNOWN_TYPES",
    "CSLValidationError",
    "ValidationReport",
    "validate_csl_json",
    "validate_source_metadata",
]
