"""CSL style loader + alias table.

Bundled styles are pinned to commit
``77bcd6d4052c606156de90f9f48405250b64a4ea`` of
https://github.com/citation-style-language/styles (2026-04-23). Refreshing
them is a deliberate act — bump the pin in ``docs/reference/architecture.md``
and in the comment at the top of this file, regenerate the ``render_snapshot_*``
tests, and note it in ``CHANGELOG.md``. The §10.2 contract pins the *metadata
shape* that styles consume, not the style files themselves — so style updates
are usually minor-level unless they change rendering observably.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citeproc.frontend import CitationStylesStyle


# Bundled CSL styles → filename under `src/citeformer/render/styles/`.
# Aliases map to the same file (e.g. "apa" and "apa-7" both → apa.csl since
# the bundled APA is APA 7th edition).
_BUNDLED_FILENAMES: dict[str, str] = {
    "apa": "apa.csl",
    "apa-7": "apa.csl",
    "mla": "modern-language-association.csl",
    "mla-9": "modern-language-association.csl",
    "chicago-author-date": "chicago-author-date.csl",
    "chicago": "chicago-author-date.csl",
    "ieee": "ieee.csl",
    "nature": "nature.csl",
}


# Canonical / primary names (without aliases). Order matters for docs + tests.
_CANONICAL_NAMES: tuple[str, ...] = (
    "apa-7",
    "mla-9",
    "chicago-author-date",
    "ieee",
    "nature",
)


def bundled_style_names() -> list[str]:
    """Return the canonical list of bundled style identifiers.

    Aliases (e.g. ``"apa"`` → ``"apa-7"``) are omitted. Use `resolve_style_path`
    or `load_style` with any name, alias or canonical, to get the file.
    """
    return list(_CANONICAL_NAMES)


def resolve_style_path(name_or_path: str) -> Path:
    """Resolve a bundled style name or filesystem path to an absolute CSL path.

    Args:
        name_or_path: Either a bundled name (``"apa-7"``, ``"ieee"``, …) or
            an absolute / relative filesystem path to a ``.csl`` file.

    Returns:
        Absolute path to the CSL file.

    Raises:
        FileNotFoundError: If the path doesn't exist and the name isn't bundled.
    """
    # Bundled names win over filesystem collisions — no way to shadow a
    # bundled style by creating a file with the same bare name in cwd.
    if name_or_path in _BUNDLED_FILENAMES:
        filename = _BUNDLED_FILENAMES[name_or_path]
        resource = files("citeformer.render").joinpath(filename)
        path = Path(str(resource))
        if not path.exists():  # pragma: no cover — sanity check
            raise FileNotFoundError(
                f"Bundled style {name_or_path!r} resolves to {path!s} but the "
                "file is missing. This is a packaging bug — re-install citeformer."
            )
        return path

    path = Path(name_or_path).expanduser().resolve()
    if path.exists():
        return path

    raise FileNotFoundError(
        f"Style {name_or_path!r} not found. "
        f"Bundled names: {bundled_style_names()}. "
        "Or provide an absolute path to a .csl file."
    )


@lru_cache(maxsize=32)
def load_style(name_or_path: str) -> CitationStylesStyle:
    """Load and cache a CSL style.

    Style parsing is fast (~0.2ms) but not free; we cache because a single
    citeformer instance may render many `generate()` calls against the same
    style. Cache is keyed by the input string, so ``"apa-7"`` and ``"apa"``
    cache separately even though they resolve to the same file — acceptable.

    Args:
        name_or_path: Bundled name or filesystem path.

    Returns:
        A parsed `CitationStylesStyle` ready for `CitationStylesBibliography`.
    """
    from citeproc.frontend import CitationStylesStyle

    path = resolve_style_path(name_or_path)
    return CitationStylesStyle(str(path), locale="en-US", validate=False)


def style_citation_format(name_or_path: str) -> str:
    """Classify a style's inline-citation format without rendering.

    Inspects the CSL XML ``<info><category citation-format="…">`` attribute.

    Args:
        name_or_path: Bundled name or filesystem path.

    Returns:
        One of ``"author-date"``, ``"author"`` (MLA-style author + locator),
        ``"numeric"``, ``"note"``, ``"label"``, or ``"unknown"`` if the style
        doesn't declare a category.
    """
    from lxml import etree

    path = resolve_style_path(name_or_path)
    tree = etree.parse(str(path))
    ns = {"cs": "http://purl.org/net/xbiblio/csl"}
    category = tree.find(".//cs:info/cs:category[@citation-format]", namespaces=ns)
    if category is None:
        return "unknown"
    fmt: str = category.get("citation-format", "unknown")
    return fmt
