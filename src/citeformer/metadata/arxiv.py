"""arXiv metadata fetcher.

arXiv's export API returns Atom XML, not JSON, so we parse it ourselves and
translate to CSL-JSON. The abstract is returned as the ``abstract`` field
(which `Source.from_arxiv` pops into ``content``).
"""

from __future__ import annotations

import os
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from citeformer._version import __version__
from citeformer.metadata.cache import get_metadata_cache

_ARXIV_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_DEFAULT_TIMEOUT = 30.0


def _user_agent() -> str:
    base = f"citeformer/{__version__} (+https://github.com/random-walks/citeformer)"
    mailto = os.environ.get("CITEFORMER_CROSSREF_MAILTO")
    # arXiv doesn't have a formal polite pool but they do ask for a contact
    # email in the UA. Reuse the Crossref env var rather than demand a
    # separate one.
    if mailto:
        return f"{base}; mailto:{mailto}"
    return base


def _normalize_arxiv_id(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip()
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv:", "arXiv:"):
        if arxiv_id.startswith(prefix):
            arxiv_id = arxiv_id[len(prefix) :]
            break
    # Drop version suffix (e.g. "2305.14627v3" → "2305.14627") so caches and
    # CSL ids don't fragment by revision.
    if "v" in arxiv_id:
        bare, sep, tail = arxiv_id.partition("v")
        if sep and tail.isdigit():
            arxiv_id = bare
    return arxiv_id


def fetch_arxiv(
    arxiv_id: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Fetch CSL-JSON metadata for an arXiv paper.

    Args:
        arxiv_id: arXiv identifier (e.g. ``"2305.14627"``). Accepts URL,
            ``arxiv:``, and versioned (``"2305.14627v2"``) forms; version
            suffix is stripped.
        timeout: HTTP timeout in seconds.
        use_cache: Cache the CSL-JSON under ``~/.cache/citeformer/metadata/``.

    Returns:
        CSL-JSON item dict with an extra ``abstract`` key carrying the paper
        abstract (useful as ``Source.content``).

    Raises:
        ValueError: If arXiv returns no entry for the id.
        httpx.HTTPStatusError: On HTTP errors.
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id)

    cache_key = f"arxiv:{arxiv_id}"
    if use_cache:
        cache = get_metadata_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

    params = {"id_list": arxiv_id, "max_results": "1"}
    headers = {"User-Agent": _user_agent()}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(_ARXIV_URL, params=params, headers=headers)
        response.raise_for_status()
        atom_xml = response.text

    csl = _parse_arxiv_atom(atom_xml, arxiv_id)
    if use_cache:
        get_metadata_cache().set(cache_key, csl)
    return csl


def _parse_arxiv_atom(xml: str, arxiv_id: str) -> dict[str, Any]:
    """Convert an arXiv Atom response into CSL-JSON."""
    root = ET.fromstring(xml)
    entry = root.find(f"{_ATOM_NS}entry")
    if entry is None:
        raise ValueError(f"arXiv returned no entry for id {arxiv_id!r}")

    title_el = entry.find(f"{_ATOM_NS}title")
    title = (title_el.text or "").strip() if title_el is not None else ""

    authors: list[dict[str, str]] = []
    for author_el in entry.findall(f"{_ATOM_NS}author"):
        name_el = author_el.find(f"{_ATOM_NS}name")
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        # Simple heuristic: last whitespace-separated token is `family`; the
        # rest is `given`. Works for Western-style names; for others the CSL
        # `literal` form is more correct but we can't tell without more info.
        parts = name.rsplit(" ", 1)
        if len(parts) == 2 and parts[1]:
            authors.append({"given": parts[0], "family": parts[1]})
        else:
            authors.append({"literal": name})

    published_el = entry.find(f"{_ATOM_NS}published")
    year: int | None = None
    if published_el is not None and published_el.text:
        try:
            year = int(published_el.text[:4])
        except ValueError:
            year = None

    summary_el = entry.find(f"{_ATOM_NS}summary")
    abstract = (summary_el.text or "").strip() if summary_el is not None else ""

    csl: dict[str, Any] = {
        "id": f"arxiv-{arxiv_id}",
        "type": "article-journal",
        "author": authors,
        "title": title,
        "container-title": "arXiv preprint",
        "note": f"arXiv:{arxiv_id}",
        "URL": f"https://arxiv.org/abs/{arxiv_id}",
        "abstract": abstract,
    }
    if year is not None:
        csl["issued"] = {"date-parts": [[year]]}
    return csl
