"""Crossref DOI metadata fetcher.

https://api.crossref.org/works/{doi} with ``Accept:
application/vnd.citationstyles.csl+json`` returns a CSL-JSON dict directly,
saving us a translation hop. The returned dict is suitable for handing
straight to ``citeformer.render.render_references`` as ``Source.metadata``.

Polite-pool opt-in: if the ``CITEFORMER_CROSSREF_MAILTO`` env var is set,
the outgoing User-Agent carries it so Crossref routes the request through
the polite pool (faster, more reliable rate limits). See
https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-rest-api/
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from citeformer._version import __version__
from citeformer.metadata.cache import get_metadata_cache

# `/transform` is the content-negotiation endpoint — it returns the metadata
# directly (not wrapped in Crossref's `{status, message-type, message}`
# envelope) when asked for CSL-JSON. Bare `/works/{doi}` rejects the CSL
# Accept header with a 406.
_CROSSREF_URL = "https://api.crossref.org/works/{doi}/transform"
_DEFAULT_TIMEOUT = 30.0


def _user_agent() -> str:
    base = f"citeformer/{__version__} (+https://github.com/random-walks/citeformer)"
    mailto = os.environ.get("CITEFORMER_CROSSREF_MAILTO")
    if mailto:
        return f"{base}; mailto:{mailto}"
    return base


def _normalize_doi(doi: str) -> str:
    doi = doi.strip()
    # Accept several common input forms.
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi


def fetch_crossref(
    doi: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Fetch a Crossref CSL-JSON entry for a DOI.

    Args:
        doi: DOI string. Accepts bare (``"10.1038/s41586-023-06221-2"``),
            URL (``"https://doi.org/…"``), or ``"doi:…"`` prefixed forms.
        timeout: HTTP timeout in seconds.
        use_cache: Cache the result under ``~/.cache/citeformer/metadata/``.

    Returns:
        CSL-JSON item dict containing ``id``, ``type``, ``author``, ``title``,
        ``container-title``, ``issued``, ``DOI``, ``URL``, and other fields
        Crossref has on file for this DOI.

    Raises:
        httpx.HTTPStatusError: If Crossref returns a non-2xx status.
    """
    doi = _normalize_doi(doi)

    cache_key = f"crossref:{doi}"
    if use_cache:
        cache = get_metadata_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

    url = _CROSSREF_URL.format(doi=doi)
    headers = {
        "Accept": "application/vnd.citationstyles.csl+json",
        "User-Agent": _user_agent(),
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    if use_cache:
        get_metadata_cache().set(cache_key, data)

    return data
