"""URL metadata + content extraction.

``readability-lxml`` finds the article body; raw ``lxml`` pulls OpenGraph /
Twitter / article meta tags for title / author / date / site-name. The
returned CSL-JSON uses ``type: "webpage"`` — users who know the content is
actually an article or paper should override after calling ``from_url``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from lxml import html as lxml_html
from readability import Document

from citeformer._version import __version__

_DEFAULT_TIMEOUT = 30.0


def _user_agent() -> str:
    return f"citeformer/{__version__} (+https://github.com/random-walks/citeformer)"


def _meta_property(root: Any, prop: str) -> str | None:
    results = root.xpath(f'//meta[@property="{prop}"]/@content')
    value = str(results[0]) if results else None
    return value.strip() if value else None


def _meta_name(root: Any, name: str) -> str | None:
    results = root.xpath(f'//meta[@name="{name}"]/@content')
    value = str(results[0]) if results else None
    return value.strip() if value else None


def _first_nonempty(*candidates: str | None) -> str | None:
    for c in candidates:
        if c:
            return c
    return None


def _strip_html(html_str: str) -> str:
    """Convert an HTML fragment (from readability) to plain text."""
    if not html_str.strip():
        return ""
    root = lxml_html.fromstring(html_str)
    return str(root.text_content()).strip()


def _parse_published(raw: str) -> dict[str, Any] | None:
    """Parse a published date into CSL ``issued.date-parts`` if possible."""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        # Many sites publish ``YYYY-MM-DDTHH:MM:SSZ`` or just ``YYYY-MM-DD``;
        # try a bare 4-digit year as last resort.
        if len(raw) >= 4 and raw[:4].isdigit():
            try:
                return {"date-parts": [[int(raw[:4])]]}
            except ValueError:
                return None
        return None
    return {"date-parts": [[dt.year, dt.month, dt.day]]}


def extract_url(url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> tuple[dict[str, Any], str]:
    """Fetch a URL and extract CSL-JSON metadata + article text.

    Args:
        url: HTTP(S) URL.
        timeout: HTTP timeout in seconds.

    Returns:
        ``(metadata, content)``. ``metadata`` always includes ``id``,
        ``type: "webpage"``, ``URL``, and ``title`` (falls back to the URL).
        ``author``, ``issued``, and ``container-title`` are included when
        meta tags (OpenGraph / Twitter / article:*) provide them.
        ``content`` is the plain-text form of the readability-extracted
        article body.

    Raises:
        httpx.HTTPStatusError: On HTTP non-2xx.
    """
    headers = {"User-Agent": _user_agent()}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        # readability-lxml's `Document` expects a str, not bytes — it uses
        # regex on the charset declaration which chokes on bytes. Use httpx's
        # already-decoded `.text` (which respects the Content-Type header).
        html_text = response.text

    doc = Document(html_text)
    content = _strip_html(doc.summary())

    # `lxml.html.fromstring` happily accepts either str or bytes; keep str
    # for consistency with the readability handoff above.
    root = lxml_html.fromstring(html_text)

    # Title: OG > Twitter > <title> > readability.short_title()
    title_candidates = [
        _meta_property(root, "og:title"),
        _meta_property(root, "twitter:title"),
    ]
    title_tag = root.xpath("//title/text()")
    if title_tag:
        title_candidates.append(str(title_tag[0]).strip())
    title_candidates.append(doc.short_title())
    title = _first_nonempty(*title_candidates) or url

    author = _first_nonempty(
        _meta_property(root, "article:author"),
        _meta_property(root, "author"),
        _meta_name(root, "author"),
        _meta_name(root, "twitter:creator"),
    )

    published = _first_nonempty(
        _meta_property(root, "article:published_time"),
        _meta_property(root, "og:article:published_time"),
        _meta_name(root, "date"),
        _meta_name(root, "dc.date"),
    )

    site_name = _first_nonempty(
        _meta_property(root, "og:site_name"),
        _meta_property(root, "application-name"),
    )

    parsed = urlparse(url)
    metadata: dict[str, Any] = {
        "id": f"url-{parsed.netloc}-{abs(hash(url)) & 0xFFFFFF:06x}",
        "type": "webpage",
        "URL": url,
        "title": title,
    }
    if author:
        metadata["author"] = [{"literal": author}]
    if published:
        issued = _parse_published(published)
        if issued is not None:
            metadata["issued"] = issued
    if site_name:
        metadata["container-title"] = site_name

    return metadata, content
