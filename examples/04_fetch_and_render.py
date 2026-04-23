"""Fetch metadata from Crossref / arXiv, render in every bundled style.

End-to-end check of the metadata pipeline without touching a language
model: resolve DOIs + arXiv IDs to CSL-JSON, then run the home-grown
formatters over the results.

Good for:

- smoke-testing the `from_doi` / `from_arxiv` adapters on a fresh install
- sanity-checking that a new formatter handles real-world CSL-JSON (which
  Crossref returns with extra fields the spec-minimalist fixtures don't
  exercise)

Run:

    uv run python examples/04_fetch_and_render.py

Results are cached in ``~/.cache/citeformer/metadata/`` via diskcache —
repeat runs don't re-hit the network.

**Politeness**: set `CITEFORMER_CROSSREF_MAILTO=you@example.com` so the
request uses the Crossref polite pool. arXiv reuses the same variable.
"""

from __future__ import annotations

from citeformer import Source
from citeformer.render import render_single_reference
from citeformer.render.formatters import available_formatters


def _preview(source: Source, title_guess: str) -> None:
    print("=" * 78)
    print(f"From: {title_guess}")
    print(f"  Resolved id: {source.metadata.get('id', '?')}  "
          f"  type: {source.metadata.get('type', '?')}")
    print("=" * 78)
    for style in available_formatters():
        ref = render_single_reference(source, style_name=style, number=1)
        print(f"  {style:24s}  inline={ref.inline_marker:12s}  {ref.rendered}")
    print()


def main() -> None:
    # Two identifiers — a Crossref DOI (Nature article) and an arXiv id.
    # Both are well-known references with full metadata.
    doi = "10.1038/s41586-023-06221-2"
    arxiv_id = "1706.03762"  # Attention Is All You Need

    print("Fetching Crossref + arXiv metadata (cached on disk after first run)…\n")

    _preview(Source.from_doi(doi), f"DOI {doi}")
    _preview(Source.from_arxiv(arxiv_id), f"arXiv {arxiv_id}")


if __name__ == "__main__":
    main()
