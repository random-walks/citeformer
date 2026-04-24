"""Ingest a BibTeX ``.bib`` file and a Zotero CSL-JSON export.

Motivation: most research users already maintain a local bibliography in
one of these two formats. citeformer ships zero-dep adapters so the
Source list can be constructed without leaving the user's existing
workflow — no DOI round-trip, no metadata scraping.

This example covers both entry points in ~80 lines:

1. ``Source.from_bibtex(path_or_str)`` — parse a BibTeX file into a list
   of Sources. Entry types (``@article``, ``@book``, ``@inproceedings``,
   ``@misc``, …) map to CSL types; ``author = {…}`` field splits on
   ``" and "`` and detects the ``Family, Given`` vs ``Given Family``
   convention per author.

2. ``Source.from_zotero(path_or_list, filter_fn=…)`` — load a Zotero
   *Export → Better CSL JSON* dump. Deduplicates colliding ``id`` values
   (Zotero lets two items collide on citation key) and drops null
   fields.

Then feed the resulting Sources straight into the standard formatter so
you can eyeball that the metadata round-tripped cleanly.

Run::

    uv run python examples/09_bibtex_source.py

No network, no model — just ingest → render.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from citeformer import Source
from citeformer.render import render_single_reference

_BIBTEX_SAMPLE = r"""
@article{vaswani2017,
    author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
              Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and
              Kaiser, Lukasz and Polosukhin, Illia},
    title = {Attention Is All You Need},
    journal = {NeurIPS},
    year = 2017,
    volume = 30,
    pages = {5998--6008},
    doi = {10.48550/arXiv.1706.03762}
}

@inproceedings{devlin2019,
    author = "Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and
              Toutanova, Kristina",
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for
             Language Understanding",
    booktitle = "NAACL-HLT",
    year = 2019,
    month = jun,
    pages = "4171--4186"
}

@book{strunk2000,
    author = {Strunk, William and White, E. B.},
    title = {The Elements of Style},
    publisher = {Macmillan},
    year = 2000,
    edition = {4th},
    address = {New York}
}
"""


_ZOTERO_SAMPLE: list[dict[str, object]] = [
    {
        "id": "smith2021",
        "type": "article-journal",
        "author": [{"family": "Smith", "given": "Alice"}],
        "title": "Structured Decoding for RAG Pipelines",
        "container-title": "Journal of Applied AI",
        "issued": {"date-parts": [["2021", "6"]]},  # Zotero stringifies ints
        "DOI": "10.1234/example.2021",
    },
    {
        # Zotero often re-exports items — dedupe drops the second copy.
        "id": "smith2021",
        "type": "article-journal",
        "title": "Duplicate Entry — Dropped",
    },
    {
        "id": "web-resource",
        "type": "webpage",
        "title": "The Grammar Cookbook",
        "URL": "https://example.com/grammar",
        "accessed": None,  # Null field dropped on load.
    },
]


def _render_block(label: str, sources: list[Source]) -> None:
    print("=" * 78)
    print(f"{label} — {len(sources)} source(s)")
    print("=" * 78)
    for i, src in enumerate(sources, start=1):
        ref = render_single_reference(src, style_name="apa-7", number=i)
        kind = src.metadata.get("type", "?")
        print(f"  [{i}]  {kind:18s}  {ref.rendered}")
    print()


def main() -> None:
    # 1) BibTeX — write a tempfile to demonstrate the Path entry point.
    with TemporaryDirectory() as tmp:
        bib_path = Path(tmp) / "library.bib"
        bib_path.write_text(_BIBTEX_SAMPLE, encoding="utf-8")
        bibtex_sources = Source.from_bibtex(bib_path)

    _render_block("BibTeX (from file)", bibtex_sources)

    # 2) Zotero — hand in the in-memory list directly.
    zotero_sources = Source.from_zotero(_ZOTERO_SAMPLE)
    _render_block("Zotero CSL-JSON (dedupe + null-drop on)", zotero_sources)

    # 3) Zotero filter — the list supports arbitrary predicate filtering
    #    at load time (useful for pulling just articles, just books, or a
    #    specific tag).
    webpages = Source.from_zotero(
        _ZOTERO_SAMPLE,
        filter_fn=lambda item: item.get("type") == "webpage",
    )
    _render_block("Zotero — filter_fn(type == 'webpage')", webpages)


if __name__ == "__main__":
    main()
