"""Standalone renderer demo — all six bundled styles on one CSL-JSON item.

Useful when:

- you want citeformer as a pure CSL renderer (no LLM involved)
- you're debugging a CSL-JSON fixture and want to preview how it'll look
- you're writing a new formatter and want to diff against the existing ones

Run:

    uv run python examples/03_standalone_rendering.py

Installs needed: core only. No torch, no xgrammar.
"""

from __future__ import annotations

from citeformer import Source
from citeformer.render import render_single_reference
from citeformer.render.formatters import available_formatters


def main() -> None:
    items = [
        {
            "id": "vaswani-attention",
            "type": "article-journal",
            "title": "Attention Is All You Need",
            "author": [
                {"family": "Vaswani", "given": "Ashish"},
                {"family": "Shazeer", "given": "Noam"},
                {"family": "Parmar", "given": "Niki"},
                {"family": "Uszkoreit", "given": "Jakob"},
                {"family": "Jones", "given": "Llion"},
                {"family": "Gomez", "given": "Aidan N."},
                {"family": "Kaiser", "given": "\u0141ukasz"},
                {"family": "Polosukhin", "given": "Illia"},
            ],
            "issued": {"date-parts": [[2017]]},
            "container-title": "Advances in Neural Information Processing Systems",
            "volume": "30",
            "page": "5998-6008",
            "DOI": "10.48550/arXiv.1706.03762",
        },
        {
            "id": "austen-pride",
            "type": "book",
            "title": "Pride and Prejudice",
            "author": [{"family": "Austen", "given": "Jane"}],
            "issued": {"date-parts": [[1813]]},
            "publisher": "T. Egerton",
            "publisher-place": "London",
        },
        {
            "id": "knuth-algorithms",
            "type": "chapter",
            "title": "Fundamental Algorithms",
            "author": [{"family": "Knuth", "given": "Donald E."}],
            "issued": {"date-parts": [[1997]]},
            "container-title": "The Art of Computer Programming",
            "volume": "1",
            "edition": "3rd",
            "publisher": "Addison-Wesley",
        },
    ]

    for item in items:
        print("=" * 78)
        print(f"CSL-JSON id: {item['id']}  |  type: {item['type']}")
        print("=" * 78)
        source = Source(metadata=item, content="")
        for style in available_formatters():
            ref = render_single_reference(source, style_name=style, number=1)
            print(f"  {style:24s}  inline={ref.inline_marker:12s}  {ref.rendered}")
        print()


if __name__ == "__main__":
    main()
