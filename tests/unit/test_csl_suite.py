"""CSL 1.0 canonical-case suite — 50 hand-curated CSL-JSON items × 6 formatters.

Complements ``test_render_csl.py`` (which snapshots 4 canonical items) with a
broader set that exercises the corners of the CSL 1.0 item-type registry and
the field-presence matrix each formatter dispatches on. Every case here is a
realistic citation a downstream user might hand us — deliberately sampled from
the CSL-JSON schema's canonical types and from the field shapes we've seen
trip home-grown formatters in practice (et-al cutoffs, name particles,
literal-author orgs, single-page vs range, missing dates, Unicode names,
legal / patent / dataset types).

We use pytest-regressions to snapshot all six formatters' output per case into
six YAML files; any drift in the rendering layer shows up as a small, readable
diff. A change that *should* update the snapshot is expected to re-generate
the files with ``pytest --force-regen`` and land in the same PR.

Not a CSL-reference conformance suite — our formatters are home-grown (ADR-004)
and deliberately diverge from citeproc-js reference output where we think the
diff reads better. This suite locks *our* output, not theirs.
"""

from __future__ import annotations

import pytest

from citeformer import Source
from citeformer.render import render_single_reference
from citeformer.render.formatters import available_formatters

# --- The 50 canonical cases ---------------------------------------------------

_CASES: list[tuple[str, dict]] = [
    (
        "classic_book_single_author",
        {
            "id": "poe-raven",
            "type": "book",
            "author": [{"family": "Poe", "given": "Edgar Allan"}],
            "title": "The Raven and Other Poems",
            "publisher": "Wiley and Putnam",
            "publisher-place": "New York",
            "issued": {"date-parts": [[1845]]},
        },
    ),
    (
        "article_two_authors_doi",
        {
            "id": "smith-jones-2023",
            "type": "article-journal",
            "author": [
                {"family": "Smith", "given": "Alice"},
                {"family": "Jones", "given": "Bob"},
            ],
            "title": "Constrained Decoding for RAG",
            "container-title": "Journal of Applied AI",
            "volume": "12",
            "issue": "3",
            "page": "45-62",
            "issued": {"date-parts": [[2023, 4]]},
            "DOI": "10.1234/jaai.2023.12.3.45",
        },
    ),
    (
        "article_single_page",
        {
            "id": "chen-2024-single-page",
            "type": "article-journal",
            "author": [{"family": "Chen", "given": "Wei"}],
            "title": "Short Note on Type Systems",
            "container-title": "SIGPLAN Notices",
            "volume": "59",
            "issue": "2",
            "page": "7",
            "issued": {"date-parts": [[2024]]},
        },
    ),
    (
        "article_many_authors_et_al",
        {
            "id": "vaswani-2017-attention",
            "type": "article-journal",
            "author": [
                {"family": "Vaswani", "given": "Ashish"},
                {"family": "Shazeer", "given": "Noam"},
                {"family": "Parmar", "given": "Niki"},
                {"family": "Uszkoreit", "given": "Jakob"},
                {"family": "Jones", "given": "Llion"},
                {"family": "Gomez", "given": "Aidan"},
                {"family": "Kaiser", "given": "Lukasz"},
                {"family": "Polosukhin", "given": "Illia"},
            ],
            "title": "Attention Is All You Need",
            "container-title": "Advances in Neural Information Processing Systems",
            "volume": "30",
            "issued": {"date-parts": [[2017]]},
        },
    ),
    (
        "book_chapter_with_editor",
        {
            "id": "melville-loomings",
            "type": "chapter",
            "author": [{"family": "Melville", "given": "Herman"}],
            "editor": [{"family": "Tanner", "given": "Tony"}],
            "title": "Loomings",
            "container-title": "Moby-Dick or The Whale",
            "publisher": "Harper & Brothers",
            "publisher-place": "New York",
            "page": "1-6",
            "issued": {"date-parts": [[1851]]},
        },
    ),
    (
        "phd_thesis",
        {
            "id": "austen-thesis",
            "type": "thesis",
            "author": [{"family": "Austen", "given": "Jane"}],
            "title": "A Critical Edition of Early Novels",
            "publisher": "University of Oxford",
            "genre": "PhD dissertation",
            "issued": {"date-parts": [[1813]]},
        },
    ),
    (
        "conference_paper",
        {
            "id": "devlin-2019-bert",
            "type": "paper-conference",
            "author": [
                {"family": "Devlin", "given": "Jacob"},
                {"family": "Chang", "given": "Ming-Wei"},
                {"family": "Lee", "given": "Kenton"},
                {"family": "Toutanova", "given": "Kristina"},
            ],
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "container-title": "NAACL-HLT",
            "page": "4171-4186",
            "issued": {"date-parts": [[2019]]},
            "publisher": "ACL",
        },
    ),
    (
        "technical_report",
        {
            "id": "openai-2023-gpt4-report",
            "type": "report",
            "author": [{"literal": "OpenAI"}],
            "title": "GPT-4 Technical Report",
            "publisher": "OpenAI",
            "number": "arXiv:2303.08774",
            "issued": {"date-parts": [[2023, 3, 27]]},
        },
    ),
    (
        "webpage_with_accessed",
        {
            "id": "claude-docs",
            "type": "webpage",
            "title": "Claude Documentation",
            "container-title": "Anthropic",
            "URL": "https://docs.anthropic.com/claude",
            "issued": {"date-parts": [[2025]]},
            "accessed": {"date-parts": [[2026, 4, 23]]},
        },
    ),
    (
        "book_with_edition",
        {
            "id": "strunk-white-elements",
            "type": "book",
            "author": [
                {"family": "Strunk", "given": "William"},
                {"family": "White", "given": "E. B."},
            ],
            "title": "The Elements of Style",
            "publisher": "Macmillan",
            "publisher-place": "New York",
            "edition": "4th",
            "issued": {"date-parts": [[2000]]},
            "ISBN": "978-0205309023",
        },
    ),
    (
        "journal_with_issn",
        {
            "id": "turing-1950",
            "type": "article-journal",
            "author": [{"family": "Turing", "given": "Alan M."}],
            "title": "Computing Machinery and Intelligence",
            "container-title": "Mind",
            "volume": "59",
            "issue": "236",
            "page": "433-460",
            "issued": {"date-parts": [[1950, 10]]},
            "ISSN": "0026-4423",
        },
    ),
    (
        "multi_volume_series",
        {
            "id": "knuth-taocp-vol3",
            "type": "book",
            "author": [{"family": "Knuth", "given": "Donald E."}],
            "title": "The Art of Computer Programming",
            "volume": "3",
            "collection-title": "Sorting and Searching",
            "publisher": "Addison-Wesley",
            "edition": "2nd",
            "issued": {"date-parts": [[1998]]},
        },
    ),
    (
        "author_with_particle_van_der",
        {
            "id": "van-der-maaten-2008",
            "type": "article-journal",
            "author": [
                {"family": "van der Maaten", "given": "Laurens"},
                {"family": "Hinton", "given": "Geoffrey"},
            ],
            "title": "Visualizing Data using t-SNE",
            "container-title": "Journal of Machine Learning Research",
            "volume": "9",
            "page": "2579-2605",
            "issued": {"date-parts": [[2008]]},
        },
    ),
    (
        "author_with_particle_de_la",
        {
            "id": "de-la-torre-2012",
            "type": "article-journal",
            "author": [{"family": "de la Torre", "given": "Fernando"}],
            "title": "A Least-Squares Framework for Component Analysis",
            "container-title": "IEEE Transactions on PAMI",
            "volume": "34",
            "issue": "6",
            "page": "1041-1055",
            "issued": {"date-parts": [[2012, 6]]},
        },
    ),
    (
        "author_with_particle_von",
        {
            "id": "von-neumann-1945",
            "type": "report",
            "author": [{"family": "von Neumann", "given": "John"}],
            "title": "First Draft of a Report on the EDVAC",
            "publisher": "University of Pennsylvania",
            "issued": {"date-parts": [[1945, 6, 30]]},
        },
    ),
    (
        "literal_organization_author",
        {
            "id": "wcag-2023",
            "type": "report",
            "author": [{"literal": "World Wide Web Consortium"}],
            "title": "Web Content Accessibility Guidelines (WCAG) 2.2",
            "publisher": "W3C",
            "issued": {"date-parts": [[2023, 10]]},
            "URL": "https://www.w3.org/TR/WCAG22/",
        },
    ),
    (
        "single_word_family_name",
        {
            "id": "plato-republic",
            "type": "book",
            "author": [{"family": "Plato"}],
            "title": "The Republic",
            "publisher": "Penguin Classics",
            "translator": [{"family": "Lee", "given": "H. D. P."}],
            "issued": {"date-parts": [[-380]]},
        },
    ),
    (
        "missing_author",
        {
            "id": "anonymous-1850",
            "type": "book",
            "title": "A Treatise on Diverse Topics",
            "publisher": "Smith and Co.",
            "publisher-place": "London",
            "issued": {"date-parts": [[1850]]},
        },
    ),
    (
        "missing_year",
        {
            "id": "undated-draft",
            "type": "article-journal",
            "author": [{"family": "Doe", "given": "Jane"}],
            "title": "Draft: Notes on Reproducibility",
            "container-title": "Preprint",
        },
    ),
    (
        "date_with_month",
        {
            "id": "dated-2024-09",
            "type": "article-journal",
            "author": [{"family": "Okafor", "given": "Ngozi"}],
            "title": "Fall 2024 Roundup",
            "container-title": "Annals of Statistics",
            "issued": {"date-parts": [[2024, 9]]},
            "volume": "52",
            "issue": "3",
            "page": "1100-1125",
        },
    ),
    (
        "date_with_month_and_day",
        {
            "id": "blog-2026-01-15",
            "type": "post-weblog",
            "author": [{"family": "Patel", "given": "Priya"}],
            "title": "What We Learned Shipping v1.0",
            "container-title": "Engineering Blog",
            "issued": {"date-parts": [[2026, 1, 15]]},
            "URL": "https://example.com/blog/v1",
        },
    ),
    (
        "very_long_title",
        {
            "id": "long-title",
            "type": "article-journal",
            "author": [{"family": "Ramirez", "given": "Isabella"}],
            "title": (
                "An Exceedingly Verbose and Deliberately Over-specified Title "
                "Intended to Stress-test the Formatter's Line-wrap and "
                "Punctuation Behaviour Across All Bundled Styles"
            ),
            "container-title": "Journal of Unusually Long Titles",
            "issued": {"date-parts": [[2022]]},
            "volume": "1",
            "page": "1-2",
        },
    ),
    (
        "title_with_colon_subtitle",
        {
            "id": "sutton-barto-rl",
            "type": "book",
            "author": [
                {"family": "Sutton", "given": "Richard S."},
                {"family": "Barto", "given": "Andrew G."},
            ],
            "title": "Reinforcement Learning: An Introduction",
            "publisher": "MIT Press",
            "publisher-place": "Cambridge, MA",
            "edition": "2nd",
            "issued": {"date-parts": [[2018]]},
        },
    ),
    (
        "title_with_question_mark",
        {
            "id": "bender-2021-stochastic",
            "type": "paper-conference",
            "author": [
                {"family": "Bender", "given": "Emily M."},
                {"family": "Gebru", "given": "Timnit"},
                {"family": "McMillan-Major", "given": "Angelina"},
                {"family": "Shmitchell", "given": "Shmargaret"},
            ],
            "title": "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?",
            "container-title": "FAccT '21",
            "page": "610-623",
            "issued": {"date-parts": [[2021, 3]]},
        },
    ),
    (
        "title_with_parentheses",
        {
            "id": "brown-2020-gpt3",
            "type": "article-journal",
            "author": [{"family": "Brown", "given": "Tom B."}],
            "title": "Language Models are Few-Shot Learners (GPT-3)",
            "container-title": "NeurIPS",
            "volume": "33",
            "issued": {"date-parts": [[2020]]},
        },
    ),
    (
        "unicode_scandinavian_surname",
        {
            "id": "odegard-2015",
            "type": "article-journal",
            "author": [{"family": "Ødegård", "given": "Ástor"}],
            "title": "Unicode Handling in Scientific Software",
            "container-title": "Software Engineering Research",
            "volume": "5",
            "issue": "2",
            "page": "12-28",
            "issued": {"date-parts": [[2015]]},
        },
    ),
    (
        "cjk_literal_author",
        {
            "id": "wang-2024",
            "type": "article-journal",
            "author": [{"literal": "王伟"}, {"literal": "田中花子"}],
            "title": "Cross-lingual Representation Learning",
            "container-title": "Journal of Multilingual NLP",
            "volume": "3",
            "page": "100-120",
            "issued": {"date-parts": [[2024]]},
        },
    ),
    (
        "title_with_italics_quotes",
        {
            "id": "italics-paper",
            "type": "book",
            "author": [{"family": "Johnson", "given": "Karen"}],
            "title": 'The Rhetoric of "Truth" in Late Modern Fiction',
            "publisher": "Harvard University Press",
            "issued": {"date-parts": [[2019]]},
        },
    ),
    (
        "newspaper_article",
        {
            "id": "nyt-2025-03",
            "type": "article-newspaper",
            "author": [{"family": "O'Brien", "given": "Liam"}],
            "title": "AI Tools Reshape Newsroom Practices",
            "container-title": "The New York Times",
            "issued": {"date-parts": [[2025, 3, 18]]},
            "section": "Business",
            "URL": "https://www.nytimes.com/2025/03/18/ai-newsroom",
        },
    ),
    (
        "magazine_article",
        {
            "id": "wired-2024-06",
            "type": "article-magazine",
            "author": [{"family": "Nguyen", "given": "Thao"}],
            "title": "Inside the Quiet Revolution of Structured Decoding",
            "container-title": "Wired",
            "issued": {"date-parts": [[2024, 6]]},
            "page": "48-55",
        },
    ),
    (
        "software_with_version",
        {
            "id": "citeformer-v0-1",
            "type": "software",
            "author": [{"family": "Ubik", "given": "Blaise"}],
            "title": "citeformer",
            "version": "0.1.0",
            "URL": "https://github.com/random-walks/citeformer",
            "issued": {"date-parts": [[2026]]},
        },
    ),
    (
        "dataset_with_doi",
        {
            "id": "imagenet-2009",
            "type": "dataset",
            "author": [
                {"family": "Deng", "given": "Jia"},
                {"family": "Dong", "given": "Wei"},
                {"family": "Socher", "given": "Richard"},
            ],
            "title": "ImageNet: A Large-Scale Hierarchical Image Database",
            "publisher": "Stanford Vision Lab",
            "issued": {"date-parts": [[2009]]},
            "DOI": "10.1109/CVPR.2009.5206848",
        },
    ),
    (
        "patent_item",
        {
            "id": "patent-us9999999",
            "type": "patent",
            "author": [{"family": "Edison", "given": "Thomas"}],
            "title": "A Method and Apparatus for Electric Illumination",
            "number": "US9999999",
            "publisher": "United States Patent and Trademark Office",
            "issued": {"date-parts": [[1880, 1, 27]]},
        },
    ),
    (
        "map_item",
        {
            "id": "ordnance-survey-2020",
            "type": "map",
            "author": [{"literal": "Ordnance Survey"}],
            "title": "OS Landranger Map 89: West Cumbria",
            "publisher": "Ordnance Survey",
            "publisher-place": "Southampton",
            "issued": {"date-parts": [[2020]]},
        },
    ),
    (
        "figure_item",
        {
            "id": "fig-saturn",
            "type": "figure",
            "author": [{"literal": "NASA Voyager Imaging Team"}],
            "title": "Saturn from Voyager 2",
            "publisher": "NASA/JPL",
            "issued": {"date-parts": [[1981, 8]]},
            "URL": "https://photojournal.jpl.nasa.gov/catalog/PIA00335",
        },
    ),
    (
        "speech_item",
        {
            "id": "mlk-1963",
            "type": "speech",
            "author": [{"family": "King", "given": "Martin Luther", "suffix": "Jr."}],
            "title": "I Have a Dream",
            "event": "March on Washington for Jobs and Freedom",
            "event-place": "Washington, D.C.",
            "issued": {"date-parts": [[1963, 8, 28]]},
        },
    ),
    (
        "interview_item",
        {
            "id": "interview-feynman",
            "type": "interview",
            "author": [{"family": "Feynman", "given": "Richard P."}],
            "title": "The Pleasure of Finding Things Out",
            "container-title": "BBC Horizon",
            "issued": {"date-parts": [[1981]]},
        },
    ),
    (
        "legislation_item",
        {
            "id": "us-cra-2025",
            "type": "legislation",
            "title": "Consumer Privacy Act of 2025",
            "number": "H.R. 1234",
            "jurisdiction": "United States",
            "issued": {"date-parts": [[2025]]},
        },
    ),
    (
        "bill_item",
        {
            "id": "bill-s-42",
            "type": "bill",
            "title": "Algorithmic Accountability Bill",
            "number": "S. 42",
            "issued": {"date-parts": [[2024, 1, 10]]},
        },
    ),
    (
        "review_item",
        {
            "id": "book-review",
            "type": "review",
            "author": [{"family": "Garcia", "given": "María Elena"}],
            "title": "Review of 'Thinking, Fast and Slow'",
            "container-title": "The New York Review of Books",
            "issued": {"date-parts": [[2012, 11]]},
        },
    ),
    (
        "review_book_item",
        {
            "id": "book-review-2",
            "type": "review-book",
            "author": [{"family": "Hassan", "given": "Yasmin"}],
            "title": "A Lively Polemic",
            "reviewed-title": "The Righteous Mind",
            "reviewed-author": [{"family": "Haidt", "given": "Jonathan"}],
            "container-title": "London Review of Books",
            "issued": {"date-parts": [[2013, 4]]},
        },
    ),
    (
        "broadcast_item",
        {
            "id": "npr-2024-08",
            "type": "broadcast",
            "author": [{"family": "Shapiro", "given": "Ari"}],
            "title": "All Things Considered: AI in Schools",
            "container-title": "NPR",
            "issued": {"date-parts": [[2024, 8, 12]]},
        },
    ),
    (
        "musical_score_item",
        {
            "id": "beethoven-9",
            "type": "musical_score",
            "author": [{"family": "van Beethoven", "given": "Ludwig"}],
            "title": "Symphony No. 9 in D minor, Op. 125",
            "publisher": "Breitkopf & Härtel",
            "issued": {"date-parts": [[1826]]},
        },
    ),
    (
        "motion_picture_item",
        {
            "id": "kubrick-2001",
            "type": "motion_picture",
            "director": [{"family": "Kubrick", "given": "Stanley"}],
            "title": "2001: A Space Odyssey",
            "publisher": "MGM",
            "issued": {"date-parts": [[1968, 4, 6]]},
        },
    ),
    (
        "personal_communication",
        {
            "id": "pc-smith-2025",
            "type": "personal_communication",
            "author": [{"family": "Smith", "given": "Jane"}],
            "title": "Email to the author",
            "issued": {"date-parts": [[2025, 6, 14]]},
        },
    ),
    (
        "entry_dictionary",
        {
            "id": "oed-serendipity",
            "type": "entry-dictionary",
            "title": "Serendipity",
            "container-title": "Oxford English Dictionary",
            "publisher": "Oxford University Press",
            "issued": {"date-parts": [[2023]]},
            "URL": "https://www.oed.com/dictionary/serendipity_n",
        },
    ),
    (
        "entry_encyclopedia",
        {
            "id": "britannica-ai",
            "type": "entry-encyclopedia",
            "author": [{"family": "Russell", "given": "Stuart"}],
            "title": "Artificial intelligence",
            "container-title": "Encyclopaedia Britannica",
            "publisher": "Encyclopaedia Britannica, Inc.",
            "issued": {"date-parts": [[2024]]},
        },
    ),
    (
        "manuscript_item",
        {
            "id": "manuscript-x",
            "type": "manuscript",
            "author": [{"family": "Emerson", "given": "Ralph Waldo"}],
            "title": "Notebook C: Unpublished Reflections",
            "publisher": "Houghton Library, Harvard University",
            "issued": {"date-parts": [[1838]]},
        },
    ),
    (
        "hyphenated_given_name",
        {
            "id": "mary-jane-2020",
            "type": "article-journal",
            "author": [{"family": "Watson", "given": "Mary-Jane"}],
            "title": "Kinetic Analysis in Early Cinema",
            "container-title": "Film Studies Quarterly",
            "volume": "15",
            "page": "33-44",
            "issued": {"date-parts": [[2020]]},
        },
    ),
    (
        "bc_year_date",
        {
            "id": "sun-tzu",
            "type": "book",
            "author": [{"literal": "Sun Tzu"}],
            "title": "The Art of War",
            "publisher": "Public Domain",
            "issued": {"date-parts": [[-500]]},
        },
    ),
]

# Sanity-check the suite size (catch accidental truncation).
assert len(_CASES) >= 50, f"expected >=50 cases, got {len(_CASES)}"


# --- Test helpers -------------------------------------------------------------


def _render_all_for_style(style: str) -> list[dict]:
    """Render every case through one formatter, package for snapshot."""
    out: list[dict] = []
    for case_id, item in _CASES:
        source = Source(metadata=item, content="")
        try:
            ref = render_single_reference(source, style_name=style, number=1)
            entry: dict = {
                "case_id": case_id,
                "inline": ref.inline_marker,
                "bib": ref.rendered,
            }
        except Exception as exc:
            entry = {
                "case_id": case_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        out.append(entry)
    return out


def _all_cases_render_cleanly(style: str) -> None:
    """Sanity: no case raises and no output contains `..` / `  ` / whitespace tails."""
    for case_id, item in _CASES:
        source = Source(metadata=item, content="")
        ref = render_single_reference(source, style_name=style, number=1)
        assert ".." not in ref.rendered, f"{style} double-period on {case_id}"
        assert "  " not in ref.rendered, f"{style} double-space on {case_id}"
        assert ref.rendered == ref.rendered.strip(), f"{style} whitespace on {case_id}"
        assert ref.rendered.strip(), f"{style} empty output on {case_id}"


# --- Hygiene test (fast sanity) ----------------------------------------------


@pytest.mark.parametrize("style", available_formatters())
def test_all_cases_render_cleanly(style: str) -> None:
    _all_cases_render_cleanly(style)


# --- Snapshot per formatter --------------------------------------------------


def test_csl_suite_snapshot_apa_7(data_regression) -> None:  # type: ignore[no-untyped-def]
    data_regression.check(_render_all_for_style("apa-7"))


def test_csl_suite_snapshot_mla_9(data_regression) -> None:  # type: ignore[no-untyped-def]
    data_regression.check(_render_all_for_style("mla-9"))


def test_csl_suite_snapshot_chicago_author_date(data_regression) -> None:  # type: ignore[no-untyped-def]
    data_regression.check(_render_all_for_style("chicago-author-date"))


def test_csl_suite_snapshot_ieee(data_regression) -> None:  # type: ignore[no-untyped-def]
    data_regression.check(_render_all_for_style("ieee"))


def test_csl_suite_snapshot_nature(data_regression) -> None:  # type: ignore[no-untyped-def]
    data_regression.check(_render_all_for_style("nature"))


def test_csl_suite_snapshot_vancouver(data_regression) -> None:  # type: ignore[no-untyped-def]
    data_regression.check(_render_all_for_style("vancouver"))
