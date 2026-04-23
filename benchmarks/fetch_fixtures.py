"""Pre-fetch AI-paper fixtures for the demo benchmark.

Runs ``citeformer.metadata.fetch_arxiv`` against a canonical set of
well-known AI papers and writes the CSL-JSON metadata + abstracts as a
committable fixture at ``benchmarks/fixtures/ai_papers.json``.

Pass ``--fulltext`` to also download each paper's PDF and extract the body
text via ``pypdf`` into a ``fulltext`` field on each entry. That's what
enables the full-text NLI premise mode in the benchmarks
(``benchmarks.demo --premise fulltext``). ~2 MB/paper, cached through
diskcache via ``citeformer.metadata``.

Re-run after arXiv metadata updates:

    uv run python -m benchmarks.fetch_fixtures
    uv run python -m benchmarks.fetch_fixtures --fulltext

The arXiv version suffix is stripped by ``fetch_arxiv`` so output is stable
across paper revisions.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import httpx

from citeformer._version import __version__
from citeformer.metadata import fetch_arxiv

_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
_PDF_TIMEOUT = 60.0

# Six canonical AI papers — mix of architecture, prompting, scaling, and
# finetuning work. Keeps the benchmark's RAG context small (six sources fit in
# any reasonable context window) while covering enough ground that the model
# has plausible things to say.
PAPERS: list[tuple[str, str]] = [
    ("1706.03762", "Attention Is All You Need (Vaswani et al., 2017)"),
    ("1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)"),
    ("2005.14165", "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)"),
    ("2201.11903", "Chain-of-Thought Prompting (Wei et al., 2022)"),
    ("2302.13971", "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)"),
    ("2305.14314", "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)"),
]


def _fetch_fulltext(arxiv_id: str) -> str:
    """Download the arXiv PDF and extract body text via ``pypdf``.

    Simpler than running a GROBID server — pypdf can't distinguish body
    text from headers/footers/figures cleanly, but it captures enough
    for NLI entailment experiments. See ``benchmarks/README.md`` for
    the caveat.
    """
    from pypdf import PdfReader

    url = _ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    headers = {
        "User-Agent": f"citeformer/{__version__} (+https://github.com/random-walks/citeformer)"
    }
    with httpx.Client(timeout=_PDF_TIMEOUT, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        pdf_bytes = response.content

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    # Collapse whitespace; pypdf emits each line verbatim with ragged spaces.
    return "\n".join(line.strip() for line in "\n".join(pages).splitlines() if line.strip())


def main() -> None:
    """Fetch each paper's CSL-JSON + abstract and write the fixture file."""
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--fulltext",
        action="store_true",
        help="Also download each paper's PDF and add a `fulltext` field.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
        help="Cap full-text length per paper (default 20k). Abstracts are ~2k.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    out_path = repo_root / "benchmarks" / "fixtures" / "ai_papers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Re-use any existing entries (preserves `fulltext` across runs that
    # don't pass `--fulltext`).
    existing: dict[str, dict] = {}
    if out_path.exists():
        for entry in json.loads(out_path.read_text()):
            existing[entry["arxiv_id"]] = entry

    entries = []
    for arxiv_id, label in PAPERS:
        print(f"  fetching {arxiv_id}: {label}")
        csl = fetch_arxiv(arxiv_id, use_cache=True)
        entry: dict = {"arxiv_id": arxiv_id, "label": label, "csl": csl}

        prior = existing.get(arxiv_id, {})
        if args.fulltext:
            print("    downloading PDF + extracting body text …")
            try:
                text = _fetch_fulltext(arxiv_id)
                entry["fulltext"] = text[: args.max_chars]
            except Exception as e:
                print(f"    WARN: fulltext fetch failed for {arxiv_id}: {e}")
                if "fulltext" in prior:
                    entry["fulltext"] = prior["fulltext"]
        elif "fulltext" in prior:
            # Preserve previously-fetched fulltext on non-fulltext runs.
            entry["fulltext"] = prior["fulltext"]

        entries.append(entry)

    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    mode = "fulltext" if args.fulltext else "metadata-only"
    print(f"wrote {len(entries)} entries ({mode}) → {out_path}")


if __name__ == "__main__":
    main()
