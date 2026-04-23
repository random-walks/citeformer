"""Pre-fetch AI-paper fixtures for the demo benchmark.

Runs ``citeformer.metadata.fetch_arxiv`` against a canonical set of
well-known AI papers and writes the CSL-JSON metadata + abstracts as a
committable fixture at ``benchmarks/fixtures/ai_papers.json``. The benchmark
then loads the fixture (no network) so it's reproducible anywhere.

Re-run after arXiv metadata updates:

    uv run python -m benchmarks.fetch_fixtures

The arXiv version suffix is stripped by ``fetch_arxiv`` so output is stable
across paper revisions.
"""

from __future__ import annotations

import json
from pathlib import Path

from citeformer.metadata import fetch_arxiv

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


def main() -> None:
    """Fetch each paper's CSL-JSON + abstract and write the fixture file."""
    repo_root = Path(__file__).parent.parent
    out_path = repo_root / "benchmarks" / "fixtures" / "ai_papers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for arxiv_id, label in PAPERS:
        print(f"  fetching {arxiv_id}: {label}")
        csl = fetch_arxiv(arxiv_id, use_cache=True)
        entries.append({"arxiv_id": arxiv_id, "label": label, "csl": csl})

    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"wrote {len(entries)} entries → {out_path}")


if __name__ == "__main__":
    main()
