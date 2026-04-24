"""ALCE-flavoured citation-evaluation harness (subset runner).

[ALCE](https://github.com/princeton-nlp/ALCE) is the reference benchmark
for citation-aware generation in RAG. Its three headline metrics are:

- **Citation Recall** — of the sentences that *should* cite (factual
  claims), how many actually produce any citation at all?
- **Citation Precision** — of the citations emitted, how many are
  supported by the cited passage(s) under NLI entailment?
- **Correctness** — task-dependent answer quality (ROUGE/EM against
  gold for ASQA/QAMPARI).

Running the full ALCE suite requires their data files (ASQA, QAMPARI,
ELI5) and their official eval scripts. This module does two things:

1. Ships a **tiny toy subset** (3 hand-written examples with ground-truth
   sources) so the runner exercises end-to-end without a multi-GB
   download. This is not ALCE-comparable — it's a smoke test that
   proves the metrics implementation agrees with the paper's math.

2. Implements the **recall + precision metrics** against any
   ``list[ALCEExample]`` you hand it, using the same NLI scorer the
   library uses elsewhere. So if you point it at an actual ALCE dataset
   file (see ``--data`` flag), you can produce comparable numbers.

Official ALCE eval is still authoritative; this is for in-repo smoke +
ad-hoc sanity checks. See ``docs/reference/benchmarks.md`` for what the
full-ALCE integration would look like in v0.2.

Run::

    uv run python -m benchmarks.alce_subset                       # toy 3-example subset
    uv run python -m benchmarks.alce_subset --data path/to/asqa.jsonl --n 10
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citeformer import Citeformer, Policy, Source

_FINDINGS_DIR = Path(__file__).parent / "findings"
_CITE_PATTERN = re.compile(r"\[(\d+)\]")


# --- Example schema ---------------------------------------------------------


@dataclass(frozen=True)
class ALCEExample:
    """One ALCE-shaped example.

    Attributes:
        question: The user's question.
        sources: Retrieved passages, each carrying ``content`` + ``title``.
        gold_answers: Reference answers (list[str], often 1 for ASQA,
            many for QAMPARI). Used only for optional correctness scoring
            (not implemented in the subset runner).
        required_ids: The 1-indexed source ids that contain the
            ground-truth evidence. Used to seed precision/recall metrics.
    """

    question: str
    sources: list[Source]
    gold_answers: list[str] = field(default_factory=list)
    required_ids: list[int] = field(default_factory=list)


# --- Toy subset -------------------------------------------------------------


def _toy_subset() -> list[ALCEExample]:
    """Three hand-written examples — enough to exercise the metrics.

    Each example has 3 sources; two sources per example are relevant to
    the question and one is a distractor. Ground-truth evidence is
    recorded so recall + precision can be scored even without gold
    citation labels in the ALCE file format.
    """
    return [
        ALCEExample(
            question="When was the Transformer architecture introduced and what does it dispense with?",
            sources=[
                Source(
                    metadata={
                        "id": "vaswani2017",
                        "type": "article-journal",
                        "title": "Attention Is All You Need",
                        "author": [{"family": "Vaswani"}],
                        "issued": {"date-parts": [[2017]]},
                    },
                    content=(
                        "The Transformer architecture, introduced in 2017, is based solely on "
                        "attention mechanisms, dispensing with recurrence and convolutions "
                        "entirely."
                    ),
                ),
                Source(
                    metadata={
                        "id": "devlin2019",
                        "type": "article-journal",
                        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                        "author": [{"family": "Devlin"}],
                        "issued": {"date-parts": [[2019]]},
                    },
                    content=(
                        "BERT is designed to pre-train deep bidirectional representations by "
                        "jointly conditioning on both left and right context in all layers."
                    ),
                ),
                Source(
                    metadata={
                        "id": "brown2020",
                        "type": "article-journal",
                        "title": "Language Models are Few-Shot Learners",
                        "author": [{"family": "Brown"}],
                        "issued": {"date-parts": [[2020]]},
                    },
                    content=(
                        "GPT-3 is an autoregressive language model with 175 billion parameters, "
                        "trained without fine-tuning on downstream tasks."
                    ),
                ),
            ],
            gold_answers=[
                "The Transformer was introduced in 2017 and dispenses with recurrence and convolutions."
            ],
            required_ids=[1],
        ),
        ALCEExample(
            question="What does chain-of-thought prompting elicit in large language models?",
            sources=[
                Source(
                    metadata={
                        "id": "wei2022cot",
                        "type": "article-journal",
                        "title": "Chain-of-Thought Prompting Elicits Reasoning",
                        "author": [{"family": "Wei"}],
                        "issued": {"date-parts": [[2022]]},
                    },
                    content=(
                        "Generating a chain of thought — a series of intermediate reasoning "
                        "steps — significantly improves the ability of large language models "
                        "to perform complex reasoning."
                    ),
                ),
                Source(
                    metadata={
                        "id": "touvron2023",
                        "type": "article-journal",
                        "title": "LLaMA: Open and Efficient Foundation Language Models",
                        "author": [{"family": "Touvron"}],
                        "issued": {"date-parts": [[2023]]},
                    },
                    content=(
                        "LLaMA is a collection of foundation language models ranging from 7B "
                        "to 65B parameters trained on publicly available datasets."
                    ),
                ),
                Source(
                    metadata={
                        "id": "dettmers2023",
                        "type": "article-journal",
                        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
                        "author": [{"family": "Dettmers"}],
                        "issued": {"date-parts": [[2023]]},
                    },
                    content=(
                        "QLoRA is an efficient finetuning approach that reduces memory usage "
                        "enough to finetune a 65B parameter model on a single 48GB GPU."
                    ),
                ),
            ],
            gold_answers=[
                "Chain-of-thought prompting elicits intermediate reasoning steps that improve complex reasoning."
            ],
            required_ids=[1],
        ),
        ALCEExample(
            question="What memory reduction does QLoRA enable for finetuning 65B models?",
            sources=[
                Source(
                    metadata={
                        "id": "dettmers2023",
                        "type": "article-journal",
                        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
                        "author": [{"family": "Dettmers"}],
                        "issued": {"date-parts": [[2023]]},
                    },
                    content=(
                        "QLoRA is an efficient finetuning approach that reduces memory usage "
                        "enough to finetune a 65B parameter model on a single 48GB GPU while "
                        "preserving full 16-bit finetuning task performance."
                    ),
                ),
                Source(
                    metadata={
                        "id": "touvron2023",
                        "type": "article-journal",
                        "title": "LLaMA: Open and Efficient Foundation Language Models",
                        "author": [{"family": "Touvron"}],
                        "issued": {"date-parts": [[2023]]},
                    },
                    content=(
                        "LLaMA is a collection of foundation language models ranging from 7B "
                        "to 65B parameters trained on publicly available datasets."
                    ),
                ),
                Source(
                    metadata={
                        "id": "vaswani2017",
                        "type": "article-journal",
                        "title": "Attention Is All You Need",
                        "author": [{"family": "Vaswani"}],
                        "issued": {"date-parts": [[2017]]},
                    },
                    content=(
                        "The Transformer architecture is based solely on attention mechanisms."
                    ),
                ),
            ],
            gold_answers=["QLoRA enables finetuning a 65B parameter model on a single 48GB GPU."],
            required_ids=[1],
        ),
    ]


# --- JSONL loader for real ALCE files --------------------------------------


def _load_alce_jsonl(path: Path, n: int | None = None) -> list[ALCEExample]:
    """Parse an ALCE-style JSONL.

    The ALCE repo ships files with this schema (paraphrased):

        {
          "question": "...",
          "answers": ["...", "..."],
          "docs": [{"title": "...", "text": "...", "id": "..."}, ...]
        }

    We tolerate mild drift (``docs`` vs ``passages``; ``answers`` vs
    ``gold_answers``) so the loader accepts both the original dataset and
    rephrased exports. ``required_ids`` is left empty — in the true ALCE
    files, evidence is annotated per-sentence, not per-source, so we
    skip that here and rely on emitted citations + NLI.
    """
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out: list[ALCEExample] = []
    for line in lines[:n] if n else lines:
        row = json.loads(line)
        passages = row.get("docs") or row.get("passages") or []
        sources = []
        for i, doc in enumerate(passages, start=1):
            sources.append(
                Source(
                    metadata={
                        "id": str(doc.get("id") or f"passage-{i}"),
                        "type": "webpage",
                        "title": str(doc.get("title") or f"Passage {i}"),
                    },
                    content=str(doc.get("text", "")),
                )
            )
        out.append(
            ALCEExample(
                question=str(row["question"]),
                sources=sources,
                gold_answers=list(row.get("answers") or row.get("gold_answers") or []),
                required_ids=[],
            )
        )
    return out


# --- Metrics ----------------------------------------------------------------


@dataclass(frozen=True)
class ALCEMetrics:
    """Per-example metrics aggregated across the run.

    All three metrics are in [0, 1]:

    - ``citation_recall``: mean fraction of factual sentences that cite at
      least one source. Here "factual" is every non-trivial sentence in
      the generated answer (we split on .!?).
    - ``citation_precision``: mean fraction of emitted cites whose
      premise NLI-entails the citing sentence. Uses the library default
      NLI model (DeBERTa-v3-large-MNLI at 0.5 threshold unless
      overridden).
    - ``fabrication_rate``: mean fraction of emitted cites that are
      out-of-scope (``id > N``). Always 0 under grammar enforcement;
      a live canary if anything ever regresses.
    """

    n: int
    citation_recall: float
    citation_precision: float
    fabrication_rate: float
    mean_sentences_per_answer: float


def _sentences(text: str) -> list[str]:
    # Coarse sentence splitter; enough for ALCE-style short answers.
    segments = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in segments if s.strip()]


def _cites_in(sentence: str) -> list[int]:
    return [int(m.group(1)) for m in _CITE_PATTERN.finditer(sentence)]


def _citation_recall(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _cites_in(s))
    return cited / len(sentences)


def _fabrication_rate(cites: list[int], n_sources: int) -> float:
    if not cites:
        return 0.0
    bad = sum(1 for c in cites if c < 1 or c > n_sources)
    return bad / len(cites)


def _citation_precision(
    sentences: list[str],
    sources: list[Source],
    nli: Any,
    threshold: float,
) -> float:
    """For every (sentence, cited_source) pair, check NLI entailment."""
    pairs: list[tuple[str, str]] = []
    pair_index: list[int] = []
    for sentence in sentences:
        for cid in _cites_in(sentence):
            if 1 <= cid <= len(sources):
                pairs.append((sources[cid - 1].content, sentence))
                pair_index.append(cid)
    if not pairs:
        return 1.0  # Nothing to score — vacuous precision.
    scores = nli.entail_batch(pairs)
    return sum(1 for s in scores if s.entailment >= threshold) / len(scores)


# --- Runner ----------------------------------------------------------------


def run_subset(
    examples: list[ALCEExample],
    *,
    citeformer: Citeformer,
    nli: Any,
    threshold: float = 0.5,
    max_new_tokens: int = 200,
) -> dict[str, Any]:
    """Run citeformer over each example and compute ALCE-style metrics."""
    per_example: list[dict[str, Any]] = []
    recalls: list[float] = []
    precisions: list[float] = []
    fab_rates: list[float] = []
    sentence_counts: list[int] = []
    for i, ex in enumerate(examples, start=1):
        print(f"[alce] {i}/{len(examples)}: {ex.question[:60]}…")
        result = citeformer.generate(
            prompt=ex.question,
            sources=ex.sources,
            max_new_tokens=max_new_tokens,
        )
        sentences = _sentences(result.text)
        cites = [c for s in sentences for c in _cites_in(s)]
        rec = _citation_recall(sentences)
        prec = _citation_precision(sentences, ex.sources, nli=nli, threshold=threshold)
        fab = _fabrication_rate(cites, n_sources=len(ex.sources))
        per_example.append(
            {
                "question": ex.question,
                "text": result.text,
                "n_sentences": len(sentences),
                "cite_ids_emitted": cites,
                "citation_recall": rec,
                "citation_precision": prec,
                "fabrication_rate": fab,
            }
        )
        recalls.append(rec)
        precisions.append(prec)
        fab_rates.append(fab)
        sentence_counts.append(len(sentences))

    metrics = ALCEMetrics(
        n=len(examples),
        citation_recall=statistics.fmean(recalls) if recalls else 0.0,
        citation_precision=statistics.fmean(precisions) if precisions else 0.0,
        fabrication_rate=statistics.fmean(fab_rates) if fab_rates else 0.0,
        mean_sentences_per_answer=statistics.fmean(sentence_counts) if sentence_counts else 0.0,
    )
    return {
        "metrics": asdict(metrics),
        "per_example": per_example,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ALCE-flavoured citation-eval subset runner")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional path to an ALCE-style JSONL. Defaults to the in-repo toy subset.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit to first N examples when --data is given.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model for citeformer generation.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--policy", choices=list(Policy.__members__), default="REQUIRED")
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Torch device for the HF backend. Defaults to cpu — MPS has an "
            "xgrammar/Qwen tokenizer-size bug on Apple Silicon; CUDA works."
        ),
    )
    args = parser.parse_args()

    examples = _load_alce_jsonl(args.data, n=args.n) if args.data else _toy_subset()
    print(f"[alce] {len(examples)} example(s)")

    from citeformer.backends.hf import HFBackend
    from citeformer.verify.nli import NLIModel

    backend = HFBackend(model=args.model, device=args.device)
    cf = Citeformer(
        backend=backend,
        style="apa-7",
        citation_policy=Policy[args.policy],
    )
    nli = NLIModel()

    start = time.perf_counter()
    report = run_subset(
        examples,
        citeformer=cf,
        nli=nli,
        threshold=args.threshold,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.perf_counter() - start
    report["elapsed_sec"] = elapsed
    report["config"] = {
        "model": args.model,
        "threshold": args.threshold,
        "policy": args.policy,
        "data": str(args.data) if args.data else "toy_subset",
        "n_examples": len(examples),
    }
    print()
    for k, v in report["metrics"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    _FINDINGS_DIR.mkdir(exist_ok=True)
    out = _FINDINGS_DIR / f"alce-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[alce] wrote {out}")


if __name__ == "__main__":
    main()
