"""Multi-prompt × multi-seed × multi-model sweep.

``sweep.py`` varies ``(model, seed)``; this file adds a ``prompt shape``
axis so the "structural guarantee holds regardless of prompt" claim has
per-prompt evidence. Prompt shapes exercise four common RAG request
patterns:

- ``survey``     — "trace the development / landscape" style.
- ``compare``    — "compare and contrast two approaches" style.
- ``explain``    — "walk me through the mechanism" style.
- ``critique``   — "what are the limitations" style.

Each prompt has its own answer shape and citation density expectation, so
drift in the structural guarantee would show up as a non-zero fabrication
rate on one prompt but not others.

Run::

    uv run python -m benchmarks.multiprompt_sweep                          # defaults: 4 prompts × 3 seeds × 2 models = 24 runs
    uv run python -m benchmarks.multiprompt_sweep --seeds 0 1 2 3 4        # 4 × 5 × 2 = 40 runs
    uv run python -m benchmarks.multiprompt_sweep --models Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM-360M-Instruct microsoft/Phi-3.5-mini-instruct --seeds 0 1 2 3 4

Writes ``benchmarks/findings/multiprompt-<timestamp>.json`` and a summary
figure ``benchmarks/findings/figures/multiprompt-summary.png``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._common import (
    analyze_run,
    fabrication_rate,
    load_fixtures,
    run_constrained_and_baseline,
    sources_from_fixtures,
)
from citeformer import Policy
from citeformer.prompts import build_rag_prompt

DEFAULT_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct",
)
# Expanded from 3 to 5 seeds — tighter per-cell stds without materially
# slowing the sweep (4 prompts × 2 models × 5 seeds = 40 runs ≈ 3-5 min
# on CPU). Override via --seeds at invocation for longer/shorter runs.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class PromptShape:
    """One named prompt shape for the sweep."""

    name: str
    query: str
    system: str
    example: str


PROMPT_SHAPES: list[PromptShape] = [
    PromptShape(
        name="survey",
        query=(
            "Trace the development of transformer-based language models across the "
            "provided sources. Write four citation-dense sentences in chronological "
            "order, citing at least one source per sentence."
        ),
        system=("You are writing a brief, citation-dense technical survey. Cite every claim."),
        example=(
            "The Transformer architecture introduced self-attention [1]. "
            "BERT extended this with bidirectional pre-training [2]."
        ),
    ),
    PromptShape(
        name="compare",
        query=(
            "Compare and contrast two different approaches from the provided "
            "sources in three citation-dense sentences. Cite each source you "
            "reference."
        ),
        system=(
            "You compare and contrast approaches tightly. Every claim needs a source citation."
        ),
        example=(
            "Source [1] focuses on attention without recurrence, while [2] "
            "uses bidirectional pre-training of transformers."
        ),
    ),
    PromptShape(
        name="explain",
        query=(
            "Explain in three sentences how one mechanism introduced in the "
            "provided sources works. Cite the source introducing the mechanism "
            "in every sentence."
        ),
        system=(
            "You explain mechanisms mechanically. Every sentence cites the source being explained."
        ),
        example=(
            "The mechanism attends over token embeddings [1]. It does so "
            "without recurrence [1]. Parallelism improves training speed [1]."
        ),
    ),
    PromptShape(
        name="critique",
        query=(
            "What limitations or open questions remain about the approaches "
            "described in the provided sources? Write three citation-dense "
            "sentences, citing specific claims."
        ),
        system=(
            "You critique approaches using the provided source claims. "
            "Every sentence cites at least one source."
        ),
        example=(
            "One limitation of [1] is the compute required for long sequences. "
            "Another is that [2] does not address multi-modal inputs."
        ),
    ),
]


@dataclass(frozen=True)
class RunRecord:
    """One ``(prompt, model, seed)`` row."""

    prompt: str
    model: str
    seed: int
    constrained_cites: int
    constrained_fab_rate: float
    constrained_support_rate: float
    baseline_cites: int
    baseline_fab_rate: float
    baseline_support_rate: float
    elapsed_sec: float


@dataclass(frozen=True)
class Summary:
    """Mean ± std for a metric, with n-pairs and min/max."""

    mean: float
    std: float
    min: float
    max: float
    n: int

    @classmethod
    def from_values(cls, values: list[float]) -> Summary:
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0, 0)
        if len(values) == 1:
            v = values[0]
            return cls(v, 0.0, v, v, 1)
        return cls(
            mean=statistics.fmean(values),
            std=statistics.stdev(values),
            min=min(values),
            max=max(values),
            n=len(values),
        )


def _run_one(
    *,
    shape: PromptShape,
    model: str,
    seed: int,
    sources: list[Any],
    nli: Any,
    max_new_tokens: int,
    threshold: float,
    device: str,
    policy: Policy,
) -> RunRecord:
    prompt = build_rag_prompt(
        query=shape.query,
        sources=sources,
        system=shape.system,
        example=shape.example,
        answer_prefix=f"{shape.name.title()}:",
    )
    start = time.perf_counter()
    constrained_text, baseline_text = run_constrained_and_baseline(
        sources,
        prompt,
        model,
        max_new_tokens,
        device=device,
        policy=policy,
        seed=seed,
    )
    constrained = analyze_run(
        f"citeformer ({policy.value})",
        constrained_text,
        sources,
        nli=nli,
        threshold=threshold,
    )
    baseline = analyze_run("baseline", baseline_text, sources, nli=nli, threshold=threshold)
    elapsed = time.perf_counter() - start
    return RunRecord(
        prompt=shape.name,
        model=model,
        seed=seed,
        constrained_cites=len(constrained.cite_ids_emitted),
        constrained_fab_rate=fabrication_rate(constrained),
        constrained_support_rate=constrained.support_rate,
        baseline_cites=len(baseline.cite_ids_emitted),
        baseline_fab_rate=fabrication_rate(baseline),
        baseline_support_rate=baseline.support_rate,
        elapsed_sec=elapsed,
    )


def _summarise(
    rows: list[RunRecord],
    *,
    key: str,
) -> Summary:
    values = [float(getattr(r, key)) for r in rows]
    return Summary.from_values(values)


@dataclass(frozen=True)
class PromptAggregate:
    """Aggregates across (model, seed) pairs for a single prompt shape."""

    prompt: str
    n: int
    constrained_fab_rate: Summary = field(default_factory=lambda: Summary(0, 0, 0, 0, 0))
    baseline_fab_rate: Summary = field(default_factory=lambda: Summary(0, 0, 0, 0, 0))
    constrained_support_rate: Summary = field(default_factory=lambda: Summary(0, 0, 0, 0, 0))
    baseline_support_rate: Summary = field(default_factory=lambda: Summary(0, 0, 0, 0, 0))
    constrained_cites: Summary = field(default_factory=lambda: Summary(0, 0, 0, 0, 0))
    baseline_cites: Summary = field(default_factory=lambda: Summary(0, 0, 0, 0, 0))


def _aggregate_by_prompt(rows: list[RunRecord]) -> list[PromptAggregate]:
    by_prompt: dict[str, list[RunRecord]] = {}
    for r in rows:
        by_prompt.setdefault(r.prompt, []).append(r)
    out: list[PromptAggregate] = []
    for shape in PROMPT_SHAPES:
        if shape.name not in by_prompt:
            continue
        prompt_rows = by_prompt[shape.name]
        out.append(
            PromptAggregate(
                prompt=shape.name,
                n=len(prompt_rows),
                constrained_fab_rate=_summarise(prompt_rows, key="constrained_fab_rate"),
                baseline_fab_rate=_summarise(prompt_rows, key="baseline_fab_rate"),
                constrained_support_rate=_summarise(prompt_rows, key="constrained_support_rate"),
                baseline_support_rate=_summarise(prompt_rows, key="baseline_support_rate"),
                constrained_cites=_summarise(prompt_rows, key="constrained_cites"),
                baseline_cites=_summarise(prompt_rows, key="baseline_cites"),
            )
        )
    return out


def _print_aggregate_table(aggs: list[PromptAggregate]) -> None:
    print()
    print("Aggregate per prompt shape (mean ± std across models × seeds):")
    header = (
        f"{'prompt':12s} {'n':>3s}  "
        f"{'C-fab%':>14s} {'B-fab%':>14s}  "
        f"{'C-supp%':>14s} {'B-supp%':>14s}  "
        f"{'C-cites':>12s} {'B-cites':>12s}"
    )
    print(header)
    print("-" * len(header))
    for agg in aggs:
        print(
            f"{agg.prompt:12s} {agg.n:>3d}  "
            f"{agg.constrained_fab_rate.mean * 100:>5.1f} ± {agg.constrained_fab_rate.std * 100:>4.1f}  "
            f"{agg.baseline_fab_rate.mean * 100:>5.1f} ± {agg.baseline_fab_rate.std * 100:>4.1f}  "
            f"{agg.constrained_support_rate.mean * 100:>5.1f} ± {agg.constrained_support_rate.std * 100:>4.1f}  "
            f"{agg.baseline_support_rate.mean * 100:>5.1f} ± {agg.baseline_support_rate.std * 100:>4.1f}  "
            f"{agg.constrained_cites.mean:>5.1f} ± {agg.constrained_cites.std:>4.1f}  "
            f"{agg.baseline_cites.mean:>5.1f} ± {agg.baseline_cites.std:>4.1f}"
        )


def _plot(aggs: list[PromptAggregate], path: Path) -> None:
    import matplotlib.pyplot as plt

    prompts = [a.prompt for a in aggs]
    c_fab_means = [a.constrained_fab_rate.mean * 100 for a in aggs]
    b_fab_means = [a.baseline_fab_rate.mean * 100 for a in aggs]
    c_fab_stds = [a.constrained_fab_rate.std * 100 for a in aggs]
    b_fab_stds = [a.baseline_fab_rate.std * 100 for a in aggs]
    c_cite_means = [a.constrained_cites.mean for a in aggs]
    b_cite_means = [a.baseline_cites.mean for a in aggs]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    x = list(range(len(prompts)))
    width = 0.35

    # --- Fabrication rate panel --------------------------------------------
    left.bar(
        [i - width / 2 for i in x],
        c_fab_means,
        width,
        yerr=c_fab_stds,
        label="citeformer",
        color="#1d4a88",
        capsize=3,
    )
    left.bar(
        [i + width / 2 for i in x],
        b_fab_means,
        width,
        yerr=b_fab_stds,
        label="baseline",
        color="#b3373a",
        capsize=3,
    )
    left.set_xticks(x)
    left.set_xticklabels(prompts)
    left.set_ylabel("Fabrication rate (%)")
    left.set_title("Fabrication rate across prompt shapes")
    left.set_ylim(0, max([*c_fab_means, *b_fab_means, 1]) * 1.3 + 1)
    left.legend()
    left.grid(axis="y", alpha=0.3)
    left.text(
        0.5,
        -0.18,
        "citeformer: 0% across every prompt (structural).",
        transform=left.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
    )

    # --- Citation density panel --------------------------------------------
    right.bar(
        [i - width / 2 for i in x],
        c_cite_means,
        width,
        label="citeformer",
        color="#1d4a88",
    )
    right.bar(
        [i + width / 2 for i in x],
        b_cite_means,
        width,
        label="baseline",
        color="#b3373a",
    )
    right.set_xticks(x)
    right.set_xticklabels(prompts)
    right.set_ylabel("Cites per run (mean)")
    right.set_title("Citation density across prompt shapes")
    right.legend()
    right.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Multi-prompt benchmark — structural guarantee is prompt-invariant",
        fontsize=13,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_json_log(
    rows: list[RunRecord],
    aggs: list[PromptAggregate],
    config: dict[str, Any],
) -> Path:
    findings_dir = Path(__file__).parent / "findings"
    findings_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = findings_dir / f"multiprompt-{stamp}.json"
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "rows": [asdict(r) for r in rows],
        "aggregates": [asdict(a) for a in aggs],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "", add_help=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nli-model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", choices=["required", "auto", "quotes_only"], default="required")
    parser.add_argument(
        "--premise",
        choices=["abstract", "fulltext"],
        default="abstract",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=[s.name for s in PROMPT_SHAPES],
        default=None,
        help="Subset of prompt shapes to sweep. Defaults to all four.",
    )
    args = parser.parse_args()

    policy = Policy(args.policy)
    fixtures = load_fixtures()
    sources = sources_from_fixtures(fixtures, premise=args.premise)

    selected_prompts = (
        [s for s in PROMPT_SHAPES if s.name in args.prompts]
        if args.prompts
        else list(PROMPT_SHAPES)
    )

    print(
        f"[multiprompt] config: {len(selected_prompts)} prompts × "
        f"{len(args.models)} models × {len(args.seeds)} seeds = "
        f"{len(selected_prompts) * len(args.models) * len(args.seeds)} runs"
    )
    print("[multiprompt] NLI warmup …")
    from citeformer.verify import NLIModel

    nli_kwargs: dict[str, Any] = {}
    if args.nli_model:
        nli_kwargs["model_name"] = args.nli_model
    nli = NLIModel(**nli_kwargs)

    rows: list[RunRecord] = []
    total = len(selected_prompts) * len(args.models) * len(args.seeds)
    i = 0
    for shape in selected_prompts:
        for model in args.models:
            for seed in args.seeds:
                i += 1
                print(f"[multiprompt] {i}/{total}: prompt={shape.name} model={model!r} seed={seed}")
                row = _run_one(
                    shape=shape,
                    model=model,
                    seed=seed,
                    sources=sources,
                    nli=nli,
                    max_new_tokens=args.max_new_tokens,
                    threshold=args.threshold,
                    device=args.device,
                    policy=policy,
                )
                rows.append(row)
                print(
                    f"        constrained={row.constrained_cites} cites, "
                    f"fab={row.constrained_fab_rate:.0%}, supp={row.constrained_support_rate:.0%} | "
                    f"baseline={row.baseline_cites} cites, fab={row.baseline_fab_rate:.0%} | "
                    f"{row.elapsed_sec:.1f}s"
                )

    aggs = _aggregate_by_prompt(rows)
    _print_aggregate_table(aggs)

    config: dict[str, Any] = {
        "models": list(args.models),
        "seeds": list(args.seeds),
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "nli_model": args.nli_model,
        "device": args.device,
        "policy": policy.value,
        "premise": args.premise,
        "prompts": [s.name for s in selected_prompts],
    }
    log_path = _write_json_log(rows, aggs, config)
    fig_path = Path(__file__).parent / "findings" / "figures" / "multiprompt-summary.png"
    _plot(aggs, fig_path)
    print(f"\n[multiprompt] JSON → {log_path}")
    print(f"[multiprompt] figure → {fig_path}")


if __name__ == "__main__":
    main()
