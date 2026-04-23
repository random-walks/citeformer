"""Multi-seed + multi-model benchmark sweep.

`demo.py` reports a single run — useful for a demo, noisy for a claim. This
script runs the same prompt across (model, seed) pairs and reports mean ±
std for each metric. Turns "citeformer got 14% support rate that one time"
into "citeformer gets 17% ± 4 across five seeds on Qwen 0.5B."

Results are written both to stdout (pretty table) and to a JSON log at
``benchmarks/findings/sweep-<timestamp>.json`` so multi-run comparison is
possible without rerunning.

Run:

    uv sync --extra dev --extra hf --extra verify
    uv run python -m benchmarks.sweep
    uv run python -m benchmarks.sweep --models Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM-360M-Instruct
    uv run python -m benchmarks.sweep --seeds 0 1 2 3 4

Defaults pick two small models that are typically already cached after
running ``demo.py``. Larger models (Phi-3.5-mini, Llama-3.2-3B) work via
``--models ...`` but will be downloaded on first use.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
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

# Two small models that most users already have cached after `demo.py`. Larger
# models work through `--models` but aren't baked into the default set because
# we don't want the sweep to silently download ~7 GB on first run.
DEFAULT_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct",
)
DEFAULT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class RunRecord:
    """One (model, seed) row in the sweep's output table."""

    model: str
    seed: int
    policy: str
    constrained_cites: int
    constrained_fab_rate: float
    constrained_support_rate: float
    baseline_cites: int
    baseline_fab_rate: float
    baseline_support_rate: float
    elapsed_sec: float


@dataclass(frozen=True)
class Aggregate:
    """Mean / std / min / max for a single metric across seeds."""

    mean: float
    std: float
    min: float
    max: float
    n: int

    @classmethod
    def from_values(cls, values: list[float]) -> Aggregate:
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


def _default_prompt(sources: list[Any]) -> str:
    """Same shape as `demo.py` — share the prompt for apples-to-apples."""
    return build_rag_prompt(
        query=(
            "Write five citation-dense sentences tracing the development of "
            "transformer-based language models, citing at least one of the "
            "sources in every sentence."
        ),
        sources=sources,
        system=(
            "You are writing a brief, citation-dense technical survey. "
            "CITE EVERY CLAIM."
        ),
        example=(
            "The Transformer architecture introduced self-attention [1]. "
            "BERT extended this with bidirectional pre-training [2]."
        ),
        answer_prefix="Survey:",
    )


def _one_run(
    model: str,
    seed: int,
    *,
    sources: list[Any],
    prompt: str,
    nli: Any,
    max_new_tokens: int,
    threshold: float,
    device: str,
    policy: Policy,
) -> RunRecord:
    """Execute a single (model, seed) combination, returning one row."""
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
    baseline = analyze_run(
        "baseline",
        baseline_text,
        sources,
        nli=nli,
        threshold=threshold,
    )
    elapsed = time.perf_counter() - start
    return RunRecord(
        model=model,
        seed=seed,
        policy=policy.value,
        constrained_cites=len(constrained.cite_ids_emitted),
        constrained_fab_rate=fabrication_rate(constrained),
        constrained_support_rate=constrained.support_rate,
        baseline_cites=len(baseline.cite_ids_emitted),
        baseline_fab_rate=fabrication_rate(baseline),
        baseline_support_rate=baseline.support_rate,
        elapsed_sec=elapsed,
    )


def _print_per_run_table(rows: list[RunRecord]) -> None:
    header = (
        f"{'model':35s} {'seed':>4s} {'policy':8s} "
        f"{'C-cites':>7s} {'C-fab%':>6s} {'C-supp%':>7s} "
        f"{'B-cites':>7s} {'B-fab%':>6s} {'B-supp%':>7s} {'sec':>5s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        # Shorten the model name display so long HF paths don't misalign the table.
        short_model = r.model.split("/")[-1]
        print(
            f"{short_model:35s} {r.seed:>4d} {r.policy:8s} "
            f"{r.constrained_cites:>7d} {r.constrained_fab_rate * 100:>6.1f} "
            f"{r.constrained_support_rate * 100:>7.1f} "
            f"{r.baseline_cites:>7d} {r.baseline_fab_rate * 100:>6.1f} "
            f"{r.baseline_support_rate * 100:>7.1f} "
            f"{r.elapsed_sec:>5.1f}"
        )


def _aggregate_by_model(
    rows: list[RunRecord],
) -> dict[str, dict[str, Aggregate]]:
    """Group rows by model and compute per-metric aggregates across seeds."""
    by_model: dict[str, list[RunRecord]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)

    out: dict[str, dict[str, Aggregate]] = {}
    for model, records in by_model.items():
        out[model] = {
            "constrained_support_rate": Aggregate.from_values(
                [r.constrained_support_rate for r in records]
            ),
            "baseline_support_rate": Aggregate.from_values(
                [r.baseline_support_rate for r in records]
            ),
            "constrained_cites": Aggregate.from_values(
                [float(r.constrained_cites) for r in records]
            ),
            "baseline_cites": Aggregate.from_values(
                [float(r.baseline_cites) for r in records]
            ),
            "constrained_fab_rate": Aggregate.from_values(
                [r.constrained_fab_rate for r in records]
            ),
            "baseline_fab_rate": Aggregate.from_values(
                [r.baseline_fab_rate for r in records]
            ),
            "elapsed_sec": Aggregate.from_values([r.elapsed_sec for r in records]),
        }
    return out


def _print_aggregate_table(aggregates: dict[str, dict[str, Aggregate]]) -> None:
    print()
    print("Aggregate across seeds (mean ± std):")
    print(
        f"{'model':35s}  {'n':>3s}  "
        f"{'C-supp%':>16s} {'B-supp%':>16s}  "
        f"{'C-cites':>14s} {'B-cites':>14s}  "
        f"{'C-fab%':>14s} {'B-fab%':>14s}"
    )
    print("-" * 150)
    for model, agg in aggregates.items():
        short_model = model.split("/")[-1]
        csup = agg["constrained_support_rate"]
        bsup = agg["baseline_support_rate"]
        cct = agg["constrained_cites"]
        bct = agg["baseline_cites"]
        cfab = agg["constrained_fab_rate"]
        bfab = agg["baseline_fab_rate"]
        print(
            f"{short_model:35s}  {csup.n:>3d}  "
            f"{csup.mean * 100:>6.1f} \u00b1 {csup.std * 100:>5.1f}  "
            f"{bsup.mean * 100:>6.1f} \u00b1 {bsup.std * 100:>5.1f}  "
            f"{cct.mean:>5.1f} \u00b1 {cct.std:>4.1f}   "
            f"{bct.mean:>5.1f} \u00b1 {bct.std:>4.1f}   "
            f"{cfab.mean * 100:>5.1f} \u00b1 {cfab.std * 100:>4.1f}   "
            f"{bfab.mean * 100:>5.1f} \u00b1 {bfab.std * 100:>4.1f}"
        )


def _write_json_log(
    rows: list[RunRecord],
    aggregates: dict[str, dict[str, Aggregate]],
    config: dict[str, Any],
) -> Path:
    """Persist the sweep to benchmarks/findings/sweep-<timestamp>.json."""
    findings_dir = Path(__file__).parent / "findings"
    findings_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = findings_dir / f"sweep-{stamp}.json"
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "rows": [asdict(r) for r in rows],
        "aggregates": {
            model: {metric: asdict(agg) for metric, agg in metrics.items()}
            for model, metrics in aggregates.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    """Entry point for ``python -m benchmarks.sweep``."""
    parser = argparse.ArgumentParser(description=__doc__ or "", add_help=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"HF model ids to sweep. Defaults: {', '.join(DEFAULT_MODELS)}.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Torch seeds to iterate. Defaults: {list(DEFAULT_SEEDS)}.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nli-model", default=None, help="Override the NLI model")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument(
        "--policy",
        choices=["required", "auto", "quotes_only"],
        default="required",
    )
    args = parser.parse_args()

    fixtures = load_fixtures()
    sources = sources_from_fixtures(fixtures)
    prompt = _default_prompt(sources)
    policy = Policy(args.policy)

    print(f"[sweep] NLI warmup (loaded once, reused across {len(args.models) * len(args.seeds)} runs)…")
    from citeformer.verify import NLIModel

    nli_kwargs: dict[str, Any] = {}
    if args.nli_model:
        nli_kwargs["model_name"] = args.nli_model
    nli = NLIModel(**nli_kwargs)

    total = len(args.models) * len(args.seeds)
    rows: list[RunRecord] = []
    for i, (model, seed) in enumerate(
        [(m, s) for m in args.models for s in args.seeds], start=1
    ):
        print(f"[sweep] {i}/{total}: model={model!r} seed={seed}")
        row = _one_run(
            model,
            seed,
            sources=sources,
            prompt=prompt,
            nli=nli,
            max_new_tokens=args.max_new_tokens,
            threshold=args.threshold,
            device=args.device,
            policy=policy,
        )
        rows.append(row)
        print(
            f"        constrained={row.constrained_cites} cites, "
            f"support={row.constrained_support_rate:.0%}, fab={row.constrained_fab_rate:.0%} | "
            f"baseline={row.baseline_cites} cites, "
            f"support={row.baseline_support_rate:.0%}, fab={row.baseline_fab_rate:.0%} | "
            f"{row.elapsed_sec:.1f}s"
        )

    print()
    _print_per_run_table(rows)

    aggregates = _aggregate_by_model(rows)
    _print_aggregate_table(aggregates)

    config: dict[str, Any] = {
        "models": list(args.models),
        "seeds": list(args.seeds),
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "nli_model": args.nli_model,
        "device": args.device,
        "policy": policy.value,
    }
    log_path = _write_json_log(rows, aggregates, config)
    print(f"\n[sweep] wrote {log_path}")


if __name__ == "__main__":
    main()
