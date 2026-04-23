"""Generate the annotated benchmark figures used in the README + docs.

Reads the sweep JSON logs under ``benchmarks/findings/`` and emits PNG
figures to ``benchmarks/findings/figures/``. Two figures land today:

- ``fabrication-structural-vs-empirical.png`` — the cover figure. Stacks
  three scenarios (sweep aggregate, adversarial baseline, structural ceiling)
  side-by-side to make the "0% vs N%" swing unmistakable.
- ``sweep-summary.png`` — per-model bar chart with mean ± std error bars.
  Shows fabrication and NLI support rate for citeformer vs. baseline on
  each model in the latest sweep.

Run:

    uv run python -m benchmarks.plot

Add ``--adversarial-fab 100 --adversarial-cites 8 --adversarial-label ...``
to override the adversarial numbers (the sweep JSON doesn't capture them —
they come from an `adversarial.py` run and we'd overbake the script if we
tried to auto-correlate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Palette: structural = blue, empirical = orange. Works for color-blind
# (uses both color + hatch patterns for the cover figure).
_CITEFORMER_COLOR = "#2563eb"  # blue
_BASELINE_COLOR = "#ea580c"  # orange
_GRID_COLOR = "#e5e7eb"
_TEXT_COLOR = "#111827"

_FINDINGS_DIR = Path(__file__).parent / "findings"
_FIGURES_DIR = _FINDINGS_DIR / "figures"


def _load_latest_sweep(findings_dir: Path) -> dict[str, Any]:
    """Pick the most recent ``sweep-*.json`` by mtime."""
    sweeps = sorted(findings_dir.glob("sweep-*.json"))
    if not sweeps:
        raise SystemExit(
            f"No sweep JSONs in {findings_dir}. Run `python -m benchmarks.sweep` first."
        )
    latest = sweeps[-1]
    return dict(json.loads(latest.read_text()))


def _load_merged_sweeps(findings_dir: Path) -> dict[str, Any]:
    """Merge every `sweep-*.json` in `findings_dir`, keyed by (model, premise).

    When a sweep overwrites a prior run for the same (model, premise), the
    newer file wins. Different premise modes on the same model coexist — so
    plots can compare e.g. abstract vs. fulltext for Qwen 0.5B side by side.
    Older sweeps without a `config.premise` key are treated as `abstract`
    (the pre-v0.1 default).
    """
    sweeps = sorted(findings_dir.glob("sweep-*.json"))
    if not sweeps:
        raise SystemExit(
            f"No sweep JSONs in {findings_dir}. Run `python -m benchmarks.sweep` first."
        )

    # Key = (model, premise). Later file wins.
    merged_aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    merged_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    merged_config: dict[str, Any] = {}

    for path in sweeps:
        data = json.loads(path.read_text())
        premise = data.get("config", {}).get("premise", "abstract")
        for model, agg in data.get("aggregates", {}).items():
            key = (model, premise)
            merged_aggregates[key] = agg
            merged_rows_by_key[key] = []
        for row in data.get("rows", []):
            key = (row["model"], premise)
            if key in merged_rows_by_key:
                merged_rows_by_key[key].append({**row, "premise": premise})
        merged_config = data.get("config", merged_config)

    flat_rows = [row for rows in merged_rows_by_key.values() for row in rows]

    return {
        # Legacy keys kept for back-compat with cover / summary plots: latest
        # premise per model wins (fulltext > abstract when both exist).
        "aggregates": _latest_premise_per_model(merged_aggregates),
        "rows": flat_rows,
        "config": merged_config,
        # New: keyed by (model, premise) for premise-comparison plots.
        "by_premise": {f"{m}::{p}": agg for (m, p), agg in merged_aggregates.items()},
    }


def _latest_premise_per_model(
    merged: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collapse (model, premise) → model, preferring `abstract` when present.

    Grammar enforcement is independent of NLI premise — fabrication and
    citation-count metrics are the same regardless of which premise we
    score against. So for the default "sweep summary" view we prefer the
    premise we have the most seed data for (abstract, since that was the
    v0.1 default). The premise-comparison plot reads both premises
    directly from `by_premise`, so we don't lose the fulltext narrative.
    """
    priority = {"abstract": 1, "fulltext": 0}
    best: dict[str, tuple[int, dict[str, Any]]] = {}
    for (model, premise), agg in merged.items():
        score = priority.get(premise, 0)
        if model not in best or score > best[model][0]:
            best[model] = (score, agg)
    return {m: agg for m, (_, agg) in best.items()}


def _style_axes(ax: Axes) -> None:
    """Uniform styling: subtle grid, clean spines, dark text."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=_GRID_COLOR, linestyle="-", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID_COLOR)
    ax.spines["bottom"].set_color(_GRID_COLOR)
    ax.tick_params(colors=_TEXT_COLOR, which="both")
    for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        lbl.set_color(_TEXT_COLOR)


def _annotate_bars(ax: Axes, rects: Any, fmt: str = "{:.1f}%") -> None:
    """Put the percentage value on top of each bar."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=_TEXT_COLOR,
        )


def _plot_fabrication_cover(
    *,
    sweep: dict[str, Any],
    adversarial_baseline_fab: float,
    adversarial_label: str,
    output_path: Path,
) -> None:
    """The README cover figure: structural vs. empirical fabrication.

    Three scenarios on the x-axis:
    - "sweep avg" — baseline's mean fab rate across the default sweep
    - adversarial run — the prompt that *invites* `[7]`/`[8]`
    - "worst case" — a hypothetical adversary capped by nothing

    citeformer is always 0% (annotated "structural — grammar-enforced");
    baseline is whatever the observed rate is.
    """
    # Aggregate baseline fabrication mean across all models in the sweep.
    rates = [agg["baseline_fab_rate"]["mean"] * 100 for agg in sweep["aggregates"].values()]
    sweep_avg = sum(rates) / len(rates) if rates else 0.0
    n_runs = len(sweep.get("rows", []))
    n_models = len(sweep["aggregates"])

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    scenarios = [
        f"Standard sweep\n({n_models} models, {n_runs} runs total)",
        adversarial_label,
    ]
    baseline_rates = [sweep_avg, adversarial_baseline_fab]
    citeformer_rates = [0.0, 0.0]

    import numpy as np

    x = np.arange(len(scenarios))
    width = 0.38

    rects1 = ax.bar(
        x - width / 2,
        baseline_rates,
        width,
        color=_BASELINE_COLOR,
        label="Baseline (plain HF generate)",
        edgecolor="white",
        linewidth=1,
    )
    rects2 = ax.bar(
        x + width / 2,
        citeformer_rates,
        width,
        color=_CITEFORMER_COLOR,
        label="citeformer (grammar-enforced)",
        edgecolor="white",
        linewidth=1,
    )

    ax.set_ylabel("Fabrication rate (%)", fontsize=11, fontweight="semibold")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_title(
        "Citation fabrication is structural, not statistical",
        fontsize=14,
        fontweight="bold",
        pad=14,
        loc="left",
    )
    ax.text(
        0.0,
        1.02,
        "% of emitted [N] markers pointing at a source that doesn't exist",
        transform=ax.transAxes,
        fontsize=10,
        color="#4b5563",
        ha="left",
        va="bottom",
        fontstyle="italic",
    )

    _annotate_bars(ax, rects1)
    _annotate_bars(ax, rects2)

    # Annotate the structural 0% — the key message.
    ax.annotate(
        "structural: grammar mask\neliminates out-of-range tokens\nat every decode step",
        xy=(x[1] + width / 2, 0),
        xytext=(x[1] + width / 2, 55),
        fontsize=9,
        color=_CITEFORMER_COLOR,
        ha="center",
        arrowprops={
            "arrowstyle": "->",
            "color": _CITEFORMER_COLOR,
            "lw": 1.2,
        },
    )

    ax.legend(loc="upper left", frameon=False, fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_sweep_summary(sweep: dict[str, Any], output_path: Path) -> None:
    """Per-model comparison: citeformer vs. baseline, cite count + fab rate.

    Two subplots:
    - left: mean cites emitted, with std error bars
    - right: mean fabrication rate, with std error bars
    """
    import numpy as np

    aggregates: dict[str, dict[str, Any]] = sweep["aggregates"]
    models = list(aggregates.keys())
    short_names = [m.split("/")[-1] for m in models]

    c_cites_mean = [aggregates[m]["constrained_cites"]["mean"] for m in models]
    c_cites_std = [aggregates[m]["constrained_cites"]["std"] for m in models]
    b_cites_mean = [aggregates[m]["baseline_cites"]["mean"] for m in models]
    b_cites_std = [aggregates[m]["baseline_cites"]["std"] for m in models]

    c_fab_mean = [aggregates[m]["constrained_fab_rate"]["mean"] * 100 for m in models]
    c_fab_std = [aggregates[m]["constrained_fab_rate"]["std"] * 100 for m in models]
    b_fab_mean = [aggregates[m]["baseline_fab_rate"]["mean"] * 100 for m in models]
    b_fab_std = [aggregates[m]["baseline_fab_rate"]["std"] * 100 for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    x = np.arange(len(models))
    width = 0.38

    # Citations emitted
    r1 = ax1.bar(
        x - width / 2,
        b_cites_mean,
        width,
        yerr=b_cites_std,
        color=_BASELINE_COLOR,
        label="Baseline",
        edgecolor="white",
        linewidth=1,
        capsize=4,
    )
    r2 = ax1.bar(
        x + width / 2,
        c_cites_mean,
        width,
        yerr=c_cites_std,
        color=_CITEFORMER_COLOR,
        label="citeformer",
        edgecolor="white",
        linewidth=1,
        capsize=4,
    )
    ax1.set_title(
        "Citations emitted per run (mean \u00b1 std)",
        fontsize=12,
        fontweight="semibold",
        loc="left",
        pad=8,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=9, rotation=15, ha="right")
    ax1.set_ylabel("Citations emitted", fontsize=10)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    _annotate_bars(ax1, r1, fmt="{:.0f}")
    _annotate_bars(ax1, r2, fmt="{:.0f}")
    _style_axes(ax1)

    # Fabrication rate
    r3 = ax2.bar(
        x - width / 2,
        b_fab_mean,
        width,
        yerr=b_fab_std,
        color=_BASELINE_COLOR,
        edgecolor="white",
        linewidth=1,
        capsize=4,
    )
    r4 = ax2.bar(
        x + width / 2,
        c_fab_mean,
        width,
        yerr=c_fab_std,
        color=_CITEFORMER_COLOR,
        edgecolor="white",
        linewidth=1,
        capsize=4,
    )
    ax2.set_title(
        "Fabrication rate (mean \u00b1 std)",
        fontsize=12,
        fontweight="semibold",
        loc="left",
        pad=8,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=9, rotation=15, ha="right")
    ax2.set_ylabel("Fabrication rate (%)", fontsize=10)
    ax2.set_ylim(0, max(max([*b_fab_mean, 1]) * 2.2, 15))
    _annotate_bars(ax2, r3)
    _annotate_bars(ax2, r4)
    _style_axes(ax2)

    # Global footer: run config.
    n_seeds = len(sweep["config"]["seeds"])
    policy = sweep["config"]["policy"]
    fig.text(
        0.01,
        -0.02,
        f"Policy: {policy}   |   Seeds: {n_seeds}   |   "
        f"NLI threshold: {sweep['config']['threshold']}",
        fontsize=8,
        color="#6b7280",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_premise_comparison(sweep: dict[str, Any], output_path: Path) -> None:
    """Show how NLI support rate swings between abstract and fulltext premises.

    For each model that has both abstract-premise and fulltext-premise runs
    in the findings directory, plot paired bars: citeformer support rate
    (abstract) vs. citeformer support rate (fulltext). Dramatically
    illustrates that the "small-model ceiling" was often premise-driven,
    not model-driven.
    """
    import numpy as np

    by_premise: dict[str, dict[str, Any]] = sweep["by_premise"]

    # Regroup into {model: {premise: agg}}. Skip models with only one premise.
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for key, agg in by_premise.items():
        model, premise = key.rsplit("::", 1)
        grouped.setdefault(model, {})[premise] = agg
    paired = {m: v for m, v in grouped.items() if "abstract" in v and "fulltext" in v}

    if not paired:
        print("[plot] no abstract+fulltext pairs found — skipping premise-comparison plot")
        return

    models = list(paired.keys())
    short_names = [m.split("/")[-1] for m in models]
    abstract_supp = [paired[m]["abstract"]["constrained_support_rate"]["mean"] * 100 for m in models]
    fulltext_supp = [paired[m]["fulltext"]["constrained_support_rate"]["mean"] * 100 for m in models]
    abstract_std = [paired[m]["abstract"]["constrained_support_rate"]["std"] * 100 for m in models]
    fulltext_std = [paired[m]["fulltext"]["constrained_support_rate"]["std"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    x = np.arange(len(models))
    width = 0.38

    # Use a secondary palette distinct from the fabrication cover:
    # abstract = muted gray, fulltext = bold purple.
    abstract_color = "#9ca3af"
    fulltext_color = "#7c3aed"

    r1 = ax.bar(
        x - width / 2,
        abstract_supp,
        width,
        yerr=abstract_std,
        color=abstract_color,
        label="Abstract-only premise (~1-2k chars)",
        edgecolor="white",
        linewidth=1,
        capsize=4,
    )
    r2 = ax.bar(
        x + width / 2,
        fulltext_supp,
        width,
        yerr=fulltext_std,
        color=fulltext_color,
        label="Full-text premise (~20k chars via pypdf)",
        edgecolor="white",
        linewidth=1,
        capsize=4,
    )

    # Draw swing arrows between each pair.
    for i, model in enumerate(models):
        ax.annotate(
            "",
            xy=(x[i] + width / 2, fulltext_supp[i]),
            xytext=(x[i] - width / 2, abstract_supp[i]),
            arrowprops={
                "arrowstyle": "->",
                "color": fulltext_color,
                "lw": 1.5,
                "alpha": 0.4,
            },
        )
        lift = fulltext_supp[i] - abstract_supp[i]
        ax.text(
            x[i],
            max(fulltext_supp[i], abstract_supp[i]) + 6,
            f"+{lift:.0f} pts",
            ha="center",
            fontsize=10,
            color=fulltext_color,
            fontweight="semibold",
        )
        del model  # appease ruff

    ax.set_ylabel("citeformer NLI support rate (%)", fontsize=11, fontweight="semibold")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_title(
        "Support-rate ceiling was the premise, not the model",
        fontsize=14,
        fontweight="bold",
        pad=14,
        loc="left",
    )
    ax.text(
        0.0,
        1.02,
        "Swapping the NLI premise from abstract to PDF body text lifts support rates dramatically",
        transform=ax.transAxes,
        fontsize=10,
        color="#4b5563",
        ha="left",
        va="bottom",
        fontstyle="italic",
    )
    _annotate_bars(ax, r1)
    _annotate_bars(ax, r2)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Entry point for ``python -m benchmarks.plot``."""
    parser = argparse.ArgumentParser(description=__doc__ or "", add_help=True)
    parser.add_argument(
        "--findings-dir",
        type=Path,
        default=_FINDINGS_DIR,
        help="Directory containing sweep-*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_FIGURES_DIR,
        help="Where to write PNGs.",
    )
    parser.add_argument(
        "--adversarial-baseline-fab",
        type=float,
        default=100.0,
        help=(
            "Baseline fabrication rate from the adversarial run, in percent. "
            "Defaults to 100 (the observed Qwen 0.5B, seed=0 value)."
        ),
    )
    parser.add_argument(
        "--adversarial-label",
        default="Adversarial prompt\n(demands [7] and [8])",
        help="X-axis label for the adversarial bar in the cover figure.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Merge all sweep files so multi-session data (e.g. bigger-model runs
    # landing separately) shows up on the same plot.
    sweep = _load_merged_sweeps(args.findings_dir)

    cover = args.output_dir / "fabrication-structural-vs-empirical.png"
    _plot_fabrication_cover(
        sweep=sweep,
        adversarial_baseline_fab=args.adversarial_baseline_fab,
        adversarial_label=args.adversarial_label,
        output_path=cover,
    )
    print(f"[plot] wrote {cover}")

    summary = args.output_dir / "sweep-summary.png"
    _plot_sweep_summary(sweep, summary)
    print(f"[plot] wrote {summary}")

    premise = args.output_dir / "premise-comparison.png"
    _plot_premise_comparison(sweep, premise)
    if premise.exists():
        print(f"[plot] wrote {premise}")


if __name__ == "__main__":
    main()
