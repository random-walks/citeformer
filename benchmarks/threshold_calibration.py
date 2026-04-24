"""NLI threshold calibration — precision/recall sweep per model.

Runs the configured NLI model over the hand-labeled triples in
``calibration_data.CALIBRATION_TRIPLES`` and reports precision / recall / F1
at every threshold in ``THRESHOLDS``. Writes a JSON log under
``findings/`` and produces a two-panel figure with the P/R/F1 curves plus a
classic precision-recall scatter.

Run::

    uv run python -m benchmarks.threshold_calibration                  # default model
    uv run python -m benchmarks.threshold_calibration --model smaller  # compare two heads
    uv run python -m benchmarks.threshold_calibration --model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

The point is to replace the "0.5 everywhere" default with a principled,
reproducible recommendation — and to let users re-run on their own labelled
data to pick a threshold that fits their domain.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmarks.calibration_data import CALIBRATION_TRIPLES

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "ai_papers.json"
FINDINGS_DIR = Path(__file__).parent / "findings"
FIGURES_DIR = FINDINGS_DIR / "figures"


#: Thresholds we sweep. 0.05..0.95 in 0.05 steps — finer at the ends where
#: precision typically ramps steeply.
THRESHOLDS: list[float] = [round(0.05 * i, 2) for i in range(1, 20)]


def _build_pairs() -> list[tuple[str, str, bool]]:
    """Resolve each calibration triple against the fixtures abstract."""
    data = json.loads(FIXTURES_PATH.read_text())
    by_label: dict[str, str] = {
        entry["label"]: entry["csl"].get("abstract", "")
        for entry in data
        if entry["csl"].get("abstract")
    }
    missing = [
        paper_label for paper_label, _, _ in CALIBRATION_TRIPLES if paper_label not in by_label
    ]
    if missing:
        raise RuntimeError(
            f"Calibration data references {len(missing)} paper(s) without "
            f"abstracts in fixtures: {missing[:3]}..."
        )
    return [
        (by_label[paper_label], hypothesis, label)
        for paper_label, hypothesis, label in CALIBRATION_TRIPLES
    ]


def _score(pairs: list[tuple[str, str, bool]], nli: Any) -> list[float]:
    """Score every pair with the NLI model. Returns entailment probabilities."""
    results = nli.entail_batch([(premise, hypothesis) for premise, hypothesis, _ in pairs])
    return [r.entailment for r in results]


def _metrics_at_threshold(
    labels: list[bool],
    scores: list[float],
    threshold: float,
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores, strict=True):
        predicted = score >= threshold
        if label and predicted:
            tp += 1
        elif not label and predicted:
            fp += 1
        elif not label and not predicted:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _plot(
    model: str,
    results: list[dict[str, float]],
    out_path: Path,
) -> None:
    """Two-panel calibration figure: threshold sweep + P/R scatter."""
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]
    best_idx = max(range(len(results)), key=lambda i: results[i]["f1"])
    best = results[best_idx]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Threshold sweep ----------------------------------------------------
    left.plot(thresholds, precisions, label="Precision", color="#0a7", linewidth=2)
    left.plot(thresholds, recalls, label="Recall", color="#b33", linewidth=2)
    left.plot(thresholds, f1s, label="F1", color="#225", linewidth=2.5)
    left.axvline(
        x=best["threshold"],
        color="#225",
        linestyle=":",
        alpha=0.5,
        label=f"best F1 @ {best['threshold']:.2f}",
    )
    left.axvline(x=0.5, color="grey", linestyle="--", alpha=0.4, label="default 0.5")
    left.set_xlabel("Entailment threshold")
    left.set_ylabel("Score")
    left.set_ylim(0, 1.05)
    left.set_xlim(0, 1)
    left.grid(alpha=0.25)
    left.legend(loc="lower left", fontsize=9)
    left.set_title(
        f"P / R / F1 across thresholds\nbest F1 = {best['f1']:.2f} "
        f"(P={best['precision']:.2f}, R={best['recall']:.2f}) @ t={best['threshold']:.2f}"
    )

    # --- Precision/Recall curve -------------------------------------------
    right.plot(recalls, precisions, "o-", color="#225", markersize=4, linewidth=1.5)
    right.scatter(
        [best["recall"]],
        [best["precision"]],
        s=180,
        facecolor="#0a7",
        edgecolor="white",
        zorder=5,
        label=f"best F1 @ t={best['threshold']:.2f}",
    )
    # Annotate a handful of threshold labels.
    seen_t: set[float] = set()
    for r in results:
        label_t = round(r["threshold"], 2)
        if label_t in (0.1, 0.3, 0.5, 0.7, 0.9) and label_t not in seen_t:
            right.annotate(
                f"t={label_t:.2f}",
                xy=(r["recall"], r["precision"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#666",
            )
            seen_t.add(label_t)
    right.set_xlabel("Recall")
    right.set_ylabel("Precision")
    right.set_xlim(-0.02, 1.02)
    right.set_ylim(-0.02, 1.05)
    right.grid(alpha=0.25)
    right.set_title("Precision / Recall trade-off")
    right.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"NLI threshold calibration — {model}\n"
        f"{len(CALIBRATION_TRIPLES)} hand-labelled (premise, hypothesis) pairs",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="NLI threshold calibration sweep")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "HF NLI model to calibrate. Defaults to the citeformer default "
            "(DeBERTa-v3-large-MNLI variants). Pass any HF MNLI/NLI checkpoint."
        ),
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Use chunked NLI scoring (slower but handles long premises). "
        "Default off since abstracts fit in a single window.",
    )
    args = parser.parse_args()

    from citeformer.verify.nli import DEFAULT_NLI_MODEL, NLIModel

    model_name = args.model or DEFAULT_NLI_MODEL
    print(f"[calibration] loading NLI model: {model_name}")
    nli_kwargs: dict[str, Any] = {}
    if args.chunked:
        nli_kwargs["chunk_premise"] = True
    nli = NLIModel(model_name=model_name, **nli_kwargs)

    pairs = _build_pairs()
    labels = [triple[2] for triple in CALIBRATION_TRIPLES]
    print(f"[calibration] scoring {len(pairs)} pairs …")
    start = time.time()
    scores = _score(pairs, nli)
    print(f"[calibration]  scored in {time.time() - start:.1f}s")

    results = [_metrics_at_threshold(labels, scores, t) for t in THRESHOLDS]
    best = max(results, key=lambda r: r["f1"])
    print(
        f"[calibration] best F1 = {best['f1']:.3f} at threshold = {best['threshold']:.2f}  "
        f"(P={best['precision']:.3f} R={best['recall']:.3f})"
    )

    FINDINGS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = model_name.split("/")[-1].replace(".", "-")
    log_path = FINDINGS_DIR / f"nli_calibration_{model_slug}.json"
    log_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "chunked": bool(args.chunked),
                "n_pairs": len(pairs),
                "best": best,
                "thresholds": results,
                "raw_scores": [
                    {
                        "paper": paper_label,
                        "hypothesis": hypothesis,
                        "label": label,
                        "entailment": score,
                    }
                    for (paper_label, hypothesis, label), score in zip(
                        CALIBRATION_TRIPLES, scores, strict=True
                    )
                ],
            },
            indent=2,
        )
    )
    print(f"[calibration] JSON log → {log_path}")

    fig_path = FIGURES_DIR / f"nli_calibration_{model_slug}.png"
    _plot(model_name, results, fig_path)
    print(f"[calibration] figure  → {fig_path}")


if __name__ == "__main__":
    main()
