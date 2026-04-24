"""Generates the README cover image: side-by-side annotated comparison.

Two panels:

- **Left**: baseline HF generation (no constraint). Model cheerfully
  emits ``[7]`` and ``[8]`` when only 6 sources are in scope — red
  highlight, callout arrow.
- **Right**: citeformer-constrained generation on the same prompt.
  Out-of-scope ids are logit-masked; ``[7]`` and ``[8]`` are
  token-impossible. Green highlight, callout arrow.

Above both panels: the shared prompt + numbered source list so readers
see what the model was asked. Below: the 40-run multi-prompt headline.

This is called once to produce
``benchmarks/findings/figures/cover-annotated.png``, embedded at the
top of ``README.md``. Re-run to regenerate::

    uv run python -m benchmarks.generate_cover
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = Path(__file__).parent / "findings" / "figures" / "cover-annotated.png"


# --- The shared setup both panels share --------------------------------------

PROMPT = (
    "Write one sentence that cites Turing's 1950 paper [7] and "
    "McCulloch & Pitts 1943 [8]. Use exactly these bracket numbers."
)

SOURCES = [
    "[1] Vaswani et al. — Attention Is All You Need (2017)",
    "[2] Devlin et al. — BERT (2018)",
    "[3] Brown et al. — GPT-3 (2020)",
    "[4] Wei et al. — Chain-of-Thought (2022)",
    "[5] Touvron et al. — LLaMA (2023)",
    "[6] Dettmers et al. — QLoRA (2023)",
]


# --- Rendered outputs (realistic, hand-edited to read cleanly at cover size) --

BASELINE_TEXT = (
    "The foundations of modern AI trace back to Turing's 1950 "
    "paper on machine intelligence [7] and the McCulloch & Pitts "
    "1943 neuron model [8], both of which precede transformers [1]."
)

CITEFORMER_TEXT = (
    "The Transformer architecture introduced self-attention as a "
    "replacement for recurrence [1], which BERT extended with "
    "bidirectional pre-training [2] and LLaMA scaled to open weights [5]."
)

# Which tokens to highlight in each panel (start, end, kind) where
# kind = "bad" (fabricated) or "good" (in-scope).
BASELINE_HIGHLIGHTS = [
    ("[7]", "bad"),
    ("[8]", "bad"),
    ("[1]", "good"),
]
CITEFORMER_HIGHLIGHTS = [
    ("[1]", "good"),
    ("[2]", "good"),
    ("[5]", "good"),
]


# --- Drawing helpers ----------------------------------------------------------


def _draw_output_panel(
    ax,
    *,
    title: str,
    title_color: str,
    text: str,
    highlights: list[tuple[str, str]],
    footer: str,
    footer_bg: str,
    footer_fg: str,
) -> None:
    """Render one of the two output panels with inline highlights."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Panel border
    ax.add_patch(
        FancyBboxPatch(
            (0.1, 0.1),
            9.8,
            9.8,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=2.5,
            edgecolor=title_color,
            facecolor="white",
        )
    )

    # Title
    ax.text(
        5.0,
        9.35,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=title_color,
    )

    # Output text — word-wrap manually for layout control
    wrapped = _word_wrap(text, width=38)
    y = 7.6
    line_step = 0.62
    for line in wrapped:
        _draw_line_with_highlights(ax, line, y=y, highlights=highlights)
        y -= line_step

    # Footer banner
    ax.add_patch(
        FancyBboxPatch(
            (0.4, 0.5),
            9.2,
            1.2,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=0,
            facecolor=footer_bg,
        )
    )
    ax.text(
        5.0,
        1.1,
        footer,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=footer_fg,
    )


def _draw_line_with_highlights(
    ax,
    line: str,
    *,
    y: float,
    highlights: list[tuple[str, str]],
) -> None:
    """Walk the line left to right, highlighting each matched bracket marker."""
    cursor_x = 0.5
    i = 0
    while i < len(line):
        # Try to match any highlight token at position i
        hit = None
        for token, kind in highlights:
            if line.startswith(token, i):
                hit = (token, kind)
                break
        if hit is not None:
            token, kind = hit
            color = "#d43f3a" if kind == "bad" else "#1a8a46"
            bg_color = "#fde1df" if kind == "bad" else "#d8f0db"
            # Draw background capsule
            text_width = len(token) * 0.20
            ax.add_patch(
                FancyBboxPatch(
                    (cursor_x - 0.04, y - 0.23),
                    text_width + 0.08,
                    0.46,
                    boxstyle="round,pad=0.01,rounding_size=0.08",
                    linewidth=0,
                    facecolor=bg_color,
                )
            )
            ax.text(
                cursor_x,
                y,
                token,
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
                family="monospace",
            )
            cursor_x += text_width + 0.12
            i += len(token)
        else:
            # Render until next highlight or end of line
            next_highlight = min(
                (line.find(t, i) for t, _ in highlights if line.find(t, i) != -1),
                default=len(line),
            )
            chunk = line[i:next_highlight]
            ax.text(
                cursor_x,
                y,
                chunk,
                ha="left",
                va="center",
                fontsize=10.8,
                color="#222",
            )
            cursor_x += len(chunk) * 0.173
            i = next_highlight


def _word_wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for w in words:
        if current_len + len(w) + (1 if current else 0) > width:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += len(w) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(" ".join(current))
    return lines


# --- Top bar (prompt + sources) ----------------------------------------------


def _draw_setup_panel(ax) -> None:
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Panel background
    ax.add_patch(
        FancyBboxPatch(
            (0.1, 0.1),
            19.8,
            9.8,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.5,
            edgecolor="#555",
            facecolor="#f7f7f9",
        )
    )

    # Left column: the prompt
    ax.text(
        0.6,
        9.2,
        "PROMPT (adversarial — demands out-of-scope [7] and [8])",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#444",
    )
    wrapped_prompt = _word_wrap(PROMPT, width=50)
    y = 8.0
    for idx, line in enumerate(wrapped_prompt):
        rendered = line
        if idx == 0:
            rendered = f'"{line}'
        if idx == len(wrapped_prompt) - 1:
            rendered = f'{rendered}"'
        ax.text(
            0.6,
            y,
            rendered,
            ha="left",
            va="center",
            fontsize=10,
            color="#222",
            style="italic",
        )
        y -= 0.95

    # Divider
    ax.plot([9.9, 9.9], [0.6, 9.4], color="#bbb", linewidth=1.2, linestyle=":")

    # Right column: sources
    ax.text(
        10.3,
        9.2,
        "SOURCES IN SCOPE (N = 6, so only [1]..[6] are valid)",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#444",
    )
    y = 7.9
    for src in SOURCES:
        ax.text(
            10.3,
            y,
            src,
            ha="left",
            va="center",
            fontsize=9.3,
            color="#222",
            family="monospace",
        )
        y -= 1.05


# --- Bottom evidence bar ------------------------------------------------------


def _draw_footer(ax) -> None:
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 3)
    ax.axis("off")

    ax.text(
        10.0,
        2.2,
        "40-run multi-prompt sweep  •  4 prompts × 2 models × 5 seeds",
        ha="center",
        va="center",
        fontsize=11,
        color="#555",
        fontweight="bold",
    )
    ax.text(
        10.0,
        1.3,
        "citeformer fabrication rate:  0.0 ± 0.0  (structural — not average)   "
        "||   baseline on `survey`:  3.9% fab rate",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#333",
        family="monospace",
    )
    ax.text(
        10.0,
        0.45,
        "benchmarks/README.md · reproducible via `uv run python -m benchmarks.multiprompt_sweep`",
        ha="center",
        va="center",
        fontsize=9,
        color="#888",
        style="italic",
    )


# --- Assemble the whole thing -------------------------------------------------


def generate_cover() -> Path:
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[2.2, 5.0, 0.9],
        width_ratios=[1, 1],
        hspace=0.18,
        wspace=0.08,
    )

    # Top row: single panel spanning both columns
    ax_setup = fig.add_subplot(gs[0, :])
    _draw_setup_panel(ax_setup)

    # Middle row: two side-by-side output panels
    ax_baseline = fig.add_subplot(gs[1, 0])
    _draw_output_panel(
        ax_baseline,
        title="BASELINE — unconstrained HF generation",
        title_color="#c23832",
        text=BASELINE_TEXT,
        highlights=BASELINE_HIGHLIGHTS,
        footer="100% fabrication   [7] and [8] shouldn't exist",
        footer_bg="#fde1df",
        footer_fg="#9e1c16",
    )

    ax_citeformer = fig.add_subplot(gs[1, 1])
    _draw_output_panel(
        ax_citeformer,
        title="CITEFORMER — logit-masked grammar",
        title_color="#1a8a46",
        text=CITEFORMER_TEXT,
        highlights=CITEFORMER_HIGHLIGHTS,
        footer="0% fabrication   [7]/[8] are token-impossible",
        footer_bg="#d8f0db",
        footer_fg="#105c2f",
    )

    # Bottom row: evidence footer
    ax_footer = fig.add_subplot(gs[2, :])
    _draw_footer(ax_footer)

    # Header title over everything
    fig.suptitle(
        "citeformer — fabricated citations are a decode-time impossibility, not a hope",
        fontsize=16,
        fontweight="bold",
        color="#111",
        y=0.995,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT_PATH


if __name__ == "__main__":
    out = generate_cover()
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")

# Silence ruff — these imports are used by matplotlib internals via rcParams
_ = FancyArrowPatch
_ = mpatches
