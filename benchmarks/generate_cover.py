"""Generate tweet-friendly cover + thread images for citeformer.

Three images, all 1200×675 (Twitter's 16:9 inline format) with big text
readable on a phone thumb:

- ``cover-annotated.png`` — the main cover. Big headline, side-by-side
  "WITHOUT citeformer" (AI invents a fake source) vs "WITH citeformer"
  (AI can only pick from real sources). Middle-school-level wording;
  the technical framing sits in a smaller subtitle.
- ``thread-backends.png`` — "works with 7 major AI providers" grid.
  Local vs API columns with provider names at 40pt.
- ``thread-evidence.png`` — the 0/40-run proof in one glance. Giant
  "0" number with the structural callout.

Run::

    uv run python -m benchmarks.generate_cover

Regenerate any time the branding or numbers change.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_FIGURES = Path(__file__).parent / "findings" / "figures"

# Tweet-friendly: 16:9 at 1200×675 renders crisply inline on Twitter /
# LinkedIn / GitHub READMEs. Bigger figsize means bigger fonts relative
# to the frame — which is what we want for phone legibility.
_FIGSIZE = (12.0, 6.75)
_DPI = 100  # 1200×675 at this figsize

# Brand palette — kept consistent across all three images.
_RED = "#c23832"
_RED_BG = "#fde1df"
_GREEN = "#1a8a46"
_GREEN_BG = "#d8f0db"
_INK = "#111"
_MUTED = "#555"
_BG = "#fafafa"


# -----------------------------------------------------------------------------
# Cover — the "fake vs real" side-by-side
# -----------------------------------------------------------------------------


def _draw_cover(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # --- Top headline ---------------------------------------------------------
    ax.text(
        50,
        92,
        "AI can make up citations that don't exist.",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        50,
        85,
        "citeformer makes that physically impossible.",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=_GREEN,
    )

    # --- Setup strip ----------------------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (6, 71),
            88,
            8,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            linewidth=0,
            facecolor="#eef2f7",
        )
    )
    ax.text(
        50,
        75.5,
        "You gave the AI 6 real sources:  [1]   [2]   [3]   [4]   [5]   [6]",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=_INK,
        family="monospace",
    )
    ax.text(
        50,
        72.5,
        "Only these six are valid. Anything else is made up.",
        ha="center",
        va="center",
        fontsize=11,
        color=_MUTED,
        style="italic",
    )

    # --- Left panel: WITHOUT citeformer --------------------------------------
    _draw_quadrant(
        ax,
        x=4,
        y=8,
        w=45,
        h=58,
        accent=_RED,
        bg_accent=_RED_BG,
        title="WITHOUT citeformer",
        subtitle="how LLMs normally behave",
        big_token="[7]",
        big_token_label="← a source that doesn't exist",
        outcome="FAKE — AI made it up",
    )

    # --- Right panel: WITH citeformer ---------------------------------------
    _draw_quadrant(
        ax,
        x=51,
        y=8,
        w=45,
        h=58,
        accent=_GREEN,
        bg_accent=_GREEN_BG,
        title="WITH citeformer",
        subtitle="structurally impossible to invent a source",
        big_token="[3]",
        big_token_label="← one of your real sources",
        outcome="REAL — verifiable",
    )

    # --- Bottom evidence strip -----------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (6, 0.6),
            88,
            5.8,
            boxstyle="round,pad=0.02,rounding_size=0.5",
            linewidth=0,
            facecolor="#2b2b2e",
        )
    )
    ax.text(
        50,
        3.8,
        "Tested across 40 runs.  Zero fakes. Every time.",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#fff",
    )


def _draw_quadrant(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    accent: str,
    bg_accent: str,
    title: str,
    subtitle: str,
    big_token: str,
    big_token_label: str,
    outcome: str,
) -> None:
    """Single panel with a title strip, a giant cite token, and an outcome tag."""
    # Card border
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            linewidth=2.5,
            edgecolor=accent,
            facecolor="white",
        )
    )

    # Title strip
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 10),
            w,
            10,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            linewidth=0,
            facecolor=accent,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 5.5,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + w / 2,
        y + h - 8.8,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
        color="white",
        style="italic",
    )

    # Giant cite token in the middle
    ax.text(
        x + w / 2,
        y + h / 2 + 2,
        big_token,
        ha="center",
        va="center",
        fontsize=72,
        fontweight="bold",
        color=accent,
        family="monospace",
    )
    # Annotation under the token
    ax.text(
        x + w / 2,
        y + h / 2 - 11,
        big_token_label,
        ha="center",
        va="center",
        fontsize=12,
        color=_INK,
        style="italic",
    )

    # Outcome tag at the bottom
    ax.add_patch(
        FancyBboxPatch(
            (x + 4, y + 3),
            w - 8,
            7,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            linewidth=0,
            facecolor=bg_accent,
        )
    )
    ax.text(
        x + w / 2,
        y + 6.5,
        outcome,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=accent,
    )


def _render_cover(path: Path) -> None:
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
    fig = plt.figure(figsize=_FIGSIZE, facecolor=_BG)
    ax = fig.add_axes((0, 0, 1, 1))
    _draw_cover(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, facecolor=_BG)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Backends — "works with 7 major AI providers"
# -----------------------------------------------------------------------------


def _draw_backends(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title
    ax.text(
        50,
        91,
        "Works with 7 major AI providers.",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        50,
        83,
        "Same guarantee across all of them.",
        ha="center",
        va="center",
        fontsize=16,
        color=_MUTED,
        style="italic",
    )

    # Two columns
    _draw_backend_col(
        ax,
        x=6,
        y=12,
        w=43,
        h=64,
        accent="#3a7ac4",
        bg_accent="#e4edf8",
        title="LOCAL",
        subtitle="runs on your computer",
        providers=[
            "HuggingFace transformers",
            "vLLM",
            "llama.cpp",
        ],
        tier="logit-layer enforcement",
    )

    _draw_backend_col(
        ax,
        x=51,
        y=12,
        w=43,
        h=64,
        accent="#a857b0",
        bg_accent="#f2e4f4",
        title="API",
        subtitle="calls the provider's cloud",
        providers=[
            "OpenAI (GPT-4o, ...)",
            "Anthropic (Claude)",
            "Google Gemini",
            "Mistral",
        ],
        tier="schema + provider-native",
    )

    # Footer: the shared result
    ax.add_patch(
        FancyBboxPatch(
            (6, 2),
            88,
            7,
            boxstyle="round,pad=0.02,rounding_size=0.5",
            linewidth=0,
            facecolor=_GREEN_BG,
        )
    )
    ax.text(
        50,
        5.5,
        "All 7 produce the same output type.  Zero fakes across all of them.",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=_GREEN,
    )


def _draw_backend_col(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    accent: str,
    bg_accent: str,
    title: str,
    subtitle: str,
    providers: list[str],
    tier: str,
) -> None:
    # Card
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            linewidth=2,
            edgecolor=accent,
            facecolor="white",
        )
    )

    # Header strip
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 11),
            w,
            11,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            linewidth=0,
            facecolor=accent,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 5.5,
        title,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + w / 2,
        y + h - 9.5,
        subtitle,
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        style="italic",
    )

    # Provider list
    line_y = y + h - 18
    for name in providers:
        ax.text(
            x + w / 2,
            line_y,
            name,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=_INK,
        )
        line_y -= 7

    # Tier tag at bottom
    tag_y = y + 4
    ax.add_patch(
        FancyBboxPatch(
            (x + 4, tag_y),
            w - 8,
            5.5,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            linewidth=0,
            facecolor=bg_accent,
        )
    )
    ax.text(
        x + w / 2,
        tag_y + 2.7,
        tier,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=accent,
    )


def _render_backends(path: Path) -> None:
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
    fig = plt.figure(figsize=_FIGSIZE, facecolor=_BG)
    ax = fig.add_axes((0, 0, 1, 1))
    _draw_backends(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, facecolor=_BG)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Evidence — the 0/40-run result as a single-glance image
# -----------------------------------------------------------------------------


def _draw_evidence(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Giant "0" — the headline number
    ax.text(
        50,
        63,
        "0",
        ha="center",
        va="center",
        fontsize=180,
        fontweight="bold",
        color=_GREEN,
    )
    ax.text(
        50,
        36,
        "fake citations",
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        50,
        28,
        "across 40 benchmark runs — 4 prompts × 2 models × 5 seeds",
        ha="center",
        va="center",
        fontsize=14,
        color=_MUTED,
        style="italic",
    )

    # Bottom callout: it's a structural guarantee, not a stat
    ax.add_patch(
        FancyBboxPatch(
            (10, 10),
            80,
            12,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            linewidth=2,
            edgecolor=_GREEN,
            facecolor=_GREEN_BG,
        )
    )
    ax.text(
        50,
        17.5,
        "it's a structural contract, not an average.",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=_GREEN,
    )
    ax.text(
        50,
        13,
        "the standard deviations are literally zero because there's no variance to measure.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=_INK,
        style="italic",
    )

    # Top tag
    ax.text(
        50,
        93,
        "citeformer benchmark — April 2026",
        ha="center",
        va="center",
        fontsize=12,
        color=_MUTED,
        fontweight="bold",
    )


def _render_evidence(path: Path) -> None:
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
    fig = plt.figure(figsize=_FIGSIZE, facecolor=_BG)
    ax = fig.add_axes((0, 0, 1, 1))
    _draw_evidence(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, facecolor=_BG)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def generate_all() -> list[Path]:
    cover = _FIGURES / "cover-annotated.png"
    backends = _FIGURES / "thread-backends.png"
    evidence = _FIGURES / "thread-evidence.png"
    _render_cover(cover)
    _render_backends(backends)
    _render_evidence(evidence)
    return [cover, backends, evidence]


if __name__ == "__main__":
    for path in generate_all():
        size_kb = path.stat().st_size / 1024
        print(f"wrote {path}  ({size_kb:.0f} KB)")
