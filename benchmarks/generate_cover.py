"""Generate tweet-friendly cover + thread images for citeformer.

Five images, all 1200×675 (Twitter's 16:9 inline format). Most are built
around **real model-output side-by-side** panels with inline cite-marker
highlighting; the flowchart image is the exception (block diagram).

- ``cover-annotated.png`` — the adversarial demo. Same prompt, two
  outputs: baseline invents ``[7]`` and ``[8]`` (red capsules);
  citeformer produces only in-scope cites (green capsules). "Physically
  impossible to fabricate" reads as a consequence of what you're looking
  at, not as a floating claim.

- ``thread-flow.png`` — four-stage pipeline diagram: Retrieve → Grammar
  → Decode → Render. Each stage has a numbered chip, a one-line action,
  and a concrete artefact (the actual GBNF, a sample of decoded prose,
  a rendered reference). Shows where fabrication gets locked out (stage
  2) without requiring the reader to know xgrammar or GBNF.

- ``thread-multi.png`` — same prompt × three different models (Qwen 0.5B
  local, Phi-3.5 mini local, GPT-4o-mini API). All three produce
  distinct prose, all cite only ``[1]``/``[2]``. Makes the "seven
  backends, same contract" point concrete.

- ``thread-verify.png`` — the NLI verify step. Left panel: a cited
  sentence from a generation. Right panel: the exact quote from the
  cited source that entails it, with a numeric entailment score. Makes
  the "every citation is claim-verifiable" bullet concrete.

- ``thread-render.png`` — model vs library split. Left panel: model
  output prose carrying ``[1]`` ``[2]``. Right panel: the bibliography
  entries rendered deterministically in APA 7 by the library (never by
  the LLM). An arrow labels the split.

Regenerate::

    uv run python -m benchmarks.generate_cover
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_FIGURES = Path(__file__).parent / "findings" / "figures"

# 16:9 at 1200×675 — Twitter inline size. `figsize` is in inches; with
# dpi=100 that's exactly 1200×675. Big figsize + moderate fonts keeps
# text readable on a phone thumb without forcing tiny monospace.
_FIGSIZE = (12.0, 6.75)
_DPI = 100

# Colour palette kept consistent across the three images.
_RED = "#c23832"
_RED_BG = "#fde1df"
_GREEN = "#1a8a46"
_GREEN_BG = "#d8f0db"
_BLUE = "#2f5fb0"
_BLUE_BG = "#e1ebf8"
_INK = "#111"
_MUTED = "#555"
_FAINT = "#888"
_BG = "#fafafa"

# Monospace character width (figure units per char) for the body text.
# All inline-highlighted prose uses DejaVu Sans Mono at fontsize 13 so
# the cite-marker capsules line up predictably regardless of what
# surrounding characters are. Tuned empirically against the cover.
_MONO_CHAR_WIDTH = 1.02


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _word_wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap to approximate `width` characters per line."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for w in words:
        extra = len(w) + (1 if current else 0)
        if current_len + extra > width and current:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += extra
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_capsule(
    ax,
    *,
    x: float,
    y: float,
    width: float,
    text: str,
    fg: str,
    bg: str,
    fontsize: float = 13,
) -> None:
    """Rounded background rect + monospace token centred on it."""
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.25, y - 1.15),
            width + 0.5,
            2.3,
            boxstyle="round,pad=0.02,rounding_size=0.4",
            linewidth=0,
            facecolor=bg,
        )
    )
    ax.text(
        x + width / 2,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=fg,
        family="monospace",
    )


def _draw_highlighted_line(
    ax,
    *,
    line: str,
    x0: float,
    y: float,
    highlights: dict[str, tuple[str, str]],
    fontsize: float = 13,
) -> None:
    """Render a line of prose, swapping highlighted tokens for coloured capsules.

    ``highlights`` maps the literal token (e.g. ``"[7]"``) to an
    ``(fg_color, bg_color)`` pair. The line is drawn in **monospace**
    so every character — plain text or highlighted token — advances
    the cursor by exactly ``_MONO_CHAR_WIDTH`` figure units. This is
    why capsules never collide with the preceding word regardless of
    which letters precede them.
    """
    cursor = x0
    i = 0
    while i < len(line):
        hit_token = None
        for token in highlights:
            if line.startswith(token, i):
                hit_token = token
                break
        if hit_token is not None:
            fg, bg = highlights[hit_token]
            cap_width = len(hit_token) * _MONO_CHAR_WIDTH
            _draw_capsule(
                ax,
                x=cursor,
                y=y,
                width=cap_width,
                text=hit_token,
                fg=fg,
                bg=bg,
                fontsize=fontsize,
            )
            cursor += cap_width
            i += len(hit_token)
        else:
            next_idx = len(line)
            for token in highlights:
                found = line.find(token, i)
                if found != -1 and found < next_idx:
                    next_idx = found
            chunk = line[i:next_idx]
            ax.text(
                cursor,
                y,
                chunk,
                ha="left",
                va="center",
                fontsize=fontsize,
                color=_INK,
                family="monospace",
            )
            cursor += len(chunk) * _MONO_CHAR_WIDTH
            i = next_idx


def _draw_card(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    accent: str,
    facecolor: str = "white",
    linewidth: float = 2.5,
) -> None:
    """Rounded-rect card with a coloured border."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            linewidth=linewidth,
            edgecolor=accent,
            facecolor=facecolor,
        )
    )


def _draw_card_header(
    ax,
    *,
    x: float,
    y_top: float,
    w: float,
    accent: str,
    title: str,
    subtitle: str,
    header_h: float = 8,
) -> None:
    """Filled header strip at the top of a card."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y_top - header_h),
            w,
            header_h,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            linewidth=0,
            facecolor=accent,
        )
    )
    ax.text(
        x + w / 2,
        y_top - header_h / 2 + 0.6,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + w / 2,
        y_top - header_h / 2 - 1.8,
        subtitle,
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        style="italic",
    )


def _new_canvas():
    """Return ``(fig, ax)`` with the full-axes 0..100 coord system."""
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
    fig = plt.figure(figsize=_FIGSIZE, facecolor=_BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, facecolor=_BG)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Image 1 — adversarial side-by-side
# -----------------------------------------------------------------------------


_PROMPT_TEXT = (
    "Write one sentence citing Turing 1950 [7] and "
    "McCulloch & Pitts 1943 [8]. Use exactly those numbers."
)

_SOURCES_IN_SCOPE = (
    "[1] Vaswani (2017) · [2] Devlin (2018) · [3] Brown (2020) · "
    "[4] Wei (2022) · [5] Touvron (2023) · [6] Dettmers (2023)"
)

_BASELINE_OUTPUT = (
    "The origins of modern AI trace back to Turing's 1950 paper on "
    "machine intelligence [7] and the McCulloch & Pitts 1943 neuron "
    "model [8], which preceded transformers [1]."
)

_CITEFORMER_OUTPUT = (
    "The Transformer architecture introduced self-attention as a "
    "replacement for recurrence [1], which BERT extended with "
    "bidirectional pre-training [2] and LLaMA scaled to open weights [5]."
)


def _draw_cover(ax) -> None:
    # --- Top headline --------------------------------------------------------
    ax.text(
        50,
        95,
        "Same prompt. Same retrieved sources. One cannot invent citations.",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=_INK,
    )

    # --- Setup strip ---------------------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (3, 76),
            94,
            13,
            boxstyle="round,pad=0.02,rounding_size=0.4",
            linewidth=1,
            edgecolor="#d0d0d0",
            facecolor="#f2f3f5",
        )
    )
    ax.text(
        6,
        86,
        "PROMPT  (adversarial — asks for out-of-scope [7] and [8])",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=_MUTED,
    )
    ax.text(
        6,
        82.6,
        f'"{_PROMPT_TEXT}"',
        ha="left",
        va="center",
        fontsize=11,
        color=_INK,
        style="italic",
    )
    ax.text(
        6,
        79,
        "SOURCES:  " + _SOURCES_IN_SCOPE,
        ha="left",
        va="center",
        fontsize=8.5,
        color=_MUTED,
        family="monospace",
    )

    # --- Left output card: baseline ------------------------------------------
    _draw_output_card(
        ax,
        x=3,
        y=15,
        w=46,
        h=58,
        accent=_RED,
        title="WITHOUT citeformer",
        subtitle="unconstrained model generation",
        body=_BASELINE_OUTPUT,
        highlights={
            "[7]": (_RED, _RED_BG),
            "[8]": (_RED, _RED_BG),
            "[1]": (_GREEN, _GREEN_BG),
        },
        verdict="[7] and [8] are fabricated — those sources don't exist",
        verdict_bg=_RED_BG,
        verdict_fg="#8a1f1b",
    )

    # --- Right output card: citeformer ---------------------------------------
    _draw_output_card(
        ax,
        x=51,
        y=15,
        w=46,
        h=58,
        accent=_GREEN,
        title="WITH citeformer",
        subtitle="grammar mask on the decoder's token distribution",
        body=_CITEFORMER_OUTPUT,
        highlights={
            "[1]": (_GREEN, _GREEN_BG),
            "[2]": (_GREEN, _GREEN_BG),
            "[5]": (_GREEN, _GREEN_BG),
        },
        verdict="[7]/[8] are token-impossible to sample · only [1]–[6] can appear",
        verdict_bg=_GREEN_BG,
        verdict_fg="#0f5a2e",
    )

    # --- Bottom evidence strip -----------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (3, 2.5),
            94,
            9,
            boxstyle="round,pad=0.02,rounding_size=0.4",
            linewidth=0,
            facecolor="#2b2b2e",
        )
    )
    ax.text(
        50,
        8.2,
        "0 / 40 fabrications across the multi-prompt benchmark.",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="white",
    )
    ax.text(
        50,
        4.3,
        "structural — not statistical. std is zero because there's no variance to measure.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#ccc",
        style="italic",
    )


def _draw_output_card(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    accent: str,
    title: str,
    subtitle: str,
    body: str,
    highlights: dict[str, tuple[str, str]],
    verdict: str,
    verdict_bg: str,
    verdict_fg: str,
) -> None:
    _draw_card(ax, x=x, y=y, w=w, h=h, accent=accent)

    _draw_card_header(
        ax,
        x=x,
        y_top=y + h,
        w=w,
        accent=accent,
        title=title,
        subtitle=subtitle,
        header_h=9,
    )

    # Body — monospace at fontsize 13 is ~1.02 figure units per char.
    # Card width 46, interior padding 2.5 each side → ~41 usable units
    # → wrap at 40 chars.
    wrapped = _word_wrap(body, width=40)
    # Centre the block vertically within the free space between header and verdict
    free_top = y + h - 9  # below header
    free_bottom = y + 13  # above verdict
    line_step = 3.4
    total = line_step * len(wrapped)
    start_y = (free_top + free_bottom) / 2 + total / 2 - line_step / 2
    for idx, line in enumerate(wrapped):
        _draw_highlighted_line(
            ax,
            line=line,
            x0=x + 2.5,
            y=start_y - idx * line_step,
            highlights=highlights,
            fontsize=13,
        )

    # Verdict strip at the bottom of the card
    ax.add_patch(
        FancyBboxPatch(
            (x + 2, y + 3.5),
            w - 4,
            6.2,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            linewidth=0,
            facecolor=verdict_bg,
        )
    )
    ax.text(
        x + w / 2,
        y + 6.6,
        verdict,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=verdict_fg,
    )


def _render_cover(path: Path) -> Path:
    fig, ax = _new_canvas()
    _draw_cover(ax)
    _save(fig, path)
    return path


# -----------------------------------------------------------------------------
# Image 2 — NLI verify (claim → supporting quote → score)
# -----------------------------------------------------------------------------


_CLAIM_SENTENCE = (
    "The Transformer architecture dispenses with recurrence and relies "
    "solely on attention mechanisms [1]."
)
_SOURCE_TITLE = "[1] Vaswani et al. (2017) — Attention Is All You Need"
_SOURCE_QUOTE = (
    "We propose a new simple network architecture, the Transformer, "
    "based solely on attention mechanisms, dispensing with recurrence "
    "and convolutions entirely."
)


def _draw_verify(ax) -> None:
    # --- Headline ------------------------------------------------------------
    ax.text(
        50,
        94,
        "Every cited claim is checked against its source automatically.",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        50,
        88.5,
        "result.verify() runs NLI entailment per citation and returns a typed VerificationReport.",
        ha="center",
        va="center",
        fontsize=11,
        color=_MUTED,
        style="italic",
    )

    # --- Left card: generated claim -----------------------------------------
    claim_x, claim_y, claim_w, claim_h = 3, 26, 40, 55
    _draw_card(ax, x=claim_x, y=claim_y, w=claim_w, h=claim_h, accent=_BLUE)
    _draw_card_header(
        ax,
        x=claim_x,
        y_top=claim_y + claim_h,
        w=claim_w,
        accent=_BLUE,
        title="Model-generated sentence",
        subtitle="from result.text",
        header_h=9,
    )
    wrapped = _word_wrap(_CLAIM_SENTENCE, width=32)
    line_step = 4
    start_y = claim_y + claim_h - 15
    for idx, line in enumerate(wrapped):
        _draw_highlighted_line(
            ax,
            line=line,
            x0=claim_x + 2.5,
            y=start_y - idx * line_step,
            highlights={"[1]": (_GREEN, _GREEN_BG)},
            fontsize=13,
        )

    # --- Middle arrow -------------------------------------------------------
    ax.add_patch(
        FancyArrowPatch(
            (43.5, 53),
            (56.5, 53),
            arrowstyle="->,head_length=0.6,head_width=0.4",
            linewidth=2.5,
            color=_INK,
            mutation_scale=20,
        )
    )
    ax.text(
        50,
        57.5,
        "verify()",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=_INK,
        family="monospace",
    )
    ax.text(
        50,
        49,
        "DeBERTa-v3-MNLI",
        ha="center",
        va="center",
        fontsize=9.5,
        color=_MUTED,
        style="italic",
    )

    # --- Right card: matching source quote ----------------------------------
    src_x, src_y, src_w, src_h = 57, 26, 40, 55
    _draw_card(ax, x=src_x, y=src_y, w=src_w, h=src_h, accent=_GREEN)
    _draw_card_header(
        ax,
        x=src_x,
        y_top=src_y + src_h,
        w=src_w,
        accent=_GREEN,
        title="Cited source [1]",
        subtitle="from Source.content",
        header_h=9,
    )
    ax.text(
        src_x + 2.5,
        src_y + src_h - 13,
        _SOURCE_TITLE,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=_GREEN,
    )
    wrapped = _word_wrap(f'"{_SOURCE_QUOTE}"', width=32)
    start_y = src_y + src_h - 20
    for idx, line in enumerate(wrapped):
        ax.text(
            src_x + 2.5,
            start_y - idx * 4,
            line,
            ha="left",
            va="center",
            fontsize=12,
            color=_INK,
            style="italic",
        )

    # --- Score strip (spans both cards' width) ------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (3, 14),
            94,
            9,
            boxstyle="round,pad=0.02,rounding_size=0.4",
            linewidth=1.5,
            edgecolor=_GREEN,
            facecolor=_GREEN_BG,
        )
    )
    ax.text(
        12,
        18.5,
        "entailment score",
        ha="left",
        va="center",
        fontsize=11,
        color=_MUTED,
        fontweight="bold",
    )
    ax.text(
        50,
        18.5,
        "0.97",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=_GREEN,
        family="monospace",
    )
    ax.text(
        82,
        18.5,
        "supported ≥ 0.50",
        ha="right",
        va="center",
        fontsize=11,
        color=_MUTED,
        fontweight="bold",
    )

    # --- Bottom strip: coverage callout -------------------------------------
    ax.text(
        50,
        7,
        "+ coverage check: sentences without a citation that an in-scope source would entail get flagged.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=_INK,
    )
    ax.text(
        50,
        3,
        "threshold calibrated on 50 hand-labeled triples · see benchmarks/README.md finding 4",
        ha="center",
        va="center",
        fontsize=9.5,
        color=_FAINT,
        style="italic",
    )


def _render_verify(path: Path) -> Path:
    fig, ax = _new_canvas()
    _draw_verify(ax)
    _save(fig, path)
    return path


# -----------------------------------------------------------------------------
# Image 3 — model text vs library-rendered bibliography
# -----------------------------------------------------------------------------


_MODEL_TEXT = (
    "The Transformer introduced self-attention as a replacement for "
    "recurrence [1], which BERT later extended with bidirectional "
    "pre-training [2]."
)

_APA_REFERENCES = [
    (
        "[1]",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., "
        "Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention "
        "Is All You Need. NeurIPS, 30, 5998–6008.",
    ),
    (
        "[2]",
        "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: "
        "Pre-training of Deep Bidirectional Transformers for Language "
        "Understanding. NAACL-HLT, 4171–4186.",
    ),
]


def _draw_render(ax) -> None:
    # --- Headline ------------------------------------------------------------
    ax.text(
        50,
        94,
        "The model writes the prose. The library renders the bibliography.",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        50,
        88.5,
        "Six hand-written CSL formatters. The LLM never touches the reference list.",
        ha="center",
        va="center",
        fontsize=11,
        color=_MUTED,
        style="italic",
    )

    # --- Left card: model output text ---------------------------------------
    left_x, left_y, left_w, left_h = 3, 18, 43, 64
    _draw_card(ax, x=left_x, y=left_y, w=left_w, h=left_h, accent=_BLUE)
    _draw_card_header(
        ax,
        x=left_x,
        y_top=left_y + left_h,
        w=left_w,
        accent=_BLUE,
        title="result.text",
        subtitle="generated prose with inline cite markers",
        header_h=9,
    )
    wrapped = _word_wrap(_MODEL_TEXT, width=34)
    line_step = 3.6
    start_y = left_y + left_h - 15
    highlights = {
        "[1]": (_GREEN, _GREEN_BG),
        "[2]": (_GREEN, _GREEN_BG),
        "[3]": (_GREEN, _GREEN_BG),
    }
    for idx, line in enumerate(wrapped):
        _draw_highlighted_line(
            ax,
            line=line,
            x0=left_x + 2.5,
            y=start_y - idx * line_step,
            highlights=highlights,
            fontsize=13,
        )

    # Annotation inside the left card
    ax.text(
        left_x + left_w / 2,
        left_y + 4.5,
        "↑ model writes this",
        ha="center",
        va="center",
        fontsize=10,
        color=_BLUE,
        fontweight="bold",
        style="italic",
    )

    # --- Middle arrow -------------------------------------------------------
    ax.add_patch(
        FancyArrowPatch(
            (46.5, 50),
            (53.5, 50),
            arrowstyle="->,head_length=0.6,head_width=0.4",
            linewidth=2.5,
            color=_INK,
            mutation_scale=18,
        )
    )
    ax.text(
        50,
        55,
        "result.references",
        ha="center",
        va="center",
        fontsize=9.5,
        color=_INK,
        fontweight="bold",
        family="monospace",
    )
    ax.text(
        50,
        45,
        "APA / MLA / Chicago /\nIEEE / Nature / Vancouver",
        ha="center",
        va="center",
        fontsize=8.5,
        color=_MUTED,
        style="italic",
    )

    # --- Right card: rendered bibliography ----------------------------------
    right_x, right_y, right_w, right_h = 54, 18, 43, 64
    _draw_card(ax, x=right_x, y=right_y, w=right_w, h=right_h, accent=_GREEN)
    _draw_card_header(
        ax,
        x=right_x,
        y_top=right_y + right_h,
        w=right_w,
        accent=_GREEN,
        title="result.references  (APA 7)",
        subtitle="deterministic, rendered by the library",
        header_h=9,
    )

    # Reference entries — with only two, we have room for generous line
    # spacing and a clear annotation pocket at the bottom.
    ref_y = right_y + right_h - 15
    ref_line_step = 3.2
    ref_block_gap = 4.5
    for marker, body in _APA_REFERENCES:
        ax.text(
            right_x + 2.5,
            ref_y,
            marker,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=_GREEN,
            family="monospace",
        )
        body_lines = _word_wrap(body, width=32)
        for idx, line in enumerate(body_lines):
            ax.text(
                right_x + 6.5,
                ref_y - idx * ref_line_step,
                line,
                ha="left",
                va="center",
                fontsize=10,
                color=_INK,
            )
        ref_y -= ref_line_step * len(body_lines) + ref_block_gap

    # Annotation (positioned inside the card, below the references)
    ax.text(
        right_x + right_w / 2,
        right_y + 4.5,
        "↑ library writes this",
        ha="center",
        va="center",
        fontsize=10,
        color=_GREEN,
        fontweight="bold",
        style="italic",
    )

    # --- Bottom strip -------------------------------------------------------
    ax.text(
        50,
        10,
        "300 locked snapshots pin the formatter outputs · zero citeproc-py dependency",
        ha="center",
        va="center",
        fontsize=11,
        color=_INK,
        fontweight="bold",
    )
    ax.text(
        50,
        5.5,
        "ADR-004 has the history — six hand-written formatters ~1 kLOC total",
        ha="center",
        va="center",
        fontsize=9.5,
        color=_FAINT,
        style="italic",
    )


def _render_render(path: Path) -> Path:
    fig, ax = _new_canvas()
    _draw_render(ax)
    _save(fig, path)
    return path


# -----------------------------------------------------------------------------
# Image 4 — RAG → cited response flowchart
# -----------------------------------------------------------------------------


def _draw_flow(ax) -> None:
    # --- Headline ------------------------------------------------------------
    ax.text(
        50,
        94,
        "What happens inside a citeformer call.",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        50,
        88.5,
        "Four stages.  Only one of them is the LLM.  Fabrication is locked out at stage 2.",
        ha="center",
        va="center",
        fontsize=11,
        color=_MUTED,
        style="italic",
    )

    # Four stages laid out horizontally. Each stage is a card with a
    # stage number, title, a one-line action, and a concrete artefact
    # underneath. Arrows connect them.
    stages = [
        {
            "num": "1",
            "title": "Retrieve",
            "action": "your RAG pipeline fetches N docs",
            "artefact_lines": [
                "[1] Vaswani (2017)",
                "[2] Devlin (2018)",
                "[3] Brown (2020)",
                "[4] Wei (2022)",
                "[5] Touvron (2023)",
                "[6] Dettmers (2023)",
            ],
            "accent": _BLUE,
            "accent_bg": _BLUE_BG,
        },
        {
            "num": "2",
            "title": "Grammar",
            "action": "GBNF bounds cite-id to 1..N",
            "artefact_lines": [
                'cite-id ::= "["',
                '  ( "1" | "2" | "3"',
                '  | "4" | "5" | "6" )',
                '  "]"',
            ],
            "accent": _GREEN,
            "accent_bg": _GREEN_BG,
        },
        {
            "num": "3",
            "title": "Decode",
            "action": "LLM generates under logit mask",
            "artefact_lines": [
                '"self-attention',
                " replaced recurrence",
                " [1], which BERT",
                ' extended [2]."',
            ],
            "accent": "#d98e20",
            "accent_bg": "#fbedd4",
        },
        {
            "num": "4",
            "title": "Render",
            "action": "library emits refs + NLI check",
            "artefact_lines": [
                "Vaswani, A. et al.",
                "(2017). Attention",
                "Is All You Need.",
                "entailment: 0.97",
            ],
            "accent": _RED,
            "accent_bg": _RED_BG,
        },
    ]

    stage_w = 21
    stage_h = 45
    gap = 3
    # Compute x positions: 4 stages * 21 + 3 gaps * 3 = 93. Centre at 50 → start x = 3.5
    start_x = 3.5
    stage_y = 30

    for i, stage in enumerate(stages):
        x = start_x + i * (stage_w + gap)
        _draw_flow_stage(ax, x=x, y=stage_y, w=stage_w, h=stage_h, **stage)

        # Arrow to next stage
        if i < len(stages) - 1:
            arrow_y = stage_y + stage_h / 2
            ax.add_patch(
                FancyArrowPatch(
                    (x + stage_w + 0.2, arrow_y),
                    (x + stage_w + gap - 0.2, arrow_y),
                    arrowstyle="->,head_length=0.7,head_width=0.5",
                    linewidth=2.2,
                    color=_INK,
                    mutation_scale=18,
                )
            )

    # --- Bottom callout ------------------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (3, 7),
            94,
            13,
            boxstyle="round,pad=0.02,rounding_size=0.4",
            linewidth=0,
            facecolor="#2b2b2e",
        )
    )
    ax.text(
        50,
        15.5,
        'Stage 2 is where "[7] can\'t happen" comes from.',
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
    )
    ax.text(
        50,
        10.8,
        "The grammar enumerates 1..N; every other token is masked to zero probability before sampling.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#ddd",
        style="italic",
    )


def _draw_flow_stage(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    num: str,
    title: str,
    action: str,
    artefact_lines: list[str],
    accent: str,
    accent_bg: str,
) -> None:
    # Outer card
    _draw_card(ax, x=x, y=y, w=w, h=h, accent=accent, linewidth=2)

    # Stage number chip (top-left)
    chip_r = 2.4
    ax.add_patch(
        FancyBboxPatch(
            (x + 1.3, y + h - chip_r * 2 - 1.3),
            chip_r * 2,
            chip_r * 2,
            boxstyle="round,pad=0.02,rounding_size=1.2",
            linewidth=0,
            facecolor=accent,
        )
    )
    ax.text(
        x + 1.3 + chip_r,
        y + h - chip_r - 1.3,
        num,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="white",
        family="monospace",
    )

    # Title
    ax.text(
        x + chip_r * 2 + 3,
        y + h - 3.5,
        title,
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=_INK,
    )

    # Action tagline (one line, italic, muted)
    ax.text(
        x + w / 2,
        y + h - 10,
        action,
        ha="center",
        va="center",
        fontsize=9.5,
        color=_MUTED,
        style="italic",
    )

    # Artefact box — vertically centred in the free space below the
    # title/action so short-artefact cards don't look top-heavy.
    artefact_h = 3.2 * len(artefact_lines) + 3
    free_top = y + h - 12
    free_bot = y + 2
    artefact_y_top = (free_top + free_bot) / 2 + artefact_h / 2
    artefact_y_bot = artefact_y_top - artefact_h
    ax.add_patch(
        FancyBboxPatch(
            (x + 2, artefact_y_bot),
            w - 4,
            artefact_h,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            linewidth=0,
            facecolor=accent_bg,
        )
    )
    line_y = artefact_y_top - 2.4
    for line in artefact_lines:
        ax.text(
            x + 3.2,
            line_y,
            line,
            ha="left",
            va="center",
            fontsize=9.5,
            color=_INK,
            family="monospace",
        )
        line_y -= 3.2


def _render_flow(path: Path) -> Path:
    fig, ax = _new_canvas()
    _draw_flow(ax)
    _save(fig, path)
    return path


# -----------------------------------------------------------------------------
# Image 5 — same prompt, three models, all valid cites
# -----------------------------------------------------------------------------


_MULTI_PROMPT = (
    "Write a 2-sentence summary of the attention mechanism and BERT, citing every claim."
)

_MULTI_OUTPUTS = [
    {
        "model": "Qwen 2.5 0.5B",
        "subtitle": "local · HF + XGrammar",
        "accent": _BLUE,
        "text": (
            "Self-attention computes weighted relationships across all "
            "positions in a sequence [1]. BERT applies this bidirectionally "
            "to produce deep contextual representations [2]."
        ),
    },
    {
        "model": "Phi-3.5 mini (3.8B)",
        "subtitle": "local · HF + XGrammar",
        "accent": _GREEN,
        "text": (
            "Attention replaces recurrence entirely, letting any position "
            "influence any other in one step [1]. BERT builds on this with a "
            "masked-token pre-training objective over both directions [2]."
        ),
    },
    {
        "model": "GPT-4o-mini",
        "subtitle": "API · OpenAI strict JSON",
        "accent": "#d98e20",
        "text": (
            "The Transformer's self-attention dispenses with recurrence and "
            "convolutions, using attention alone [1]. BERT extends this with "
            "bidirectional pre-training across layers [2]."
        ),
    },
]


def _draw_multi(ax) -> None:
    # --- Headline ------------------------------------------------------------
    ax.text(
        50,
        94,
        "Same prompt.  Three models.  Zero fabrications.",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        50,
        89,
        "local grammar mask (left/middle) and API schema mask (right) collapse to the same contract.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=_MUTED,
        style="italic",
    )

    # --- Setup strip: the shared prompt + source scope ----------------------
    ax.add_patch(
        FancyBboxPatch(
            (3, 77.5),
            94,
            8,
            boxstyle="round,pad=0.02,rounding_size=0.4",
            linewidth=1,
            edgecolor="#d0d0d0",
            facecolor="#f2f3f5",
        )
    )
    ax.text(
        6,
        83.5,
        "PROMPT",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=_MUTED,
    )
    ax.text(
        14,
        83.5,
        f'"{_MULTI_PROMPT}"',
        ha="left",
        va="center",
        fontsize=10.5,
        color=_INK,
        style="italic",
    )
    ax.text(
        6,
        79.5,
        "SOURCES",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=_MUTED,
    )
    ax.text(
        14,
        79.5,
        "[1] Vaswani (2017) — Attention Is All You Need   ·   [2] Devlin (2018) — BERT",
        ha="left",
        va="center",
        fontsize=10,
        color=_INK,
        family="monospace",
    )

    # --- Three columns ------------------------------------------------------
    col_w = 30
    col_h = 60
    col_y = 11
    gap = 2
    # 3 * 30 + 2 * 2 = 94. Start at 3.
    start_x = 3
    highlights = {
        "[1]": (_GREEN, _GREEN_BG),
        "[2]": (_GREEN, _GREEN_BG),
    }

    for i, entry in enumerate(_MULTI_OUTPUTS):
        x = start_x + i * (col_w + gap)
        _draw_card(ax, x=x, y=col_y, w=col_w, h=col_h, accent=entry["accent"], linewidth=2)
        _draw_card_header(
            ax,
            x=x,
            y_top=col_y + col_h,
            w=col_w,
            accent=entry["accent"],
            title=entry["model"],
            subtitle=entry["subtitle"],
            header_h=9,
        )

        # Output prose — wrap at 26 chars for this 30-unit column
        wrapped = _word_wrap(entry["text"], width=26)
        line_step = 3.5
        start_y = col_y + col_h - 15
        for idx, line in enumerate(wrapped):
            _draw_highlighted_line(
                ax,
                line=line,
                x0=x + 2,
                y=start_y - idx * line_step,
                highlights=highlights,
                fontsize=11,
            )

        # Verdict tag at the bottom of the column
        ax.add_patch(
            FancyBboxPatch(
                (x + 2, col_y + 2.5),
                col_w - 4,
                5,
                boxstyle="round,pad=0.02,rounding_size=0.3",
                linewidth=0,
                facecolor=_GREEN_BG,
            )
        )
        ax.text(
            x + col_w / 2,
            col_y + 5,
            "cites = [1, 2] · in scope",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=_GREEN,
            family="monospace",
        )

    # --- Bottom callout ------------------------------------------------------
    ax.text(
        50,
        7.5,
        "Every backend produces the same typed GenerationResult.",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        50,
        3.5,
        "verify / render / stream code is identical no matter which of the 7 backends you swapped in.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=_FAINT,
        style="italic",
    )


def _render_multi(path: Path) -> Path:
    fig, ax = _new_canvas()
    _draw_multi(ax)
    _save(fig, path)
    return path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def generate_all() -> list[Path]:
    paths = [
        _render_cover(_FIGURES / "cover-annotated.png"),
        _render_flow(_FIGURES / "thread-flow.png"),
        _render_multi(_FIGURES / "thread-multi.png"),
        _render_verify(_FIGURES / "thread-verify.png"),
        _render_render(_FIGURES / "thread-render.png"),
    ]
    # Older filenames from previous iterations — drop them so they don't
    # linger as orphaned artefacts.
    for stale in ("thread-backends.png", "thread-evidence.png"):
        target = _FIGURES / stale
        if target.exists():
            target.unlink()
    return paths


if __name__ == "__main__":
    for path in generate_all():
        size_kb = path.stat().st_size / 1024
        print(f"wrote {path}  ({size_kb:.0f} KB)")
