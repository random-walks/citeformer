# ADR-016 — Defer fine-grain windowing for `verify()` until calibration data exists

- **Status**: Decided (2026-04-25); deferred to v0.4+ (data-gated).

## Context

ADR-013 added `cited_text` preservation on `Citation` and the current
PR threads it into `verify()` as the NLI premise — when Anthropic
populated `Citation.cited_text`, NLI scores against that span instead
of the whole `Source.content`. This is the high-leverage change: long
documents whose relevant assertion was buried past DeBERTa's 512-token
horizon now get scored against just the cited span.

A natural follow-up would be **fine-grain windowing**: instead of
scoring against just `cited_text`, score against `cited_text` AND a
small surrounding window (e.g. ±200 tokens of context), then take the
max entailment. The intuition: the cited span sometimes lacks the
referent (a pronoun, a definition introduced earlier) and the
surrounding context disambiguates.

## Decision

**Defer until we have calibration data.**

Reasons:

1. **The current `cited_text`-as-premise change is itself
   uncalibrated.** We haven't yet measured whether scoring against
   the cited span produces materially different `support_rate` numbers
   vs. scoring against the full source. ADR-013 is the obvious right
   move, but the magnitude of the win is empirical, not theoretical.
2. **Adding a window without measuring first is a knob with no signal.**
   The hyperparameters multiply: window size (50 / 200 / 500 tokens),
   reduction strategy (max / mean / sum-log-prob), inclusion criteria
   (always windowed vs. only when cited_text < N tokens). Picking a
   default without data is just adding code to `tune_threshold.py`'s
   surface area.
3. **DeBERTa-v3-MNLI is bimodal** (benchmarks finding 4 — see
   `benchmarks/README.md` in the repo root for the full data)
   — scores cluster at ~0 and ~1 with little in the middle. Windowing
   matters only if the in-the-middle pairs are where the action is;
   on a bimodal distribution, the win is small.

## What would unlock it

- **A real benchmark run** with `cited_text`-as-premise vs. windowed-
  premise on the existing 50-triple calibration set, showing the
  per-window-size F1 deltas. If F1 moves >2% on the calibrated
  threshold for any window choice, ship it. If not, the complexity
  isn't worth it.
- **A user report** of a real false-positive / false-negative case
  that windowing would have caught.

Either signal makes the work obvious. Without them, the windowing
code is a tuning knob in search of a problem.

## Consequences

- `verify/entailment.py` stays at the current "use cited_text if
  present, else full source content" behaviour.
- A note in `benchmarks/README.md` flags this as a tuning-direction
  candidate so anyone running the calibration suite is reminded to
  produce the data we'd need.
- No code change in this PR beyond the existing ADR-013 work.
