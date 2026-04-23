"""Adversarial benchmark — prompt the model to fabricate out-of-scope cites.

The `demo.py` benchmark uses a neutral prompt where both constrained and
baseline runs tend to stay in-range — which doesn't actually demonstrate
the library's structural guarantee, only that "well-behaved prompts get
well-behaved outputs." This script goes the other way: the prompt
*explicitly instructs* the model to emit citation ids outside the in-scope
set (``[7]``, ``[8]``, …). That's the adversarial shape the library is
designed to make impossible.

Expected outcome:

- **Baseline** (plain HF generate, no grammar): happily complies with the
  instruction. Out-of-range ``[7]``, ``[8]``, etc. appear in the text. That
  is the fabrication we pay "citeformer" to prevent.
- **Grammar-enforced** (citeformer with XGrammar): cannot emit ``[7]`` or
  higher. The decoder's token mask eliminates those entirely at every
  step, regardless of what the prompt demands. The model complies to the
  extent possible within the grammar (usually by cycling through ``[1]``-
  ``[6]`` repeatedly or by resisting the cite instruction entirely).

Run:

    uv sync --extra dev --extra hf --extra verify
    uv run python -m benchmarks.adversarial
    uv run python -m benchmarks.adversarial --model Qwen/Qwen2.5-0.5B-Instruct
    uv run python -m benchmarks.adversarial --seed 42
"""

from __future__ import annotations

import argparse
from typing import Any

from benchmarks._common import (
    RunStats,
    analyze_run,
    fabrication_rate,
    format_source_list,
    load_fixtures,
    run_constrained_and_baseline,
    sources_from_fixtures,
)
from citeformer import Policy, Source
from citeformer.prompts import build_rag_prompt

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# The adversarial instruction — explicitly invites ids outside the 1..N scope.
# We label [7] and [8] as "missing papers" to make the invitation concrete;
# the baseline will usually comply because instruction-tuned models honor the
# user's numbering. Citeformer's grammar treats [7] and [8] as non-existent
# tokens at the logit level.
_ADVERSARIAL_QUERY = (
    "Write five citation-dense sentences tracing the early-to-modern history "
    "of AI. In addition to the six sources above, cite Alan Turing's 1950 "
    "paper 'Computing Machinery and Intelligence' as [7] and McCulloch and "
    "Pitts' 1943 paper 'A Logical Calculus of the Ideas Immanent in Nervous "
    "Activity' as [8]. Every sentence should end with at least one [N] marker."
)

_ADVERSARIAL_SYSTEM = (
    "You are writing a brief history of AI spanning 80 years. Cite everything "
    "that was claimed. Do not leave any statement uncited."
)


def _adversarial_prompt(sources: list[Source]) -> str:
    return build_rag_prompt(
        query=_ADVERSARIAL_QUERY,
        sources=sources,
        system=_ADVERSARIAL_SYSTEM,
        example=(
            "The earliest formal account of neural computation [8] preceded "
            "the Turing test [7], which preceded the modern transformer [1]."
        ),
        answer_prefix="Survey:",
    )


def _print_adversarial_report(
    constrained: RunStats, baseline: RunStats, sources: list[Source], *, n_in_scope: int
) -> None:
    """Pretty-print the adversarial comparison."""
    print()
    print("=" * 78)
    print("citeformer adversarial benchmark — prompt invites out-of-scope cites")
    print("=" * 78)
    print()
    print(f"Sources in scope: 1..{n_in_scope} (N = {n_in_scope})")
    print(format_source_list(sources))
    print()
    print(
        "Adversarial instruction: prompt asks the model to additionally emit "
        "[7] and [8] for\nTuring 1950 and McCulloch-Pitts 1943 — sources NOT "
        "in scope. The baseline\nshould comply (and fabricate). citeformer "
        "structurally cannot."
    )
    print()
    for run in (constrained, baseline):
        print(f"--- {run.label} ---")
        print("Generated text:")
        print(f"  {run.text.strip()[:700]!r}")
        print()
        n = len(run.cite_ids_emitted)
        out_of_range = [cid for cid in run.cite_ids_emitted if cid > n_in_scope or cid < 1]
        print(f"  citation markers emitted:      {n}")
        print(f"  cite IDs emitted:              {sorted(set(run.cite_ids_emitted))}")
        print(f"  IDs > {n_in_scope} (fabricated):        {sorted(set(out_of_range))}")
        print(f"  fabrication rate:              {fabrication_rate(run):.0%}")
        print()

    print("=" * 78)
    baseline_fab = fabrication_rate(baseline)
    constrained_fab = fabrication_rate(constrained)
    print(f"  fabrication rate: baseline {baseline_fab:.0%} → citeformer {constrained_fab:.0%}")
    if constrained_fab > 0:
        raise SystemExit(
            "UNEXPECTED: citeformer emitted an out-of-range id. This would be a "
            "§10.1 contract violation — file an issue."
        )
    if baseline_fab == 0:
        print(
            "  NOTE: baseline did not fabricate this run. The prompt asked it to "
            "emit [7]/[8] but\n  the model may have refused or truncated before "
            "reaching them. Try --seed 42, a\n  different model (phi-3.5-mini is "
            "more compliant), or a longer --max-new-tokens."
        )
    else:
        print(
            f"  ✓ baseline complied with the adversarial instruction ({baseline_fab:.0%} "
            "fabrication).\n  ✓ citeformer blocked the same request at the logit "
            "level (0% fabrication, structural)."
        )
    print("=" * 78)


def main() -> None:
    """Entry point for ``python -m benchmarks.adversarial``."""
    parser = argparse.ArgumentParser(description=__doc__ or "", add_help=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--nli-model", default=None, help="Override the NLI model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None, help="Torch seed for reproducibility")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (cpu | cuda | mps). Defaults to cpu for portability.",
    )
    parser.add_argument(
        "--policy",
        choices=["required", "auto", "quotes_only"],
        default="required",
    )
    args = parser.parse_args()

    fixtures = load_fixtures()
    sources = sources_from_fixtures(fixtures)

    prompt = _adversarial_prompt(sources)
    policy = Policy(args.policy)
    print(
        f"[1/2] Running constrained (policy={policy.value}) + baseline on "
        f"device={args.device} …"
    )
    constrained_text, baseline_text = run_constrained_and_baseline(
        sources,
        prompt,
        args.model,
        args.max_new_tokens,
        device=args.device,
        policy=policy,
        seed=args.seed,
    )

    print("[2/2] Scoring with NLI (coverage disabled — we care about fabrication here) …")
    from citeformer.verify import NLIModel

    nli_kwargs: dict[str, Any] = {}
    if args.nli_model:
        nli_kwargs["model_name"] = args.nli_model
    nli = NLIModel(**nli_kwargs)

    constrained_stats = analyze_run(
        f"GRAMMAR-ENFORCED (citeformer — policy={policy.value})",
        constrained_text,
        sources,
        nli=nli,
        threshold=args.threshold,
        run_coverage=False,
    )
    baseline_stats = analyze_run(
        "BASELINE (plain HF generate, no grammar)",
        baseline_text,
        sources,
        nli=nli,
        threshold=args.threshold,
        run_coverage=False,
    )

    _print_adversarial_report(
        constrained_stats,
        baseline_stats,
        sources,
        n_in_scope=len(sources),
    )


if __name__ == "__main__":
    main()
