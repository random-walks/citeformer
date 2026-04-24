"""citeformer demo benchmark — canonical AI papers as RAG sources.

Loads the six bundled paper fixtures (``benchmarks/fixtures/ai_papers.json``,
pre-fetched via ``benchmarks/fetch_fixtures.py``), runs a small instruction-
tuned model twice over the same prompt:

1. **Grammar-enforced**: through `Citeformer` with the HF backend and the
   selected policy. Citation fabrication is structurally impossible.
2. **Baseline**: plain ``model.generate()`` with no `LogitsProcessor`. Lets
   the model emit whatever ``[N]`` sequences it wants.

Then we regex-parse both outputs for ``[N]`` markers, pair each with
`VerificationReport` data, and print a side-by-side comparison.

Run:

    uv run python -m benchmarks.demo
    uv run python -m benchmarks.demo --model Qwen/Qwen2.5-0.5B-Instruct
    uv run python -m benchmarks.demo --prompt "…"
    uv run python -m benchmarks.demo --policy auto

Requires the ``hf`` + ``verify`` extras:

    uv sync --extra dev --extra hf --extra verify
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


def _default_prompt(sources: list[Source]) -> str:
    """Citation-dense survey prompt assembled via `build_rag_prompt`.

    Explicitly demands ``[N]`` markers so both runs genuinely attempt
    citations; otherwise the comparison is vacuous.
    """
    return build_rag_prompt(
        query=(
            "Write five citation-dense sentences tracing the development of "
            "transformer-based language models, citing at least one of the "
            "sources in every sentence."
        ),
        sources=sources,
        system=("You are writing a brief, citation-dense technical survey. CITE EVERY CLAIM."),
        example=(
            "The Transformer architecture introduced self-attention [1]. "
            "BERT extended this with bidirectional pre-training [2]."
        ),
        answer_prefix="Survey:",
    )


def _print_report(constrained: RunStats, baseline: RunStats, sources: list[Source]) -> None:
    """Pretty-print the side-by-side benchmark summary."""
    print()
    print("=" * 72)
    print("citeformer demo — AI papers RAG")
    print("=" * 72)
    print()
    print(f"Sources in scope (N = {len(sources)}):")
    print(format_source_list(sources))
    print()
    for run in (constrained, baseline):
        print(f"--- {run.label} ---")
        print("Generated text:")
        print(f"  {run.text.strip()[:600]!r}")
        print()
        n = len(run.cite_ids_emitted)
        print(f"  citation markers emitted:      {n}")
        print(f"  cite IDs emitted:              {sorted(set(run.cite_ids_emitted))}")
        print(f"  fabricated IDs (out of range): {run.fabricated_cite_ids}")
        print(f"  fabrication rate:              {fabrication_rate(run):.0%}")
        print(f"  NLI-supported citations:       {run.supported_count} / {n}")
        print(f"  overall support rate:          {run.support_rate:.0%}")
        print(f"  uncited-but-entailed sentences: {run.entailed_uncited_count}")
        print()

    print("=" * 72)
    print(
        f"  fabrication rate: baseline {fabrication_rate(baseline):.0%} → "
        f"citeformer {fabrication_rate(constrained):.0%}"
    )
    print(
        f"  NLI-support rate: baseline {baseline.support_rate:.0%} → "
        f"citeformer {constrained.support_rate:.0%}"
    )
    print("=" * 72)


def main() -> None:
    """Entry point for ``python -m benchmarks.demo``."""
    parser = argparse.ArgumentParser(description=__doc__ or "", add_help=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (default: %(default)s)")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--prompt", default=None, help="Override the default prompt")
    parser.add_argument("--nli-model", default=None, help="Override the NLI model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None, help="Torch seed for reproducibility")
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Torch device (cpu | cuda | mps). Defaults to cpu — on Apple Silicon "
            "MPS hits an ndarray-size limit with XGrammar + Qwen-sized tokenizers. "
            "Override to 'mps' or 'cuda' if your combination works."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=["required", "auto", "quotes_only"],
        default="required",
        help=(
            "Citation policy for the constrained run. 'required' forces every "
            "sentence to carry a citation (default — rely on ADR-009's bounded "
            "content for small-model progression). 'auto' lets the model decide."
        ),
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=None,
        help=(
            "Override the REQUIRED-policy content bound. Default uses "
            "DEFAULT_MAX_CONTENT_CHARS (240). Smaller (e.g. 60) forces cites to "
            "land more often per sentence; larger (e.g. 500) allows longer prose."
        ),
    )
    parser.add_argument(
        "--premise",
        choices=["abstract", "fulltext"],
        default="abstract",
        help=(
            "Which text to feed NLI as the premise. 'abstract' (default) uses "
            "each paper's arXiv abstract. 'fulltext' uses PDF-extracted body "
            "text (requires `python -m benchmarks.fetch_fixtures --fulltext` "
            "to have populated the fixtures)."
        ),
    )
    args = parser.parse_args()

    fixtures = load_fixtures()
    sources = sources_from_fixtures(fixtures, premise=args.premise)

    prompt = args.prompt or _default_prompt(sources)
    policy = Policy(args.policy)
    print(
        f"[1/2] Running constrained (policy={policy.value}) + baseline generation "
        f"on device={args.device}…"
    )
    constrained_text, baseline_text = run_constrained_and_baseline(
        sources,
        prompt,
        args.model,
        args.max_new_tokens,
        device=args.device,
        policy=policy,
        max_content_chars=args.max_content_chars,
        seed=args.seed,
    )

    print("[2/2] Scoring with NLI …")
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
    )
    baseline_stats = analyze_run(
        "BASELINE (plain HF generate, no grammar)",
        baseline_text,
        sources,
        nli=nli,
        threshold=args.threshold,
    )

    _print_report(constrained_stats, baseline_stats, sources)


if __name__ == "__main__":
    main()
