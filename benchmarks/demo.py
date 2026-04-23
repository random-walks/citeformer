"""citeformer demo benchmark — canonical AI papers as RAG sources.

Loads the six bundled paper fixtures (``benchmarks/fixtures/ai_papers.json``,
pre-fetched via ``benchmarks/fetch_fixtures.py``), runs a small instruction-
tuned model twice over the same prompt:

1. **Grammar-enforced**: through `Citeformer` with the HF backend and the
   `REQUIRED` policy. Citation fabrication is structurally impossible.
2. **Baseline**: plain ``model.generate()`` with no `LogitsProcessor`. Lets
   the model emit whatever ``[N]`` sequences it wants.

Then we regex-parse both outputs for ``[N]`` markers, pair each with
`VerificationReport` data, and print a side-by-side comparison.

Run:

    uv run python -m benchmarks.demo
    uv run python -m benchmarks.demo --model Qwen/Qwen2.5-0.5B-Instruct
    uv run python -m benchmarks.demo --prompt "…"

Requires the ``hf`` + ``verify`` extras:

    uv sync --extra dev --extra hf --extra verify
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citeformer import Citeformer, Policy, Source
from citeformer.render import render_references

# Default tiny instruction-tuned model. Small enough to download + run on any
# laptop; big enough to produce text that cites sources with some structure.
# Qwen 2.5 0.5B Instruct is the sweet spot: ~500 MB, genuinely instruction-
# tuned so it honors "cite with [N] markers" in the prompt.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Prompt explicitly asks for ``[N]`` citations. Under the constrained run the
# grammar ensures *any* emitted ``[N]`` is in range; under the baseline the
# model is free to fabricate. We push the prompt hard on citations so both
# runs actually attempt them — otherwise the comparison is vacuous.
DEFAULT_PROMPT = (
    "You are writing a brief, citation-dense technical survey. CITE EVERY "
    "CLAIM. Use [N] markers — for example [1] or [3]. The sources are:\n"
    "{source_list}\n\n"
    "Example sentence: The Transformer architecture introduced self-attention "
    "[1]. BERT extended this with bidirectional pre-training [2].\n\n"
    "Now write five citation-dense sentences tracing the development of "
    "transformer-based language models, citing at least one of the sources "
    "in every sentence.\n\n"
    "Survey:"
)

_CITE_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class _RunStats:
    """Stats for a single generation run (constrained or baseline)."""

    label: str
    text: str
    cite_ids_emitted: list[int]
    fabricated_cite_ids: list[int]
    supported_count: int
    entailed_uncited_count: int
    support_rate: float


def _load_fixtures(path: Path) -> list[dict[str, Any]]:
    """Load the pre-fetched paper fixtures. Fail loudly if missing."""
    if not path.exists():
        raise SystemExit(
            f"Fixtures missing at {path!s}. Run first: `uv run python -m benchmarks.fetch_fixtures`"
        )
    return json.loads(path.read_text())


def _sources_from_fixtures(fixtures: list[dict[str, Any]]) -> list[Source]:
    """Convert fixture entries to `Source` objects with the abstract as content."""
    sources: list[Source] = []
    for entry in fixtures:
        csl = dict(entry["csl"])
        abstract = str(csl.pop("abstract", "")).strip()
        sources.append(Source(metadata=csl, content=abstract))
    return sources


def _format_source_list(sources: list[Source]) -> str:
    """Render a numbered source list for the prompt + stdout."""
    lines = []
    for i, source in enumerate(sources, start=1):
        title = str(source.metadata.get("title", "Untitled")).strip()
        authors_raw = source.metadata.get("author") or []
        author_names = []
        for a in authors_raw[:3]:
            if isinstance(a, dict):
                family = a.get("family") or a.get("literal") or ""
                if family:
                    author_names.append(family)
        author_str = ", ".join(author_names)
        if len(authors_raw) > 3:
            author_str += " et al."
        lines.append(f"[{i}] {author_str}: {title}")
    return "\n".join(lines)


def _both_runs(
    sources: list[Source],
    prompt: str,
    model_name: str,
    max_new_tokens: int,
    *,
    device: str | None = None,
) -> tuple[str, str]:
    """Run constrained + baseline through a single shared HF model instance.

    Avoids loading the weights twice (halves RAM + works around MPS
    ndarray-size limits on Apple Silicon). The constrained run uses the
    full HFBackend pipeline; the baseline reaches into the backend's
    already-loaded model/tokenizer and calls ``model.generate`` with no
    LogitsProcessor.

    Returns:
        ``(constrained_text, baseline_text)``.
    """
    import torch

    from citeformer.backends.hf import HFBackend

    backend = HFBackend(model=model_name, device=device)

    # Constrained path via citeformer. AUTO policy lets cites happen
    # anywhere but the grammar still constrains which ``[N]`` tokens are
    # reachable — that's the core guarantee we're demonstrating. (REQUIRED
    # would force every sentence to end with a cite, but the current v0.1
    # GBNF grammar lets the model stall in content state and never transition
    # to the cite group — a known limitation for small models that don't
    # naturally bias toward closing brackets. Documented in
    # docs/decisions/007-required-policy-progression.md.)
    cf = Citeformer(backend=backend, citation_policy=Policy.AUTO)
    result = cf.generate(prompt=prompt, sources=sources, max_new_tokens=max_new_tokens)
    constrained_text = result.text

    # Baseline path: same model, no grammar. Reuse backend internals.
    inputs = backend.tokenizer(prompt, return_tensors="pt").to(backend.device)
    with torch.no_grad():
        output_ids = backend.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=backend.tokenizer.eos_token_id or backend.tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs.input_ids.shape[1] :]
    baseline_text = str(backend.tokenizer.decode(generated, skip_special_tokens=True))

    return constrained_text, baseline_text


def _analyze_run(
    label: str,
    text: str,
    sources: list[Source],
    *,
    nli: Any,
    threshold: float,
) -> _RunStats:
    """Run verification on a generation and compute summary stats."""
    from citeformer.core import Citation, GenerationResult

    cite_ids = [int(m.group(1)) for m in _CITE_PATTERN.finditer(text)]
    in_range = {i for i in cite_ids if 1 <= i <= len(sources)}
    fabricated = sorted({i for i in cite_ids if i not in in_range})

    # Build a minimal GenerationResult with citations so verify() can run.
    # source_id < 1 trips pydantic validation; skip but still counted in fabricated.
    citations: list[Citation] = []
    for m in _CITE_PATTERN.finditer(text):
        with contextlib.suppress(Exception):
            citations.append(
                Citation(
                    span=(m.start(), m.end()),
                    source_id=int(m.group(1)),
                )
            )

    references = render_references(sources, citations, style_name="apa-7")
    result = GenerationResult(
        text=text,
        citations=citations,
        references=references,
        sources=sources,
    )
    report = result.verify(threshold=threshold, nli=nli, run_coverage=True)

    supported = sum(1 for cs in report.per_citation if cs.supported)
    return _RunStats(
        label=label,
        text=text,
        cite_ids_emitted=cite_ids,
        fabricated_cite_ids=fabricated,
        supported_count=supported,
        entailed_uncited_count=len(report.uncited_but_entailed),
        support_rate=report.support_rate,
    )


def _print_report(constrained: _RunStats, baseline: _RunStats, sources: list[Source]) -> None:
    """Pretty-print the side-by-side benchmark summary."""
    print()
    print("=" * 72)
    print("citeformer demo — AI papers RAG")
    print("=" * 72)
    print()
    print(f"Sources in scope (N = {len(sources)}):")
    print(_format_source_list(sources))
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
        fab_rate = len(run.fabricated_cite_ids) / max(len(set(run.cite_ids_emitted)), 1)
        print(f"  fabrication rate:              {fab_rate:.0%}")
        print(f"  NLI-supported citations:       {run.supported_count} / {n}")
        print(f"  overall support rate:          {run.support_rate:.0%}")
        print(f"  uncited-but-entailed sentences: {run.entailed_uncited_count}")
        print()

    # Headline summary.
    print("=" * 72)
    baseline_fab = len(baseline.fabricated_cite_ids) / max(len(set(baseline.cite_ids_emitted)), 1)
    constrained_fab = len(constrained.fabricated_cite_ids) / max(
        len(set(constrained.cite_ids_emitted)), 1
    )
    print(f"  fabrication rate: baseline {baseline_fab:.0%} → citeformer {constrained_fab:.0%}")
    print(
        f"  NLI-support rate: baseline {baseline.support_rate:.0%} → citeformer {constrained.support_rate:.0%}"
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
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Torch device (cpu | cuda | mps). Defaults to cpu — on Apple Silicon "
            "MPS hits an ndarray-size limit with XGrammar + Qwen-sized tokenizers. "
            "Override to 'mps' or 'cuda' if your combination works."
        ),
    )
    args = parser.parse_args()

    fixture_path = Path(__file__).parent / "fixtures" / "ai_papers.json"
    fixtures = _load_fixtures(fixture_path)
    sources = _sources_from_fixtures(fixtures)

    prompt_template = args.prompt or DEFAULT_PROMPT
    prompt = prompt_template.format(source_list=_format_source_list(sources))

    print(f"[1/2] Running constrained + baseline generation on device={args.device}…")
    constrained_text, baseline_text = _both_runs(
        sources, prompt, args.model, args.max_new_tokens, device=args.device
    )

    print("[2/2] Scoring with NLI …")
    from citeformer.verify import NLIModel

    nli_kwargs: dict[str, Any] = {}
    if args.nli_model:
        nli_kwargs["model_name"] = args.nli_model
    nli = NLIModel(**nli_kwargs)

    constrained_stats = _analyze_run(
        "GRAMMAR-ENFORCED (citeformer)",
        constrained_text,
        sources,
        nli=nli,
        threshold=args.threshold,
    )
    baseline_stats = _analyze_run(
        "BASELINE (plain HF generate, no grammar)",
        baseline_text,
        sources,
        nli=nli,
        threshold=args.threshold,
    )

    _print_report(constrained_stats, baseline_stats, sources)


if __name__ == "__main__":
    main()
