"""Shared helpers for the benchmark scripts.

Extracted so `demo.py`, `adversarial.py`, and `sweep.py` don't drift on
fixture loading, source list formatting, or verification-run analysis.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citeformer import Citeformer, Policy, Source
from citeformer.render import render_references

_CITE_PATTERN = re.compile(r"\[(\d+)\]")
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ai_papers.json"


@dataclass(frozen=True)
class RunStats:
    """Stats for a single generation run (constrained or baseline).

    Attributes:
        label: Human-readable run identifier (used in report printouts).
        text: Generated text including inline ``[N]`` markers.
        cite_ids_emitted: All cite ids in order of appearance. Duplicates kept.
        fabricated_cite_ids: Sorted unique cite ids that are out of range
            (``id < 1`` or ``id > len(sources)``). Always empty under grammar
            enforcement; may be populated on the baseline path.
        supported_count: Number of citations with ``supported == True`` in the
            `VerificationReport`.
        entailed_uncited_count: Number of sentences flagged by the coverage
            check (uncited but entailed by at least one source).
        support_rate: Overall support rate from the `VerificationReport`.
    """

    label: str
    text: str
    cite_ids_emitted: list[int]
    fabricated_cite_ids: list[int]
    supported_count: int
    entailed_uncited_count: int
    support_rate: float


def load_fixtures(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Load the pre-fetched AI-paper fixtures, failing loudly if absent."""
    if not path.exists():
        raise SystemExit(
            f"Fixtures missing at {path!s}. "
            "Run first: `uv run python -m benchmarks.fetch_fixtures`"
        )
    return list(json.loads(path.read_text()))


def sources_from_fixtures(fixtures: list[dict[str, Any]]) -> list[Source]:
    """Convert fixture entries to `Source` objects with the abstract as content."""
    sources: list[Source] = []
    for entry in fixtures:
        csl = dict(entry["csl"])
        abstract = str(csl.pop("abstract", "")).strip()
        sources.append(Source(metadata=csl, content=abstract))
    return sources


def format_source_list(sources: list[Source]) -> str:
    """Render a numbered source list suitable for inclusion in a prompt."""
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


def run_constrained_and_baseline(
    sources: list[Source],
    prompt: str,
    model_name: str,
    max_new_tokens: int,
    *,
    device: str | None = None,
    policy: Policy = Policy.REQUIRED,
    max_content_chars: int | None = None,
    seed: int | None = None,
    temperature: float = 0.7,
) -> tuple[str, str]:
    """Run constrained + baseline through a single shared HF model instance.

    Shared load halves RAM and dodges the MPS ndarray-size limit on Apple
    Silicon with XGrammar + Qwen-sized tokenizers. Constrained path uses the
    full `HFBackend` pipeline; baseline reaches into the backend's already-
    loaded model/tokenizer and calls `model.generate` with no LogitsProcessor.

    Returns `(constrained_text, baseline_text)`. Both runs use the same seed
    (if any) so the baseline gets the same sampling chance as constrained.
    """
    import torch

    from citeformer.backends.hf import HFBackend

    if seed is not None:
        torch.manual_seed(seed)

    backend = HFBackend(model=model_name, device=device)

    cf = Citeformer(backend=backend, citation_policy=policy)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    if max_content_chars is not None:
        generate_kwargs["max_content_chars"] = max_content_chars
    result = cf.generate(prompt=prompt, sources=sources, **generate_kwargs)
    constrained_text = result.text

    # Baseline: same model + seed, no grammar processor.
    if seed is not None:
        torch.manual_seed(seed)
    inputs = backend.tokenizer(prompt, return_tensors="pt").to(backend.device)
    with torch.no_grad():
        output_ids = backend.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=backend.tokenizer.eos_token_id or backend.tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs.input_ids.shape[1] :]
    baseline_text = str(backend.tokenizer.decode(generated, skip_special_tokens=True))

    return constrained_text, baseline_text


def analyze_run(
    label: str,
    text: str,
    sources: list[Source],
    *,
    nli: Any,
    threshold: float,
    run_coverage: bool = True,
) -> RunStats:
    """Parse a generation's cites, verify with NLI, package as `RunStats`."""
    from citeformer.core import Citation, GenerationResult

    cite_ids = [int(m.group(1)) for m in _CITE_PATTERN.finditer(text)]
    in_range = {i for i in cite_ids if 1 <= i <= len(sources)}
    fabricated = sorted({i for i in cite_ids if i not in in_range})

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
    report = result.verify(threshold=threshold, nli=nli, run_coverage=run_coverage)

    supported = sum(1 for cs in report.per_citation if cs.supported)
    return RunStats(
        label=label,
        text=text,
        cite_ids_emitted=cite_ids,
        fabricated_cite_ids=fabricated,
        supported_count=supported,
        entailed_uncited_count=len(report.uncited_but_entailed),
        support_rate=report.support_rate,
    )


def fabrication_rate(stats: RunStats) -> float:
    """Fraction of unique cite ids emitted that are out of range, in [0, 1]."""
    emitted = set(stats.cite_ids_emitted)
    if not emitted:
        return 0.0
    return len(stats.fabricated_cite_ids) / len(emitted)
