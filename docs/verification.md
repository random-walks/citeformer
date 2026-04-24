# Verification

citeformer's grammar layer prevents **fabricated citations** at decode time
— under a grammar-enforced backend, the model physically cannot emit
``[N]`` for an `N` outside the in-scope source list. The verification
layer (P6+) runs a separate set of checks on top:

1. **Existence** — trivially `True` under grammar enforcement; a sanity
   check for out-of-range ids.
2. **Entailment** — for each emitted ``[N]``, extract the sentence
   containing the marker, strip the marker, score NLI entailment against
   the content of source `N`. `supported == entailment_score >= threshold`.
3. **Coverage** — for each sentence with *no* citation, score NLI against
   every source. Flag sentences where at least one source entails the
   sentence above threshold — that's a "missing citation" candidate.

Together, they answer three distinct questions:

- "Did the model invent a source?" → existence. Answered by construction
  under grammar enforcement.
- "Is the claim actually supported by the cited source?" → entailment.
  Only the NLI model can tell you this; citeformer wraps the scoring.
- "Are there claims in the text that *should* have been cited but
  weren't?" → coverage.

## Requirements

Verification is in the `verify` extra:

```bash
pip install 'citeformer[verify]'
```

First call to `verify()` downloads the default NLI model
(``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``, ~850 MB).
Override via the ``CITEFORMER_NLI_MODEL`` env var or the ``nli=`` kwarg:

```python
from citeformer.verify import NLIModel

small_nli = NLIModel(model_name="cross-encoder/nli-deberta-v3-base")
report = result.verify(nli=small_nli)
```

## Usage

The common path is `GenerationResult.verify()`:

```python
from citeformer import Citeformer, Source
from citeformer.backends.hf import HFBackend

cf = Citeformer(backend=HFBackend(model="Qwen/Qwen2.5-0.5B-Instruct"))
result = cf.generate(prompt="...", sources=sources)

report = result.verify(threshold=0.5)

print(f"support rate: {report.support_rate:.0%}")
for cs in report.per_citation:
    print(f"  citation {cs.citation_index}: entailment={cs.entailment_score:.2f}, supported={cs.supported}")

for flagged in report.uncited_but_entailed:
    start, end = flagged.span
    print(f"  uncited span [{start}:{end}] would be supported by source {flagged.candidate_source_id}")
```

## Advanced: using a custom `Verifier`

`GenerationResult.verify()` wraps `citeformer.verify.Verifier` with
defaults. If you want to reuse an NLI model across many generations, hold
one `Verifier` instance:

```python
from citeformer.verify import NLIModel, Verifier

nli = NLIModel(batch_size=16)  # larger batches on GPU
verifier = Verifier(threshold=0.6, nli=nli)

for prompt in prompts:
    result = cf.generate(prompt=prompt, sources=sources)
    report = verifier.verify(
        text=result.text,
        citations=result.citations,
        sources=sources,
        run_coverage=False,  # skip coverage for throughput
    )
```

## Thresholds

The default threshold (0.5) is tuned for DeBERTa-v3-large-MNLI. With the
smaller cross-encoder variants, consider lowering to 0.4 (they're more
conservative). For strict academic use (journal-ready manuscripts), raise
to 0.7 and treat `supported=False` as a real issue to fix.

## Limitations

- **Sentence splitter**: regex-based; mis-splits on abbreviations outside
  our small allowlist, URLs with dots, inline formulae. Well-behaved on
  normal prose. See [ADR-007](decisions/007-required-policy-progression-gap.md).
- **NLI scope**: scores entailment per-sentence, premise = full source
  content (truncated at 512 tokens). Multi-sentence claims that span a
  citation boundary may score low even when the global claim is supported.
- **Small models emit metadata-looking text**: outputs sometimes include a
  "References: [1] Author et al." block that our parser treats as
  citations. Their "claim" is just the reference metadata, which doesn't
  entail anything — dragging down the support rate on small models. Use
  stop-sequences or a larger model if this matters.

## Programmatic access

```python
from citeformer.verify import (
    Verifier,           # orchestrator
    NLIModel,           # transformers wrapper
    CitationSupport,    # per-citation detail
    UncitedClaim,       # per-uncited-sentence flag
    VerificationReport, # top-level shape
    check_existence,    # pure, no ML
    split_sentences,    # pure, no ML
)
```
