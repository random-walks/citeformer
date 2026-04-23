# Getting started

:::{note}
This page is a stub. The real quickstart lands once P2 ships — a working HF backend with grammar-level citation enforcement. Follow the [CHANGELOG](https://github.com/random-walks/citeformer/blob/main/CHANGELOG.md) or watch the repo for the v0.1 release announcement.
:::

## P0 — installing the scaffold

Until P2 lands, there's no working backend yet, but you can still install the package to see the shape:

```bash
pip install citeformer
python -c "import citeformer; print(citeformer.__version__)"
```

## What to expect in v0.1

After the full P0 → P6 phase plan lands (see [architecture](reference/architecture.md)), the user-facing API will look roughly like this:

```python
from citeformer import Citeformer, Source

sources = [
    Source.from_doi("10.1038/s41586-023-06221-2"),
    Source.from_arxiv("2305.14627"),
    Source(
        metadata={
            "id": "poe-raven",
            "type": "book",
            "title": "The Raven",
            "author": [{"family": "Poe", "given": "Edgar Allan"}],
            "issued": {"date-parts": [[1845]]},
        },
        content="Once upon a midnight dreary...",
    ),
]

cf = Citeformer(
    backend="hf",
    model="microsoft/Phi-3.5-mini-instruct",
    style="apa-7",
    citation_policy="required",
)
result = cf.generate(prompt="Summarize these works.", sources=sources)

print(result.text)
# → "Poe's 'The Raven' opens with mystery [3]. The Nature paper shows [1]..."

for ref in result.references:
    print(ref.rendered)
# → "Poe, E. A. (1845). The Raven."
# → ... (rendered by citeproc-py, not by the LLM)

report = result.verify()
print(report.support_rate)  # NLI-based entailment check
```

The key property: `[4]` cannot appear in `result.text` because there is no fourth source. It's structurally impossible at the logit level, not a prompt hint or post-hoc check.
