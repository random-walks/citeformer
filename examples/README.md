# citeformer examples

Runnable scripts that double as living reports — each one prints what it
finds, so rerunning is the way to audit current behavior. Commentary at the
top of each file explains intent, expected output, and anything you'd want
to know before trusting it.

## Scripts

### 1. [`01_quickstart_mock.py`](01_quickstart_mock.py) — shortest possible demo (no ML)

Uses `MockBackend` so the example runs without downloading a model, without
torch, and without the `hf` / `verify` extras. Good for:

- smoke-checking an install
- reading the pipeline shape without the noise of real generation
- CI-light integration checks in downstream projects

```bash
uv run python examples/01_quickstart_mock.py
```

### 2. [`02_rag_with_hf_and_verify.py`](02_rag_with_hf_and_verify.py) — the full pipeline

Loads a small HF model, pushes three hand-built sources into a RAG-ish
prompt, asserts grammar-level non-fabrication, and runs NLI verification
end-to-end. This is the "does citeformer actually deliver" demo.

```bash
uv sync --extra dev --extra hf --extra verify
uv run python examples/02_rag_with_hf_and_verify.py
```

### 3. [`03_standalone_rendering.py`](03_standalone_rendering.py) — renderer alone

All six bundled styles (APA 7, MLA 9, Chicago author-date, IEEE, Nature,
Vancouver) rendered against the same CSL-JSON item. Useful when you want
to preview how a reference will look before feeding it to the model — or
when you're using citeformer purely as a bibliography renderer.

```bash
uv run python examples/03_standalone_rendering.py
```

### 4. [`04_fetch_and_render.py`](04_fetch_and_render.py) — DOI → full pipeline

Grabs CSL-JSON from Crossref for a real DOI, renders it in all six styles,
and shows the inline marker for each. Hits the network; cached on disk
after the first run.

```bash
uv run python examples/04_fetch_and_render.py
```

### 5. [`05_streaming.py`](05_streaming.py) — realtime chunk streaming

Prints tokens as the model decodes them via `Citeformer.stream()`, then
calls `.finalize()` to get the full `GenerationResult` with parsed
citations + rendered references. Grammar enforcement applies to every
chunk exactly as in `generate()`.

```bash
uv sync --extra dev --extra hf
uv run python examples/05_streaming.py
```

### 6. [`06_langchain_rag.py`](06_langchain_rag.py) — LangChain `Document` → citeformer `Source`

Wraps a LangChain retriever output (hand-built for the demo, but the same
shape as `retriever.invoke(query)`) with `sources_from_documents` and pipes
it through `Citeformer.generate`. Citation fabrication is structural — no
matter what the LLM tries to cite, it's in [1..N].

```bash
uv pip install langchain-core
uv sync --extra dev --extra hf
uv run python examples/06_langchain_rag.py
```

### 7. [`07_llamaindex_rag.py`](07_llamaindex_rag.py) — LlamaIndex `NodeWithScore` → `Source`

Same pattern for LlamaIndex: `sources_from_nodes(retrieved_nodes)`,
then `Citeformer.generate`. Uses IEEE-style references to show the
style-picking works end-to-end with external node metadata.

```bash
uv pip install llama-index-core
uv sync --extra dev --extra hf
uv run python examples/07_llamaindex_rag.py
```

### 8. [`08_literature_review.ipynb`](08_literature_review.ipynb) — full academic workflow

Jupyter notebook walking through the realistic literature-review use case:
pull 6 arXiv papers on prompt reasoning, generate a paragraph-length review
under REQUIRED policy, assert the structural guarantee held, run NLI
verification over every emitted citation, and render an APA-7
bibliography. Ends with a side-by-side baseline comparison showing
fabrication in the unmasked run.

```bash
uv sync --extra dev --extra hf --extra verify --extra examples
uv run jupyter notebook examples/08_literature_review.ipynb
```

The notebook loads `Qwen/Qwen2.5-0.5B-Instruct` (CPU-friendly, ~500 MB).

## Adding a new citation style

Adding a seventh (or seventieth) citation style is handled by the
`add-citation-format` skill at
[`.claude/skills/add-citation-format/SKILL.md`](../.claude/skills/add-citation-format/SKILL.md).
It spells out:

- the template for a new `CitationFormatter` subclass
- the minimum fixture coverage (four canonical CSL types)
- registration + alias wiring

Use Claude Code's `Skill` tool with `add-citation-format` or follow the
file by hand. The existing six formatters under
[`../src/citeformer/render/formatters/`](../src/citeformer/render/formatters/)
are the working templates.

## A note on expected output

None of these scripts have a stable transcript committed to the repo —
generation with sampling is not reproducible across hardware, and pinning
a transcript would misrepresent the library as deterministic in places
where it isn't. The *shape* of the output is checked by unit + integration
tests; the *prose* is for you to read after running.
