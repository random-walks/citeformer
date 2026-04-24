---
title: citeformer demo
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
---

# citeformer — adversarial demo

**Live demo of the 100% → 0% fabrication swing on HuggingFace Spaces.**

This Space hosts the adversarial citation test from the [citeformer benchmarks](https://github.com/random-walks/citeformer/blob/main/benchmarks/README.md). A prompt *deliberately* demands the model cite sources `[7]` and `[8]` when only 6 sources are in scope. The plain HF baseline complies; citeformer's grammar mask structurally can't.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Opens a Gradio UI on `http://localhost:7860`.

## Deploying to HF Spaces

One-time setup:

```bash
pip install huggingface_hub
huggingface-cli login   # paste a write-scoped token from https://huggingface.co/settings/tokens
```

Create the Space in the web UI at <https://huggingface.co/new-space>
with **Gradio SDK**, **CPU basic** hardware.

Then from this directory:

```bash
./deploy.sh <hf-username>/<space-name>
# e.g.  ./deploy.sh random-walks/citeformer-demo
```

`deploy.sh` clones the Space, syncs `app.py` + `requirements.txt` + this
README, and pushes a commit tagged with the source citeformer SHA.
Re-run any time to redeploy — it's idempotent.

The Space takes ~2 min to warm up (Qwen 2.5 0.5B download + XGrammar compiler init), then stays warm. Total memory footprint: ~2 GB.

## Model footprint

- `Qwen/Qwen2.5-0.5B-Instruct` — ~500 MB on disk, ~1.5 GB runtime. Free-tier-friendly.
- Swap via `CITEFORMER_DEMO_MODEL=<hf-id>` env var. Any instruction-tuned HF causal LM works as long as it fits in 16 GB RAM.

## What this demonstrates

citeformer's HF backend wires [XGrammar](https://github.com/mlc-ai/xgrammar) into `transformers.model.generate()`. The grammar enumerates the valid cite ids per call (`[1] | [2] | ... | [N]` for N sources). Generating `[7]` when N=6 is *token-impossible to sample* — not hopefully-unlikely, not post-processed out.

Full guarantee + benchmark data: [github.com/random-walks/citeformer](https://github.com/random-walks/citeformer).
