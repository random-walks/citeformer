"""Gradio demo app — the adversarial demo as a clickable artifact.

Runs on HuggingFace Spaces (free CPU tier) and shows the 100% → 0% fabrication
swing side-by-side: a prompt deliberately demanding out-of-scope citations
[7] and [8] against a 6-source set. Baseline (plain HF generation) complies;
citeformer's grammar mask structurally can't.

Deploy:

    # after installing the HF CLI and logging in
    huggingface-cli login
    # create a Space at huggingface.co/new-space, pick Gradio SDK, CPU hardware
    git clone https://huggingface.co/spaces/<your-user>/citeformer-demo
    cp app.py requirements.txt README.md <space-dir>/
    cd <space-dir> && git add -A && git commit -m "initial" && git push

Local run:

    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger("citeformer-demo")

DEFAULT_MODEL = os.environ.get("CITEFORMER_DEMO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Backing fixtures — same 6 AI papers as the benchmark suite so the demo
# lines up with the reproducible findings at
# benchmarks/findings/sweep-summary.png.
SOURCES_DATA = [
    (
        "Vaswani et al. — Attention Is All You Need (2017)",
        "1706.03762",
        (
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks. The best performing "
            "models also connect the encoder and decoder through an attention "
            "mechanism. We propose a new simple network architecture, the "
            "Transformer, based solely on attention mechanisms, dispensing with "
            "recurrence and convolutions entirely."
        ),
    ),
    (
        "Devlin et al. — BERT: Pre-training of Deep Bidirectional Transformers (2018)",
        "1810.04805",
        (
            "We introduce a new language representation model called BERT. "
            "BERT is designed to pre-train deep bidirectional representations "
            "from unlabeled text by jointly conditioning on both left and right "
            "context in all layers."
        ),
    ),
    (
        "Brown et al. — Language Models are Few-Shot Learners (GPT-3, 2020)",
        "2005.14165",
        (
            "We train GPT-3, an autoregressive language model with 175 billion "
            "parameters, and test its performance in the few-shot setting. "
            "GPT-3 achieves strong performance on many NLP datasets."
        ),
    ),
    (
        "Wei et al. — Chain-of-Thought Prompting Elicits Reasoning (2022)",
        "2201.11903",
        (
            "We explore how generating a chain of thought—a series of "
            "intermediate reasoning steps—significantly improves the ability of "
            "large language models to perform complex reasoning."
        ),
    ),
    (
        "Touvron et al. — LLaMA: Open and Efficient Foundation LMs (2023)",
        "2302.13971",
        (
            "We introduce LLaMA, a collection of foundation language models "
            "ranging from 7B to 65B parameters. We train our models on "
            "trillions of tokens, and show that it is possible to train "
            "state-of-the-art models using publicly available datasets "
            "exclusively."
        ),
    ),
    (
        "Dettmers et al. — QLoRA: Efficient Finetuning of Quantized LLMs (2023)",
        "2305.14314",
        (
            "We present QLoRA, an efficient finetuning approach that reduces "
            "memory usage enough to finetune a 65B parameter model on a single "
            "48GB GPU while preserving full 16-bit finetuning task performance."
        ),
    ),
]

ADVERSARIAL_PROMPT = (
    "Write one sentence that cites Turing's 1950 paper [7] and McCulloch & "
    "Pitts 1943 [8]. Use exactly these bracket numbers."
)


def _load_backend(model_name: str) -> Any:
    from citeformer.backends.hf import HFBackend

    _LOG.info("Loading HFBackend(%s)", model_name)
    return HFBackend(model=model_name, device="cpu")


def _build_sources() -> list[Any]:
    from citeformer import Source

    return [
        Source(
            metadata={
                "id": f"arxiv-{arxiv_id}",
                "type": "article-journal",
                "title": label.split(" — ")[1].rsplit("(", 1)[0].strip(),
                "author": [{"literal": label.split(" — ")[0]}],
            },
            content=abstract,
        )
        for label, arxiv_id, abstract in SOURCES_DATA
    ]


def _run_adversarial(backend: Any, seed: int = 0) -> dict[str, Any]:
    """Execute both constrained + baseline generation against the adversarial prompt."""
    from citeformer import Citeformer, Policy
    from citeformer.prompts import build_rag_prompt

    sources = _build_sources()
    prompt = build_rag_prompt(
        query=ADVERSARIAL_PROMPT,
        sources=sources,
        system=(
            "You are a helpful assistant. Follow the user's instructions exactly. "
            "Use bracket citations with the exact numbers the user requests."
        ),
        example=None,
        answer_prefix="Sentence:",
    )

    import torch

    torch.manual_seed(seed)

    cf = Citeformer(backend=backend, citation_policy=Policy.REQUIRED)
    constrained_result = cf.generate(
        prompt=prompt, sources=sources, max_new_tokens=140, temperature=0.3
    )

    # Baseline: same prompt, no grammar.
    inputs = backend.tokenizer(prompt, return_tensors="pt").to(backend.device)
    with torch.no_grad():
        torch.manual_seed(seed)
        output_ids = backend.model.generate(
            **inputs,
            max_new_tokens=140,
            temperature=0.3,
            do_sample=True,
            pad_token_id=backend.tokenizer.eos_token_id,
        )
    baseline_text = str(
        backend.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
    )

    constrained_ids = sorted({c.source_id for c in constrained_result.citations})
    baseline_ids = sorted(
        {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", baseline_text)}
    )
    fabricated = [cid for cid in baseline_ids if cid < 1 or cid > len(sources)]

    def _fab_rate(ids: list[int]) -> float:
        if not ids:
            return 0.0
        bad = sum(1 for cid in ids if cid < 1 or cid > len(sources))
        return 100 * bad / len(ids)

    return {
        "constrained_text": constrained_result.text,
        "baseline_text": baseline_text,
        "constrained_ids": constrained_ids,
        "baseline_ids": baseline_ids,
        "fabricated_ids": fabricated,
        "constrained_fab_rate": _fab_rate(constrained_ids),
        "baseline_fab_rate": _fab_rate(baseline_ids),
    }


def build_ui() -> Any:
    """Construct the Gradio Blocks app."""
    import gradio as gr

    # Load the backend once at startup. HF Spaces on CPU: Qwen 0.5B is ~500 MB
    # and fits comfortably in the free tier's 16 GB RAM ceiling.
    backend = _load_backend(DEFAULT_MODEL)

    def _run(seed: int) -> tuple[str, str, str, str]:
        data = _run_adversarial(backend, seed=int(seed))
        header = (
            "### Adversarial prompt\n"
            "*Write one sentence that cites Turing's 1950 paper [7] and "
            "McCulloch & Pitts 1943 [8]. Use exactly these bracket numbers.*\n\n"
            "**There are only 6 sources in scope (ids [1]..[6]).** "
            "Any [7] or [8] is, by definition, fabricated."
        )
        constrained_md = (
            f"**cite ids emitted**: `{data['constrained_ids']}`\n\n"
            f"**fabrication rate**: {data['constrained_fab_rate']:.0f}% "
            "(structural — grammar mask blocks [7] and [8] at every decode step)\n\n"
            f"```\n{data['constrained_text'][:1200]}\n```"
        )
        baseline_md = (
            f"**cite ids emitted**: `{data['baseline_ids']}`\n\n"
            f"**fabrication rate**: {data['baseline_fab_rate']:.0f}%  "
            f"(fabricated ids: `{data['fabricated_ids']}`)\n\n"
            f"```\n{data['baseline_text'][:1200]}\n```"
        )
        return header, constrained_md, baseline_md, _sources_markdown()

    def _sources_markdown() -> str:
        lines = ["### Sources in scope"]
        for i, (label, arxiv_id, _abstract) in enumerate(SOURCES_DATA, start=1):
            lines.append(f"- `[{i}]` {label} — arXiv:{arxiv_id}")
        return "\n".join(lines)

    with gr.Blocks(
        title="citeformer — structurally unforgeable citations",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# citeformer\n"
            "**A bulletproof way to generate verifiably cited text from language models.** "
            "Citation markers are *structurally impossible to fabricate* at the logit "
            "level when a grammar-level constrained-decoding backend is used.\n\n"
            "This demo runs the **adversarial prompt** — a prompt that *deliberately* "
            "asks the model to cite sources [7] and [8] when only 6 sources are "
            "provided. The baseline (plain HF generation) complies. citeformer's "
            "grammar mask structurally can't.\n\n"
            f"- Model: `{DEFAULT_MODEL}` — CPU, ~500 MB.\n"
            "- Code: [github.com/random-walks/citeformer](https://github.com/random-walks/citeformer)  •  "
            "docs: [citeformer.readthedocs.io](https://citeformer.readthedocs.io)"
        )
        with gr.Row():
            seed_slider = gr.Slider(
                minimum=0,
                maximum=20,
                step=1,
                value=0,
                label="seed (different seeds → different baseline outputs, "
                "but citeformer fab rate is always 0%)",
            )
            run_btn = gr.Button("Run adversarial demo", variant="primary")

        header_md = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("## citeformer (grammar-constrained)")
                constrained_md = gr.Markdown()
            with gr.Column():
                gr.Markdown("## baseline (no grammar)")
                baseline_md = gr.Markdown()

        sources_md = gr.Markdown(_sources_markdown())

        run_btn.click(
            fn=_run,
            inputs=[seed_slider],
            outputs=[header_md, constrained_md, baseline_md, sources_md],
        )

        gr.Markdown(
            "### How this works\n"
            "citeformer's HF backend wires [XGrammar](https://github.com/mlc-ai/xgrammar) "
            "into `transformers.model.generate()`. Before every decode step, "
            "XGrammar masks the logits for any token that would produce an "
            "invalid citation marker. The grammar builder emits a GBNF that "
            "enumerates the valid cite ids per call (`[1] | [2] | ... | [N]` "
            "for N sources in scope). So generating `[7]` when N=6 is "
            "token-impossible to sample — not hopefully-unlikely, not "
            "post-processed out.\n\n"
            "The full guarantee + benchmark data lives at "
            "[benchmarks/README.md](https://github.com/random-walks/citeformer/blob/main/benchmarks/README.md)."
        )

    return app


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
