"""Hand-labeled NLI calibration triples — (premise, hypothesis, label).

Each entry cites one of the six fixture papers in `benchmarks/fixtures/
ai_papers.json`. The premise is the abstract (exactly what our abstract-
premise sweep feeds the NLI model). The hypothesis is a short claim; the
label is True iff the claim is *genuinely entailed* by the abstract.

Labelling rules I tried to hold to:

- ``True`` means every content word of the hypothesis is recoverable from
  the abstract. Paraphrasing is fine; hedged wording is fine; but if the
  hypothesis adds a specific number / benchmark / claim the abstract
  doesn't contain, it's labelled False.
- ``False`` covers three shapes:
  1. Unrelated but plausible claims ("the paper introduces a new loss
     function" when the abstract doesn't).
  2. Contradictions ("the paper argues attention alone is insufficient"
     when the abstract says the opposite).
  3. Off-topic specifics ("trained on PubMed" when the abstract says
     WMT English-to-German).

This set isn't gold-standard — two people would disagree on a few edges.
It's meant to produce a useful-enough P/R curve, not beat human annotator
agreement. 36 pairs total, roughly balanced between True and False.
"""

from __future__ import annotations

#: Hand-labelled calibration triples. ``(paper_label, hypothesis, entailed)``.
#: The paper label must match an entry in
#: ``benchmarks/fixtures/ai_papers.json`` by its ``label`` field; the script
#: resolves the premise by looking up the paper's ``csl.abstract``.
CALIBRATION_TRIPLES: list[tuple[str, str, bool]] = [
    # --- Attention Is All You Need (Vaswani et al., 2017) --------------------
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "The Transformer relies solely on attention mechanisms, without recurrence or convolutions.",
        True,
    ),
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "The authors evaluate the model on WMT 2014 English-to-German translation.",
        True,
    ),
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "The Transformer is more parallelisable than existing sequence transduction architectures.",
        True,
    ),
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "The paper argues that attention alone is insufficient and must be combined with convolutions.",
        False,
    ),
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "The authors train their model on medical question-answering data.",
        False,
    ),
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "The paper introduces a novel reinforcement learning algorithm for game playing.",
        False,
    ),
    # --- BERT (Devlin et al., 2018) -----------------------------------------
    (
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "BERT pre-trains deep bidirectional representations from unlabeled text.",
        True,
    ),
    (
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "BERT can be fine-tuned for a variety of downstream tasks with minimal architecture changes.",
        True,
    ),
    (
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "The paper reports state-of-the-art results on GLUE and SQuAD benchmarks.",
        True,
    ),
    (
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "BERT is a left-to-right language model and cannot condition on right context.",
        False,
    ),
    (
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "The authors propose a new activation function called SELU.",
        False,
    ),
    (
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "The paper focuses on improving image classification using convolutional filters.",
        False,
    ),
    # --- GPT-3 (Brown et al., 2020) -----------------------------------------
    (
        "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
        "Scaling language models substantially improves task-agnostic few-shot performance.",
        True,
    ),
    (
        "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
        "GPT-3 is a 175-billion-parameter autoregressive language model.",
        True,
    ),
    (
        "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
        "The paper evaluates GPT-3 on a wide variety of NLP tasks in the few-shot setting.",
        True,
    ),
    (
        "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
        "GPT-3 is trained via reinforcement learning from human feedback.",
        False,
    ),
    (
        "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
        "The authors fine-tune GPT-3 separately on every downstream task.",
        False,
    ),
    (
        "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
        "The paper shows that smaller models outperform larger ones in few-shot learning.",
        False,
    ),
    # --- Chain-of-Thought (Wei et al., 2022) --------------------------------
    (
        "Chain-of-Thought Prompting (Wei et al., 2022)",
        "Chain-of-thought prompting elicits reasoning in large language models.",
        True,
    ),
    (
        "Chain-of-Thought Prompting (Wei et al., 2022)",
        "The technique involves prompting with a series of intermediate reasoning steps.",
        True,
    ),
    (
        "Chain-of-Thought Prompting (Wei et al., 2022)",
        "Chain-of-thought improves performance on arithmetic and commonsense reasoning tasks.",
        True,
    ),
    (
        "Chain-of-Thought Prompting (Wei et al., 2022)",
        "The method requires fine-tuning the model on reasoning traces.",
        False,
    ),
    (
        "Chain-of-Thought Prompting (Wei et al., 2022)",
        "Chain-of-thought is primarily evaluated on image classification.",
        False,
    ),
    (
        "Chain-of-Thought Prompting (Wei et al., 2022)",
        "The paper shows chain-of-thought hurts performance on large language models.",
        False,
    ),
    # --- LLaMA (Touvron et al., 2023) ---------------------------------------
    (
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "LLaMA is a collection of foundation language models ranging from 7B to 65B parameters.",
        True,
    ),
    (
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "LLaMA is trained on publicly available datasets exclusively.",
        True,
    ),
    (
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "The authors demonstrate competitive performance with state-of-the-art models using open data.",
        True,
    ),
    (
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "LLaMA is distributed only under a proprietary closed-source license.",
        False,
    ),
    (
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "The paper focuses on text-to-image generation using diffusion models.",
        False,
    ),
    (
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "LLaMA uses exclusively a mixture-of-experts architecture.",
        False,
    ),
    # --- QLoRA (Dettmers et al., 2023) --------------------------------------
    (
        "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)",
        "QLoRA enables finetuning of quantised large language models with low memory usage.",
        True,
    ),
    (
        "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)",
        "The method backpropagates gradients through a 4-bit quantised frozen base model into LoRA adapters.",
        True,
    ),
    (
        "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)",
        "The authors demonstrate finetuning a 65B model on a single GPU.",
        True,
    ),
    (
        "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)",
        "QLoRA requires at least 8 GPUs to finetune any model.",
        False,
    ),
    (
        "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)",
        "The paper only evaluates 8-bit quantisation, not 4-bit.",
        False,
    ),
    (
        "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)",
        "The authors argue that LoRA adapters are ineffective for LLM finetuning.",
        False,
    ),
    # --- Ambiguous / mid-score claims ---------------------------------------
    # These are deliberately tricky — claims that partially overlap with the
    # abstract, overreach on specifics the abstract doesn't state, or are
    # true-but-not-literally-entailed. They push some predictions into the
    # 0.3-0.7 band where threshold choice actually matters.
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "The Transformer achieves a BLEU score above 40 on English-to-French translation.",
        True,  # abstract says 41.8, so "above 40" is defensibly entailed
    ),
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "The model trains in under a day on eight GPUs.",
        False,  # abstract says 3.5 days — "under a day" overreaches
    ),
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "The Transformer outperforms recurrent models on translation.",
        True,  # abstract claims superiority over recurrent encoder-decoder models
    ),
    (
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "BERT achieves above 90% accuracy on sentiment analysis.",
        False,  # not in the abstract — fabricated specific
    ),
    (
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "BERT can improve the state of the art on many language understanding tasks.",
        True,  # defensibly paraphrases the abstract's claim
    ),
    (
        "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
        "The paper reports that GPT-3's performance grows with scale.",
        True,  # "scaling language models substantially improves" entails this
    ),
    (
        "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
        "GPT-3 has a 32K token context window.",
        False,  # not in abstract, specific and wrong
    ),
    (
        "Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)",
        "The model can perform translation, question-answering, and cloze tasks in a few-shot manner.",
        True,  # abstract mentions "wide variety of NLP tasks"; these are canonical ones
    ),
    (
        "Chain-of-Thought Prompting (Wei et al., 2022)",
        "Chain-of-thought prompting yields gains especially on math word problems.",
        True,  # abstract mentions arithmetic reasoning
    ),
    (
        "Chain-of-Thought Prompting (Wei et al., 2022)",
        "Chain-of-thought prompting improves reasoning only on models above 100B parameters.",
        False,  # specific threshold not in abstract; overreach
    ),
    (
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "LLaMA-13B outperforms GPT-3 on most benchmarks.",
        False,  # the abstract summary we have doesn't spell out this specific comparison
    ),
    (
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "LLaMA models range in size roughly an order of magnitude.",
        True,  # 7B to 65B is ~9x, defensibly "an order of magnitude"
    ),
    (
        "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)",
        "QLoRA reduces memory by using 4-bit NormalFloat quantisation.",
        True,  # the abstract describes 4-bit quantisation of frozen base
    ),
    (
        "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)",
        "QLoRA reaches ChatGPT-level performance on the Vicuna benchmark.",
        False,  # specific benchmark / comparison not spelled out in fixture abstract
    ),
]
