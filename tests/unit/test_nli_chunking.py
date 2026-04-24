"""Unit tests for the chunked-premise logic in `citeformer.verify.nli`.

The real model (DeBERTa-v3) is a 850 MB load so these tests use a fake
tokenizer that splits on whitespace. The point is to pin the chunking
math — window sizes, strides, overlap — separately from the NLI scoring,
which is exercised in ``tests/integration/test_verify_nli.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from citeformer.verify.nli import NLIModel, NLIResult, _split_premise_into_windows


class _FakeTokenizer:
    """Whitespace-splitting tokenizer that satisfies the subset of the HF API
    that `_split_premise_into_windows` uses: ``__call__`` returning a dict with
    ``input_ids`` (a list of 'token' strings), and ``decode`` joining them.
    """

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[str]]:
        del add_special_tokens  # fake: every token stays in output
        return {"input_ids": text.split()}

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(ids)


# --- _split_premise_into_windows ------------------------------------------


def test_short_premise_returns_single_window_unchanged() -> None:
    premise = "a b c d e"
    tok = _FakeTokenizer()
    windows = _split_premise_into_windows(premise, tok, max_tokens=10, stride=5)
    assert windows == [premise]


def test_exact_max_tokens_returns_single_window() -> None:
    premise = " ".join(str(i) for i in range(10))
    tok = _FakeTokenizer()
    windows = _split_premise_into_windows(premise, tok, max_tokens=10, stride=5)
    assert len(windows) == 1
    assert windows[0] == premise


def test_longer_premise_splits_into_overlapping_windows() -> None:
    # 20 tokens, window 10, stride 5 → expect windows starting at 0, 5, 10.
    tokens = [str(i) for i in range(20)]
    premise = " ".join(tokens)
    tok = _FakeTokenizer()
    windows = _split_premise_into_windows(premise, tok, max_tokens=10, stride=5)
    # Expect windows: [0..9], [5..14], [10..19]
    assert len(windows) == 3
    assert windows[0] == " ".join(tokens[0:10])
    assert windows[1] == " ".join(tokens[5:15])
    assert windows[2] == " ".join(tokens[10:20])


def test_stride_equal_max_tokens_is_non_overlapping() -> None:
    tokens = [str(i) for i in range(15)]
    premise = " ".join(tokens)
    tok = _FakeTokenizer()
    windows = _split_premise_into_windows(premise, tok, max_tokens=5, stride=5)
    # Non-overlapping: [0..4], [5..9], [10..14]
    assert len(windows) == 3
    assert windows[0] == " ".join(tokens[0:5])
    assert windows[1] == " ".join(tokens[5:10])
    assert windows[2] == " ".join(tokens[10:15])


def test_tail_window_is_short() -> None:
    # 12 tokens, window 10, stride 10 → [0..9], [10..11]
    tokens = [str(i) for i in range(12)]
    premise = " ".join(tokens)
    tok = _FakeTokenizer()
    windows = _split_premise_into_windows(premise, tok, max_tokens=10, stride=10)
    assert len(windows) == 2
    assert windows[0] == " ".join(tokens[0:10])
    assert windows[1] == " ".join(tokens[10:12])  # tail of 2 tokens


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        _split_premise_into_windows("x", _FakeTokenizer(), max_tokens=0, stride=1)
    with pytest.raises(ValueError, match="must be > 0"):
        _split_premise_into_windows("x", _FakeTokenizer(), max_tokens=1, stride=0)


# --- NLIModel constructor validation --------------------------------------


def test_nli_model_rejects_invalid_chunk_params() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        NLIModel(max_premise_tokens=0)
    with pytest.raises(ValueError, match="must be > 0"):
        NLIModel(chunk_stride=0)
    with pytest.raises(ValueError, match="must not exceed"):
        NLIModel(max_premise_tokens=100, chunk_stride=200)


def test_nli_model_defaults_to_chunking_off() -> None:
    """Default is raw 512-token truncation for score stability.

    Max-over-windows inflates false positives on unrelated claims because
    each extra window is another chance for noise to cross threshold.
    Users opt in with ``chunk_premise=True`` when they want long-document
    scoring, usually paired with a bumped threshold.
    """
    model = NLIModel(device="cpu")
    assert model.chunk_premise is False
    assert model.max_premise_tokens == 400
    assert model.chunk_stride == 300


# --- Chunked reduction over fake NLI --------------------------------------


class _StubNLI(NLIModel):
    """NLIModel that bypasses actual weight loading; scoring comes from a
    user-supplied callable. Used to exercise the chunk-expand-reduce plumbing
    without any ML dependency heat.
    """

    def __init__(
        self,
        score_fn: Any,
        *,
        chunk_premise: bool = True,
        max_premise_tokens: int = 10,
        chunk_stride: int = 10,
    ) -> None:
        # Skip parent __init__ to avoid torch probe; set only what we use.
        self.model_name = "fake"
        self.device = "cpu"
        self.batch_size = 8
        self.chunk_premise = chunk_premise
        self.max_premise_tokens = max_premise_tokens
        self.chunk_stride = chunk_stride
        self._score_fn = score_fn

    def _entail_batch_truncated(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        return [self._score_fn(premise, hyp) for premise, hyp in pairs]


def _highest_first_token_score(premise: str, hypothesis: str) -> NLIResult:
    """Fake scorer: entailment = 0.95 if premise starts with 'match', else 0.1."""
    del hypothesis
    if premise.split()[:1] == ["match"]:
        return NLIResult(entailment=0.95, neutral=0.03, contradiction=0.02)
    return NLIResult(entailment=0.1, neutral=0.8, contradiction=0.1)


def test_chunked_takes_max_across_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long premise where only ONE window contains the signal should still
    produce the high entailment, because we take the max across windows.
    """
    from citeformer.verify import nli as nli_module

    # Patch _load_nli so the fake tokenizer is returned — the NLI model itself
    # is never invoked because our _StubNLI overrides _entail_batch_truncated.
    monkeypatch.setattr(
        nli_module,
        "_load_nli",
        lambda model_name, device: (_FakeTokenizer(), object(), {}),
    )

    model = _StubNLI(
        score_fn=_highest_first_token_score,
        max_premise_tokens=5,
        chunk_stride=5,
    )

    # 20 tokens; only the window starting with "match" scores high.
    premise_tokens = ["foo"] * 10 + ["match"] + ["bar"] * 9
    premise = " ".join(premise_tokens)
    hypothesis = "anything"

    [result] = model.entail_batch([(premise, hypothesis)])
    assert result.entailment == pytest.approx(0.95)


def test_chunked_short_premise_is_single_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Premise under window size should score exactly once (no extra cost)."""
    from citeformer.verify import nli as nli_module

    call_count = 0

    def tracking_scorer(premise: str, hypothesis: str) -> NLIResult:
        nonlocal call_count
        call_count += 1
        del premise, hypothesis
        return NLIResult(entailment=0.5, neutral=0.25, contradiction=0.25)

    monkeypatch.setattr(
        nli_module,
        "_load_nli",
        lambda model_name, device: (_FakeTokenizer(), object(), {}),
    )

    model = _StubNLI(
        score_fn=tracking_scorer,
        max_premise_tokens=100,
        chunk_stride=50,
    )
    [result] = model.entail_batch([("short premise", "claim")])
    assert call_count == 1
    assert result.entailment == 0.5


def test_chunking_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """chunk_premise=False keeps old truncation behavior (no windowing)."""
    from citeformer.verify import nli as nli_module

    monkeypatch.setattr(
        nli_module,
        "_load_nli",
        lambda model_name, device: (_FakeTokenizer(), object(), {}),
    )

    call_count = 0

    def tracking_scorer(premise: str, hypothesis: str) -> NLIResult:
        nonlocal call_count
        call_count += 1
        del premise, hypothesis
        return NLIResult(entailment=0.5, neutral=0.25, contradiction=0.25)

    model = _StubNLI(
        score_fn=tracking_scorer,
        chunk_premise=False,
        max_premise_tokens=5,
        chunk_stride=5,
    )

    # Long premise, chunking off → exactly 1 call (truncation path).
    premise_tokens = ["foo"] * 50
    premise = " ".join(premise_tokens)
    [_] = model.entail_batch([(premise, "claim")])
    assert call_count == 1


def test_chunked_preserves_order_across_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each original pair's result lands at the original index regardless of
    how many windows each pair contributes.
    """
    from citeformer.verify import nli as nli_module

    monkeypatch.setattr(
        nli_module,
        "_load_nli",
        lambda model_name, device: (_FakeTokenizer(), object(), {}),
    )

    def score_fn(premise: str, hypothesis: str) -> NLIResult:
        # Tag entailment with the hypothesis digit so we can verify ordering.
        digit = int(hypothesis.split()[0])
        return NLIResult(
            entailment=digit / 10.0,
            neutral=0.0,
            contradiction=1 - digit / 10.0,
        )

    model = _StubNLI(
        score_fn=score_fn,
        max_premise_tokens=5,
        chunk_stride=5,
    )
    pairs = [
        (" ".join(["x"] * 12), "1 claim"),
        (" ".join(["x"] * 3), "2 claim"),
        (" ".join(["x"] * 20), "3 claim"),
    ]
    results = model.entail_batch(pairs)
    assert len(results) == 3
    assert results[0].entailment == pytest.approx(0.1)
    assert results[1].entailment == pytest.approx(0.2)
    assert results[2].entailment == pytest.approx(0.3)


def test_empty_pairs_returns_empty_list() -> None:
    # No NLI stub needed — entail_batch short-circuits on empty input.
    model = _StubNLI(score_fn=lambda p, h: NLIResult(0, 0, 0))
    assert model.entail_batch([]) == []


# Unused-import guard so ruff doesn't remove SimpleNamespace (used elsewhere).
_ = SimpleNamespace
