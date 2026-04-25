"""Async-surface tests (ADR-014).

Three layers under test:

1. **Backend ABC defaults** — ``agenerate`` falls back to ``generate`` via
   ``asyncio.to_thread``; ``astream`` wraps the sync iterator the same
   way. MockBackend exercises both.
2. **Citeformer orchestrator** — ``cf.agenerate()`` returns a full
   :class:`GenerationResult`; ``cf.astream()`` returns an
   :class:`AsyncStreamingResult` with ``async for`` + ``await
   .finalize()`` symmetry.
3. **Native overrides** — ``OpenAIBackend.agenerate`` / ``astream`` use
   the lazy ``async_client``; ``AnthropicBackend.agenerate`` /
   ``astream`` use the async streaming context manager.

Pytest config has ``asyncio_mode = "auto"`` so plain ``async def
test_…`` functions just work.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from citeformer import (
    AsyncStreamingResult,
    Citeformer,
    GenerationResult,
    MarkerStyle,
    Policy,
    Source,
)
from citeformer.backends import MockBackend
from citeformer.backends.anthropic import AnthropicBackend
from citeformer.backends.openai import OpenAIBackend


@pytest.fixture
def sources() -> list[Source]:
    return [
        Source(metadata={"id": f"s{i}", "type": "book"}, content=f"Body {i}.") for i in range(1, 4)
    ]


# ----- Backend ABC defaults -------------------------------------------------


async def test_abc_default_agenerate_delegates_to_generate(sources: list[Source]) -> None:
    """MockBackend's ``agenerate`` is the inherited default — wraps ``generate``
    via ``asyncio.to_thread``. Same return value as the sync path."""
    backend = MockBackend()
    text = await backend.agenerate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "[1]" in text


async def test_abc_default_astream_yields_chunks(sources: list[Source]) -> None:
    """``astream`` default wraps the sync stream's iterator chunk-by-chunk."""
    backend = MockBackend()
    chunks: list[str] = []
    async for chunk in backend.astream(prompt="hi", sources=sources, policy=Policy.AUTO):
        chunks.append(chunk)
    assert chunks  # MockBackend emits multiple chunks
    assert "[1]" in "".join(chunks)


async def test_abc_default_astream_preserves_chunk_order_and_count(
    sources: list[Source],
) -> None:
    """The async wrapper must yield exactly the same chunk sequence as the
    sync ``stream()`` — no merging, no splitting."""
    backend = MockBackend()
    sync_chunks = list(backend.stream(prompt="hi", sources=sources, policy=Policy.AUTO))
    async_chunks: list[str] = []
    async for chunk in backend.astream(prompt="hi", sources=sources, policy=Policy.AUTO):
        async_chunks.append(chunk)
    assert async_chunks == sync_chunks


# ----- Citeformer orchestrator agenerate -----------------------------------


async def test_orchestrator_agenerate_returns_generation_result(
    sources: list[Source],
) -> None:
    """``cf.agenerate()`` returns a fully-populated GenerationResult, identical
    in structure to ``cf.generate()``."""
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    result = await cf.agenerate(prompt="hi", sources=sources)
    assert isinstance(result, GenerationResult)
    assert result.text  # has content
    assert result.citations  # parsed at least one
    assert all(1 <= c.source_id <= 3 for c in result.citations)


async def test_orchestrator_agenerate_threads_marker_style(
    sources: list[Source],
) -> None:
    """marker_style override propagates through the async path."""
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    result = await cf.agenerate(prompt="hi", sources=sources, marker_style=MarkerStyle.PAREN)
    assert "(1)" in result.text


async def test_orchestrator_agenerate_threads_policy_override(
    sources: list[Source],
) -> None:
    """Per-call ``policy=`` override works in the async path."""
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.REQUIRED)
    result = await cf.agenerate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert result.text


# ----- Citeformer orchestrator astream + AsyncStreamingResult --------------


async def test_orchestrator_astream_returns_async_streaming_result(
    sources: list[Source],
) -> None:
    """``cf.astream(...)`` returns an :class:`AsyncStreamingResult` synchronously."""
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.astream(prompt="hi", sources=sources)
    assert isinstance(stream, AsyncStreamingResult)


async def test_async_streaming_result_iterates_then_finalizes(
    sources: list[Source],
) -> None:
    """End-to-end async streaming: iterate chunks, await finalize() to get
    the full result."""
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.astream(prompt="hi", sources=sources)
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    assert chunks
    result = await stream.finalize()
    assert result.text == "".join(chunks)
    assert result.citations
    assert result.references


async def test_async_streaming_result_finalize_is_idempotent(
    sources: list[Source],
) -> None:
    """Multiple ``await finalize()`` calls return the same instance."""
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.astream(prompt="hi", sources=sources)
    async for _ in stream:
        pass
    a = await stream.finalize()
    b = await stream.finalize()
    assert a is b


async def test_async_streaming_result_finalize_without_iter_exhausts(
    sources: list[Source],
) -> None:
    """Calling ``finalize()`` without iterating first auto-exhausts."""
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.astream(prompt="hi", sources=sources)
    result = await stream.finalize()
    assert result.text  # has content despite no async-for above


async def test_async_streaming_result_text_property_updates_during_iteration(
    sources: list[Source],
) -> None:
    """``stream.text`` exposes accumulated text after each chunk."""
    cf = Citeformer(backend=MockBackend(), citation_policy=Policy.AUTO)
    stream = cf.astream(prompt="hi", sources=sources)
    last_len = 0
    async for _chunk in stream:
        assert len(stream.text) >= last_len
        last_len = len(stream.text)
    assert stream.text  # non-empty after consumption


async def test_orchestrator_async_threads_usage_when_backend_exposes_it(
    sources: list[Source],
) -> None:
    """If the backend populates ``last_usage`` during ``agenerate``, the
    orchestrator threads it onto ``GenerationResult.usage`` — same path as
    the sync version."""
    fake_payload = json.dumps(
        {
            "segments": [
                {"text": "Alpha claim.", "citations": [1]},
            ]
        }
    )
    fake_async = _AsyncFakeOpenAI(
        fake_payload,
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=12),
    )
    backend = OpenAIBackend(model="gpt-4o-mini", async_client=fake_async)
    cf = Citeformer(backend=backend, citation_policy=Policy.AUTO)
    result = await cf.agenerate(prompt="hi", sources=sources)
    assert result.usage is not None
    assert result.usage.input_tokens == 42
    assert result.usage.output_tokens == 12


# ----- OpenAIBackend native async ------------------------------------------


class _AsyncFakeOpenAI:
    """Minimal stand-in for ``openai.AsyncOpenAI`` — async ``create``."""

    def __init__(self, response_content: str, *, usage: Any = None) -> None:
        self._response = response_content
        self._usage = usage
        self.last_payload: dict[str, Any] | None = None

        async def _create(**kwargs: Any) -> Any:
            self.last_payload = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response_content))],
                usage=self._usage,
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


async def test_openai_native_agenerate_uses_async_client(sources: list[Source]) -> None:
    """``agenerate`` must hit the ``async_client``, not the sync client.

    We make the sync client raise to prove the async path doesn't touch it."""

    class _ExplodingSync:
        chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **k: (_ for _ in ()).throw(
                    AssertionError("sync client called from async path")
                )
            )
        )

    fake_async = _AsyncFakeOpenAI(json.dumps({"segments": [{"text": "Claim.", "citations": [1]}]}))
    backend = OpenAIBackend(
        model="gpt-4o-mini",
        client=_ExplodingSync(),
        async_client=fake_async,
    )
    text = await backend.agenerate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "[1]" in text


async def test_openai_native_agenerate_extracts_usage(sources: list[Source]) -> None:
    fake_async = _AsyncFakeOpenAI(
        json.dumps({"segments": [{"text": "Claim.", "citations": [1]}]}),
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
    )
    backend = OpenAIBackend(model="gpt-4o-mini", async_client=fake_async)
    await backend.agenerate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 120


async def test_openai_native_astream_yields_chunks(sources: list[Source]) -> None:
    fake_async = _AsyncFakeOpenAI(
        json.dumps(
            {
                "segments": [
                    {"text": "First claim.", "citations": [1]},
                    {"text": "Second claim.", "citations": [2]},
                ]
            }
        )
    )
    backend = OpenAIBackend(model="gpt-4o-mini", async_client=fake_async)
    chunks: list[str] = []
    async for chunk in backend.astream(prompt="hi", sources=sources, policy=Policy.AUTO):
        chunks.append(chunk)
    assert len(chunks) >= 2
    text = "".join(chunks)
    assert "[1]" in text and "[2]" in text


async def test_openai_native_agenerate_rejects_empty_sources() -> None:
    fake_async = _AsyncFakeOpenAI("{}")
    backend = OpenAIBackend(model="gpt-4o-mini", async_client=fake_async)
    with pytest.raises(ValueError, match="at least 1 source"):
        await backend.agenerate(prompt="hi", sources=[], policy=Policy.AUTO)


# ----- AnthropicBackend native async ---------------------------------------


class _AsyncFakeAnthropic:
    """Minimal stand-in for ``anthropic.AsyncAnthropic`` — async ``create``."""

    def __init__(self, content_blocks: list[Any], usage: Any = None) -> None:
        self._blocks = content_blocks
        self._usage = usage
        self.last_kwargs: dict[str, Any] | None = None

        async def _create(**kwargs: Any) -> Any:
            self.last_kwargs = kwargs
            return SimpleNamespace(content=self._blocks, usage=self._usage)

        self.messages = SimpleNamespace(create=_create)


async def test_anthropic_native_agenerate_uses_async_client(sources: list[Source]) -> None:
    blocks = [
        SimpleNamespace(
            type="text",
            text="Alpha.",
            citations=[SimpleNamespace(document_index=0)],
        ),
        SimpleNamespace(
            type="text",
            text="Beta.",
            citations=[SimpleNamespace(document_index=2)],
        ),
    ]
    fake = _AsyncFakeAnthropic(blocks)
    backend = AnthropicBackend(model="claude-sonnet-4-6", async_client=fake)
    text = await backend.agenerate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert "[1]" in text
    assert "[3]" in text


async def test_anthropic_native_agenerate_populates_rich_citations(
    sources: list[Source],
) -> None:
    """Async path must populate ``last_rich_citations`` the same way as sync —
    Citation objects in the orchestrator pick up ``cited_text`` / ``source_span``
    / ``document_title`` regardless of async vs sync entry."""
    blocks = [
        SimpleNamespace(
            type="text",
            text="Claim.",
            citations=[
                SimpleNamespace(
                    document_index=0,
                    cited_text="cited span",
                    start_char_index=10,
                    end_char_index=20,
                    document_title="Title 1",
                )
            ],
        ),
    ]
    fake = _AsyncFakeAnthropic(blocks)
    backend = AnthropicBackend(model="claude-sonnet-4-6", async_client=fake)
    cf = Citeformer(backend=backend, citation_policy=Policy.AUTO)
    result = await cf.agenerate(prompt="hi", sources=sources)
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.cited_text == "cited span"
    assert citation.source_span == (10, 20)
    assert citation.document_title == "Title 1"


async def test_anthropic_native_agenerate_extracts_usage(sources: list[Source]) -> None:
    fake = _AsyncFakeAnthropic(
        content_blocks=[],
        usage=SimpleNamespace(input_tokens=600, output_tokens=80),
    )
    backend = AnthropicBackend(model="claude-sonnet-4-6", async_client=fake)
    await backend.agenerate(prompt="hi", sources=sources, policy=Policy.AUTO)
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 600
    assert backend.last_usage.output_tokens == 80


class _AsyncFakeAnthropicStream:
    """Stand-in for the ``AsyncAnthropic.messages.stream`` async context manager."""

    def __init__(self, events: list[Any], final_message: Any) -> None:
        self._events = events
        self._final_message = final_message
        self.last_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> _AsyncFakeAnthropicStream:
        self.last_kwargs = kwargs
        return self

    async def __aenter__(self) -> _AsyncFakeAnthropicStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> _AsyncFakeAnthropicStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration  # noqa: B904

    async def get_final_message(self) -> Any:
        return self._final_message


class _AsyncFakeAnthropicWithStream:
    """Stand-in async client exposing both ``create`` and ``stream``."""

    def __init__(self, *, stream: _AsyncFakeAnthropicStream) -> None:
        async def _create(**kwargs: Any) -> Any:
            return SimpleNamespace(content=[], usage=None)

        self.messages = SimpleNamespace(create=_create, stream=stream)


async def test_anthropic_native_astream_yields_per_block(sources: list[Source]) -> None:
    """Native async streaming: one yielded chunk per ``content_block_stop``,
    citation markers attached at block boundaries."""
    block_one = SimpleNamespace(
        type="text",
        text="Alpha claim.",
        citations=[SimpleNamespace(document_index=0)],
    )
    block_two = SimpleNamespace(
        type="text",
        text="Beta claim.",
        citations=[SimpleNamespace(document_index=2)],
    )
    events = [
        SimpleNamespace(type="content_block_stop", content_block=block_one),
        SimpleNamespace(type="content_block_stop", content_block=block_two),
    ]
    final = SimpleNamespace(
        content=[block_one, block_two],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )
    stream = _AsyncFakeAnthropicStream(events, final)
    fake = _AsyncFakeAnthropicWithStream(stream=stream)
    backend = AnthropicBackend(model="claude-sonnet-4-6", async_client=fake)
    chunks: list[str] = []
    async for chunk in backend.astream(prompt="hi", sources=sources, policy=Policy.AUTO):
        chunks.append(chunk)
    assert len(chunks) == 2
    text = "".join(chunks)
    assert "[1]" in text
    assert "[3]" in text
    # ``last_usage`` populates from ``await stream.get_final_message()``.
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens == 100


async def test_anthropic_native_astream_falls_back_when_no_stream_method(
    sources: list[Source],
) -> None:
    """An async client that only mocks ``create`` (no ``stream``) should fall
    back to a single-chunk yield via ``agenerate``."""
    blocks = [
        SimpleNamespace(
            type="text",
            text="Claim.",
            citations=[SimpleNamespace(document_index=0)],
        ),
    ]
    fake = _AsyncFakeAnthropic(blocks)
    backend = AnthropicBackend(model="claude-sonnet-4-6", async_client=fake)
    chunks: list[str] = []
    async for chunk in backend.astream(prompt="hi", sources=sources, policy=Policy.AUTO):
        chunks.append(chunk)
    assert any("[1]" in c for c in chunks)


# ----- Lazy async-client construction --------------------------------------


def test_openai_async_client_is_lazy() -> None:
    """Sync-only callers must not pay the ``AsyncOpenAI()`` cost."""
    fake_sync = SimpleNamespace(chat=SimpleNamespace())
    backend = OpenAIBackend(model="gpt-4o-mini", client=fake_sync)
    # Internal cache empty; override unset.
    assert backend._async_client_cache is None
    assert backend._async_client_override is None


def test_openai_async_client_uses_override_when_provided() -> None:
    fake_async = _AsyncFakeOpenAI("{}")
    backend = OpenAIBackend(
        model="gpt-4o-mini",
        client=SimpleNamespace(chat=SimpleNamespace()),
        async_client=fake_async,
    )
    assert backend.async_client is fake_async


def test_anthropic_async_client_is_lazy() -> None:
    fake_sync = SimpleNamespace(messages=SimpleNamespace())
    backend = AnthropicBackend(model="claude-sonnet-4-6", client=fake_sync)
    assert backend._async_client_cache is None
    assert backend._async_client_override is None


def test_anthropic_async_client_uses_override_when_provided() -> None:
    fake_async = _AsyncFakeAnthropic([])
    backend = AnthropicBackend(
        model="claude-sonnet-4-6",
        client=SimpleNamespace(messages=SimpleNamespace()),
        async_client=fake_async,
    )
    assert backend.async_client is fake_async
