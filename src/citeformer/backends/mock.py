"""Scripted backend used in unit tests.

`MockBackend` emits predetermined responses given a prompt; unknown prompts fall
back to a deterministic echo that respects the source-id range. It lets the
orchestration layer be tested end-to-end without loading a real model.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from citeformer.backends.base import Backend
from citeformer.core import MarkerStyle, Policy, Source

_MARKER_TEMPLATES: dict[MarkerStyle, str] = {
    MarkerStyle.BRACKET: "[1]",
    MarkerStyle.PAREN: "(1)",
    MarkerStyle.CURLY: "{1}",
    MarkerStyle.CARET: "^1",
}


class MockBackend(Backend):
    """Scripted backend for tests.

    Construct with a `responses` mapping from prompt to pre-canned output. Any
    prompt not in the mapping gets a deterministic echo of the form
    `"Mock response for: <prompt> [1]."` — which satisfies the `REQUIRED`
    policy trivially when at least one source is in scope. The marker in the
    fallback honours the ``marker_style`` option so tests for non-bracket
    styles see the right delimiters in the echoed output.

    Attributes:
        responses: Pre-canned responses keyed by exact prompt string.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        """Construct a MockBackend.

        Args:
            responses: Optional mapping from prompt to canned response.
        """
        self.responses: dict[str, str] = dict(responses) if responses else {}

    def generate(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> str:
        """Return a scripted response or a deterministic fallback.

        The fallback echo emits a marker (shape determined by the
        ``marker_style`` option, default ``[1]``) when sources are in scope,
        so that downstream parsers and verifiers see at least one citation.
        When there are no sources, it returns the echo with no marker
        regardless of policy — the caller is responsible for not invoking
        `generate()` with an empty source list under `REQUIRED`.
        """
        del policy  # Mock doesn't vary output by policy.
        if prompt in self.responses:
            return self.responses[prompt]
        if sources:
            marker_style = options.get("marker_style", MarkerStyle.BRACKET)
            marker = _MARKER_TEMPLATES[marker_style]
            return f"Mock response for: {prompt!r} {marker}."
        return f"Mock response for: {prompt!r}."

    def stream(
        self,
        prompt: str,
        sources: list[Source],
        policy: Policy,
        **options: Any,
    ) -> Iterator[str]:
        """Yield the mock response in small chunks to exercise stream consumers.

        Splits the scripted response at word boundaries so downstream
        streaming tests see more than one chunk without relying on tokenizer
        behavior.
        """
        text = self.generate(prompt=prompt, sources=sources, policy=policy, **options)
        # Rough word-ish split: 10 char groups keep the chunk count moderate for
        # short outputs while still giving >1 chunk for anything non-trivial.
        chunk_size = 10
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]
