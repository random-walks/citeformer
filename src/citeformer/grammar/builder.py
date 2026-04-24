"""Citation-grammar construction — §10.1 contract implementation.

Emits GBNF grammars consumable by XGrammar / llguidance / llama.cpp. The point
of this module is the `cite-id` rule:

    cite-id ::= OPEN ("1" | "2" | ... | "N") CLOSE

where ``OPEN`` / ``CLOSE`` are the delimiters for the chosen
:class:`~citeformer.core.MarkerStyle` (default ``[…]``), and ``N`` is
dynamically set to ``len(sources)`` per generate() call. That's what
makes a fabricated citation (``[N+k]`` for any ``k > 0``) a logit-level
impossibility when the downstream backend masks against this grammar —
regardless of which marker shape is chosen.

Three policies sit on top:

- `REQUIRED`: every sentence must end `content cite-group sent-end`. The model
  can't close a sentence without citing. ``content`` is bounded to
  ``max_content_chars`` (default 240) so small models can't stall in content
  state indefinitely — see ``docs/decisions/009-bounded-content-required.md``.
- `QUOTES_ONLY`: only quoted spans require a trailing `cite-group`. Narrative
  prose can stand alone.
- `AUTO`: `cite-group` is allowed anywhere but not required. The `verify()`
  coverage check surfaces missing citations post-hoc instead.

Format note: we emit GBNF (the GGML grammar format used by llama.cpp and
xgrammar) rather than Lark because xgrammar's parser expects `::=` not `:`.
Semantically equivalent; just a syntax swap. Semantic validity is exercised at
integration time — the HF backend's `test_hf_backend_grammar_compiles` compiles
the emitted string with xgrammar, which is the authoritative parser.
"""

from __future__ import annotations

from dataclasses import dataclass

from citeformer.core import MarkerStyle, Policy

#: Default soft progression bound on the REQUIRED-policy `content` rule, in
#: characters. Once a sentence has accumulated this many non-terminating chars,
#: xgrammar masks everything except a citation bracket, forcing the model to
#: progress instead of stalling in content state. See ADR-009.
DEFAULT_MAX_CONTENT_CHARS = 240


@dataclass(frozen=True)
class MarkerSpec:
    """Delimiter configuration for a :class:`MarkerStyle`.

    Attributes:
        open_char: Single character that opens a marker (e.g. ``[`` / ``(`` /
            ``{`` / ``^``). Excluded from the grammar's ``text`` / ``content``
            character classes so the parser knows when a marker starts.
        close_char: Single character that closes a marker, or empty string
            for open-ended markers like ``^N``. Not part of the exclusion
            set because it can appear in regular prose.
    """

    open_char: str
    close_char: str


#: Per-style delimiters. Referenced by :func:`build_grammar` to parameterise
#: the ``cite-id`` terminal and adjacent exclusion sets.
MARKER_SPECS: dict[MarkerStyle, MarkerSpec] = {
    MarkerStyle.BRACKET: MarkerSpec(open_char="[", close_char="]"),
    MarkerStyle.PAREN: MarkerSpec(open_char="(", close_char=")"),
    MarkerStyle.CURLY: MarkerSpec(open_char="{", close_char="}"),
    MarkerStyle.CARET: MarkerSpec(open_char="^", close_char=""),
}


@dataclass(frozen=True)
class Grammar:
    """A citation-constraining GBNF grammar for one generation call.

    Attributes:
        gbnf: Full GBNF grammar string. Accepted by XGrammar's
            `compile_grammar()` and by llama.cpp's native GBNF support.
        cite_ids: 1-indexed source ids that the grammar admits, in ascending
            order. Derived from `len(sources)` at build time.
        policy: Enforcement policy that shaped the grammar body.
        marker_style: Delimiter shape used by the ``cite-id`` terminal.
            Defaults to :attr:`MarkerStyle.BRACKET` to match §10.1's canonical
            ``[N]`` shape.
        root_rule: The entry rule name. Always `"root"` — GBNF convention; also
            xgrammar's default so no explicit `root_rule_name` override needed.
        max_content_chars: Upper bound on `content` repetition for the
            REQUIRED policy. ``None`` means unbounded (legacy ``+``). For
            AUTO and QUOTES_ONLY this field is ``None`` because the bound
            only applies to REQUIRED.
    """

    gbnf: str
    cite_ids: tuple[int, ...]
    policy: Policy
    marker_style: MarkerStyle = MarkerStyle.BRACKET
    root_rule: str = "root"
    max_content_chars: int | None = None


def _shared_tail() -> str:
    """`cite-group` + whitespace rules, identical across marker styles."""
    return 'cite-group ::= cite-id (ws cite-id)*\nws ::= " "\n'


def _auto_body(open_char: str) -> str:
    return f"root ::= (text | cite-group)+\ntext ::= [^{open_char}]+\n"


def _quotes_only_body(open_char: str) -> str:
    return (
        "root ::= (text | quoted-cite)+\n"
        f'text ::= [^{open_char}"]+\n'
        "quoted-cite ::= quote cite-group\n"
        'quote ::= "\\"" [^"]* "\\""\n'
    )


def _required_body(open_char: str, max_content_chars: int | None) -> str:
    """Render the REQUIRED-policy grammar body with an optional content bound.

    When `max_content_chars` is `None` the body emits the legacy unbounded
    ``content ::= [^{open_char}.!?]+`` rule. When a positive integer, it emits a
    bounded repetition ``[^{open_char}.!?]{1, N}`` — xgrammar and llama.cpp
    both accept this syntax as of their 2026 releases.
    """
    if max_content_chars is None:
        content_rule = f"content ::= [^{open_char}.!?]+"
    else:
        if max_content_chars < 1:
            raise ValueError(f"max_content_chars must be >= 1 or None, got {max_content_chars}")
        content_rule = f"content ::= [^{open_char}.!?]{{1, {max_content_chars}}}"
    return (
        "root ::= sentence (ws sentence)*\n"
        "sentence ::= content cite-group sent-end\n"
        f"{content_rule}\n"
        'sent-end ::= "." | "!" | "?"\n'
    )


def _cite_id_rule(n_sources: int, spec: MarkerSpec) -> str:
    """Render the `cite-id` rule for `n_sources` sources with `spec` delimiters.

    For ``n_sources=3, spec=MARKER_SPECS[BRACKET]`` returns::

        cite-id ::= "[" ("1" | "2" | "3") "]"

    which is the §10.1 contract's load-bearing rule (the bracket variant is
    the canonical shape). Swapping the spec flips the delimiters; the
    alternatives enumeration — and therefore the structural fabrication-
    impossibility guarantee — is identical across marker styles.
    """
    if n_sources < 1:
        raise ValueError(f"n_sources must be >= 1, got {n_sources}")
    alternatives = " | ".join(f'"{i}"' for i in range(1, n_sources + 1))
    if spec.close_char:
        return f'cite-id ::= "{spec.open_char}" ({alternatives}) "{spec.close_char}"'
    # Open-ended markers (``^N``) have no closing delimiter.
    return f'cite-id ::= "{spec.open_char}" ({alternatives})'


def build_grammar(
    n_sources: int,
    policy: Policy,
    *,
    max_content_chars: int | None = DEFAULT_MAX_CONTENT_CHARS,
    marker_style: MarkerStyle = MarkerStyle.BRACKET,
) -> Grammar:
    """Build the citation-constraining GBNF grammar for a generation call.

    Args:
        n_sources: Number of sources in scope. Must be >= 1. Determines the
            set of valid cite ids (1..n_sources inclusive).
        policy: Citation enforcement policy.
        max_content_chars: Soft progression bound for the REQUIRED policy.
            After this many characters of content since the last sentence
            terminator, the grammar forces the model into a citation —
            closing the ADR-007 stall loophole. Set ``None`` to disable
            bounding (legacy behavior; risks stall on small models). Ignored
            for AUTO and QUOTES_ONLY policies, which have no sentence-level
            shape to bound. See
            ``docs/decisions/009-bounded-content-required.md``.
        marker_style: Visual shape for inline markers. Defaults to
            :attr:`MarkerStyle.BRACKET` (``[N]`` — §10.1's canonical form).
            Swap to ``PAREN`` / ``CURLY`` / ``CARET`` when you need the
            marker to not clash with downstream syntax (e.g. Markdown link
            syntax reserves ``[`` / ``]``). The digit-enum structural guarantee
            is identical across styles.

    Returns:
        A `Grammar` with the rendered GBNF and the metadata backends need.

    Raises:
        ValueError: If `n_sources < 1`, or if `max_content_chars` is `< 1`
            (use `None` for unbounded).
        NotImplementedError: If `policy` is not one of the `Policy` enum values
            (e.g. a future variant that a user might have hand-cast).
    """
    spec = MARKER_SPECS[marker_style]

    if policy is Policy.REQUIRED:
        body = _required_body(spec.open_char, max_content_chars)
        effective_bound = max_content_chars
    elif policy is Policy.AUTO:
        body = _auto_body(spec.open_char)
        effective_bound = None
    elif policy is Policy.QUOTES_ONLY:
        body = _quotes_only_body(spec.open_char)
        effective_bound = None
    else:  # pragma: no cover — Policy is a closed enum.
        raise NotImplementedError(f"No grammar defined for policy {policy!r}")

    gbnf = f"{body}{_shared_tail()}{_cite_id_rule(n_sources, spec)}\n"
    return Grammar(
        gbnf=gbnf,
        cite_ids=tuple(range(1, n_sources + 1)),
        policy=policy,
        marker_style=marker_style,
        max_content_chars=effective_bound,
    )
