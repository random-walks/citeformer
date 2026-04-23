"""Citation-grammar construction — §10.1 contract implementation.

Emits GBNF grammars consumable by XGrammar / llguidance / llama.cpp. The point
of this module is the `cite-id` rule:

    cite-id ::= "[" ("1" | "2" | ... | "N") "]"

where `N` is dynamically set to `len(sources)` per generate() call. That's what
makes a fabricated citation (`[N+k]` for any `k > 0`) a logit-level impossibility
when the downstream backend masks against this grammar.

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

from citeformer.core import Policy

#: Default soft progression bound on the REQUIRED-policy `content` rule, in
#: characters. Once a sentence has accumulated this many non-terminating chars,
#: xgrammar masks everything except a citation bracket, forcing the model to
#: progress instead of stalling in content state. See ADR-009.
DEFAULT_MAX_CONTENT_CHARS = 240


@dataclass(frozen=True)
class Grammar:
    """A citation-constraining GBNF grammar for one generation call.

    Attributes:
        gbnf: Full GBNF grammar string. Accepted by XGrammar's
            `compile_grammar()` and by llama.cpp's native GBNF support.
        cite_ids: 1-indexed source ids that the grammar admits, in ascending
            order. Derived from `len(sources)` at build time.
        policy: Enforcement policy that shaped the grammar body.
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
    root_rule: str = "root"
    max_content_chars: int | None = None


# Shared tail: cite-group + ws live here so the three policy bodies don't
# redefine them. The cite-id rule is appended per-call with a dynamic enum
# reflecting len(sources).
_SHARED_TAIL = """\
cite-group ::= cite-id (ws cite-id)*
ws ::= " "
"""


_AUTO_BODY = """\
root ::= (text | cite-group)+
text ::= [^\\[]+
"""

_QUOTES_ONLY_BODY = """\
root ::= (text | quoted-cite)+
text ::= [^\\[\"]+
quoted-cite ::= quote cite-group
quote ::= "\\"" [^\"]* "\\""
"""


def _required_body(max_content_chars: int | None) -> str:
    """Render the REQUIRED-policy grammar body with an optional content bound.

    When `max_content_chars` is `None` the body emits the legacy unbounded
    ``content ::= [^\\[.!?]+`` rule. When a positive integer, it emits a
    bounded repetition ``[^\\[.!?]{1, N}`` — xgrammar and llama.cpp both
    accept this syntax as of their 2026 releases.
    """
    if max_content_chars is None:
        content_rule = "content ::= [^\\[.!?]+"
    else:
        if max_content_chars < 1:
            raise ValueError(
                f"max_content_chars must be >= 1 or None, got {max_content_chars}"
            )
        content_rule = f"content ::= [^\\[.!?]{{1, {max_content_chars}}}"
    return (
        "root ::= sentence (ws sentence)*\n"
        "sentence ::= content cite-group sent-end\n"
        f"{content_rule}\n"
        'sent-end ::= "." | "!" | "?"\n'
    )


def _cite_id_rule(n_sources: int) -> str:
    """Render the `cite-id` rule for `n_sources` sources.

    For ``n_sources=3`` returns::

        cite-id ::= "[" ("1" | "2" | "3") "]"

    which is the §10.1 contract's load-bearing rule. The parenthesized enum is
    the `<digits>` meta-variable from the contract description; the brackets
    are literal.
    """
    if n_sources < 1:
        raise ValueError(f"n_sources must be >= 1, got {n_sources}")
    alternatives = " | ".join(f'"{i}"' for i in range(1, n_sources + 1))
    return f'cite-id ::= "[" ({alternatives}) "]"'


def build_grammar(
    n_sources: int,
    policy: Policy,
    *,
    max_content_chars: int | None = DEFAULT_MAX_CONTENT_CHARS,
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

    Returns:
        A `Grammar` with the rendered GBNF and the metadata backends need.

    Raises:
        ValueError: If `n_sources < 1`, or if `max_content_chars` is `< 1`
            (use `None` for unbounded).
        NotImplementedError: If `policy` is not one of the `Policy` enum values
            (e.g. a future variant that a user might have hand-cast).
    """
    if policy is Policy.REQUIRED:
        body = _required_body(max_content_chars)
        effective_bound = max_content_chars
    elif policy is Policy.AUTO:
        body = _AUTO_BODY
        effective_bound = None
    elif policy is Policy.QUOTES_ONLY:
        body = _QUOTES_ONLY_BODY
        effective_bound = None
    else:  # pragma: no cover — Policy is a closed enum.
        raise NotImplementedError(f"No grammar defined for policy {policy!r}")

    gbnf = f"{body}{_SHARED_TAIL}{_cite_id_rule(n_sources)}\n"
    return Grammar(
        gbnf=gbnf,
        cite_ids=tuple(range(1, n_sources + 1)),
        policy=policy,
        max_content_chars=effective_bound,
    )
