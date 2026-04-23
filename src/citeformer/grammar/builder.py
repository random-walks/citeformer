"""Citation-grammar construction — §10.1 contract implementation.

Emits GBNF grammars consumable by XGrammar / llguidance / llama.cpp. The point
of this module is the `cite-id` rule:

    cite-id ::= "[" ("1" | "2" | ... | "N") "]"

where `N` is dynamically set to `len(sources)` per generate() call. That's what
makes a fabricated citation (`[N+k]` for any `k > 0`) a logit-level impossibility
when the downstream backend masks against this grammar.

Three policies sit on top:

- `REQUIRED`: every sentence must end `content cite-group sent-end`. The model
  can't close a sentence without citing.
- `QUOTES_ONLY`: only quoted spans require a trailing `cite-group`. Narrative
  prose can stand alone.
- `AUTO`: `cite-group` is allowed anywhere but not required. The P6 `verify()`
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
    """

    gbnf: str
    cite_ids: tuple[int, ...]
    policy: Policy
    root_rule: str = "root"


# Shared tail: cite-group + ws live here so the three policy bodies don't
# redefine them. The cite-id rule is appended per-call with a dynamic enum
# reflecting len(sources).
_SHARED_TAIL = """\
cite-group ::= cite-id (ws cite-id)*
ws ::= " "
"""


# Per-policy grammar bodies. Each defines `root` and any auxiliary rules it
# needs. `cite-group`, `ws`, and `cite-id` come from the shared tail + rule.

_REQUIRED_BODY = """\
root ::= sentence (ws sentence)*
sentence ::= content cite-group sent-end
content ::= [^\\[.!?]+
sent-end ::= "." | "!" | "?"
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


def build_grammar(n_sources: int, policy: Policy) -> Grammar:
    """Build the citation-constraining GBNF grammar for a generation call.

    Args:
        n_sources: Number of sources in scope. Must be >= 1. Determines the
            set of valid cite ids (1..n_sources inclusive).
        policy: Citation enforcement policy.

    Returns:
        A `Grammar` with the rendered GBNF and the metadata backends need.

    Raises:
        ValueError: If `n_sources < 1`.
        NotImplementedError: If `policy` is not one of the `Policy` enum values
            (e.g. a future variant that a user might have hand-cast).
    """
    if policy is Policy.REQUIRED:
        body = _REQUIRED_BODY
    elif policy is Policy.AUTO:
        body = _AUTO_BODY
    elif policy is Policy.QUOTES_ONLY:
        body = _QUOTES_ONLY_BODY
    else:  # pragma: no cover — Policy is a closed enum.
        raise NotImplementedError(f"No grammar defined for policy {policy!r}")

    gbnf = f"{body}{_SHARED_TAIL}{_cite_id_rule(n_sources)}\n"
    return Grammar(
        gbnf=gbnf,
        cite_ids=tuple(range(1, n_sources + 1)),
        policy=policy,
    )
