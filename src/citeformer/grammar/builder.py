"""Citation-grammar construction — §10.1 contract implementation.

The whole point of this module is the `CITE_ID` terminal:

    CITE_ID: "[" ("1" | "2" | ... | "N") "]"

where `N` is dynamically set to `len(sources)` per generate() call. That's what
makes a fabricated citation (`[N+k]` for any `k > 0`) a logit-level impossibility
when the downstream backend masks against this grammar.

Three policies sit on top:

- `REQUIRED`: every sentence must end `... cite_group SENT_END`. The model
  can't close a sentence without citing.
- `QUOTES_ONLY`: only quoted spans require a trailing `cite_group`. Narrative
  prose can stand alone.
- `AUTO`: `cite_group` is allowed anywhere but not required. The P6 `verify()`
  coverage check surfaces missing citations post-hoc instead.

The emitted grammar string is [Lark](https://lark-parser.readthedocs.io) format,
which both XGrammar and llguidance accept (via conversion) in P2b.
"""

from __future__ import annotations

from dataclasses import dataclass

from citeformer.core import Policy


@dataclass(frozen=True)
class Grammar:
    """A citation-constraining EBNF grammar for one generation call.

    Attributes:
        ebnf: Full Lark-format grammar string. Ready to hand to a backend that
            converts it to its native constrained-decoding format (XGrammar
            object, llguidance constraint, llama.cpp GBNF string).
        cite_ids: 1-indexed source ids that the grammar admits, in ascending
            order. Derived from `len(sources)` at build time.
        policy: Enforcement policy that shaped the grammar body.
    """

    ebnf: str
    cite_ids: tuple[int, ...]
    policy: Policy


# Shared tail: cite_group + WS live here so the three policy bodies don't
# redefine them. The CITE_ID terminal is appended per-call with a dynamic
# enum reflecting len(sources).
_SHARED_TAIL = """
cite_group: CITE_ID (WS CITE_ID)*
WS: " "
""".rstrip()


# Per-policy grammar bodies. Each defines `start` and any auxiliary rules it
# needs. `cite_group`, `WS`, and `CITE_ID` come from the shared tail + terminal.

_REQUIRED_BODY = """\
start: sentence (WS sentence)*
sentence: CONTENT cite_group SENT_END
CONTENT: /[^\\[\\.!?]+/
SENT_END: "." | "!" | "?"
"""

_AUTO_BODY = """\
start: (TEXT | cite_group)+
TEXT: /[^\\[]+/
"""

_QUOTES_ONLY_BODY = """\
start: (TEXT | quoted_cite)+
TEXT: /[^\\[\"]+/
quoted_cite: QUOTE cite_group
QUOTE: "\\"" /[^\"]*/ "\\""
"""


def _cite_id_terminal(n_sources: int) -> str:
    """Render the `CITE_ID` terminal for `n_sources` sources.

    For ``n_sources=3`` returns::

        CITE_ID: "[" ("1" | "2" | "3") "]"

    which is the §10.1 contract's load-bearing terminal. `<digits>` in the
    contract is the parenthesized enum here — the brackets are literals.
    """
    if n_sources < 1:
        raise ValueError(f"n_sources must be >= 1, got {n_sources}")
    alternatives = " | ".join(f'"{i}"' for i in range(1, n_sources + 1))
    return f'CITE_ID: "[" ({alternatives}) "]"'


def build_grammar(n_sources: int, policy: Policy) -> Grammar:
    """Build the citation-constraining grammar for a generation call.

    Args:
        n_sources: Number of sources in scope. Must be >= 1. Determines the
            set of valid cite ids (1..n_sources inclusive).
        policy: Citation enforcement policy.

    Returns:
        A `Grammar` with the rendered EBNF and the metadata backends need.

    Raises:
        ValueError: If `n_sources < 1`.
        NotImplementedError: If `policy` is not one of the `Policy` enum values
            (e.g. a future variant that a user might have hand-cast). The
            exhaustive match on `Policy` catches the three known values.
    """
    if policy is Policy.REQUIRED:
        body = _REQUIRED_BODY
    elif policy is Policy.AUTO:
        body = _AUTO_BODY
    elif policy is Policy.QUOTES_ONLY:
        body = _QUOTES_ONLY_BODY
    else:  # pragma: no cover — Policy is a closed enum.
        raise NotImplementedError(f"No grammar defined for policy {policy!r}")

    ebnf = f"{body}\n{_SHARED_TAIL}\n{_cite_id_terminal(n_sources)}\n"
    return Grammar(
        ebnf=ebnf,
        cite_ids=tuple(range(1, n_sources + 1)),
        policy=policy,
    )


def parse_ok(grammar: Grammar, text: str) -> bool:
    """Return True iff `text` is a valid sentence under `grammar.ebnf`.

    Uses lark to parse. Imports lark lazily so that callers who never invoke
    `parse_ok` pay no import cost.

    This is primarily a debugging aid and a post-hoc check used by tests. In
    P2b the real guarantee is decode-time: the backend masks invalid tokens
    against the same EBNF, so the model physically cannot produce text that
    fails `parse_ok`.

    Args:
        grammar: The grammar to parse against.
        text: Candidate text.

    Returns:
        `True` iff the full `text` parses without error.
    """
    # Lazy import — we don't want to make lark a hard dep for every caller.
    from lark import Lark
    from lark.exceptions import LarkError

    try:
        parser = Lark(grammar.ebnf, start="start")
        parser.parse(text)
    except LarkError:
        return False
    return True
