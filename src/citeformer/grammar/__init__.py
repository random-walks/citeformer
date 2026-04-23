"""Citation grammar module — §10.1 contract.

Builds EBNF grammars that XGrammar / llguidance (P2b) mask against at decode
time, making fabricated citation markers a logit-level impossibility.

Public API:

- `Grammar`: the (ebnf, cite_ids, policy) triple.
- `build_grammar(n_sources, policy)`: build a grammar for the given source count
  and policy.
- `parse_ok(grammar, text)`: lark-based round-trip check, useful for debugging
  and for the post-hoc verification path.
"""

from __future__ import annotations

from citeformer.grammar.builder import Grammar, build_grammar, parse_ok

__all__ = ["Grammar", "build_grammar", "parse_ok"]
