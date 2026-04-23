"""Citation grammar module — §10.1 contract.

Builds GBNF grammars that XGrammar / llguidance / llama.cpp (P2b+) mask against
at decode time, making fabricated citation markers a logit-level impossibility.

Public API:

- `Grammar`: the (gbnf, cite_ids, policy, root_rule) record.
- `build_grammar(n_sources, policy)`: build a grammar for the given source
  count and policy.
"""

from __future__ import annotations

from citeformer.grammar.builder import Grammar, build_grammar

__all__ = ["Grammar", "build_grammar"]
