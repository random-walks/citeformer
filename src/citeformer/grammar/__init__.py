"""Citation grammar module — §10.1 contract.

Builds GBNF grammars that XGrammar / llguidance / llama.cpp (P2b+) mask against
at decode time, making fabricated citation markers a logit-level impossibility.

Public API:

- `Grammar`: the (gbnf, cite_ids, policy, root_rule, max_content_chars) record.
- `build_grammar(n_sources, policy, *, max_content_chars=...)`: build a grammar
  for the given source count, policy, and REQUIRED-progression bound.
- `DEFAULT_MAX_CONTENT_CHARS`: default soft bound on REQUIRED-policy content
  repetition — see ADR-009.
"""

from __future__ import annotations

from citeformer.grammar.builder import (
    DEFAULT_MAX_CONTENT_CHARS,
    MARKER_SPECS,
    Grammar,
    MarkerSpec,
    build_grammar,
)

__all__ = [
    "DEFAULT_MAX_CONTENT_CHARS",
    "MARKER_SPECS",
    "Grammar",
    "MarkerSpec",
    "build_grammar",
]
