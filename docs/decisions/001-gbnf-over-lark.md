# ADR-001 — Grammar builder emits GBNF, not Lark

- **Status**: Accepted (P2b, 2026-04-23).
- **Supersedes**: the initial Lark-emitting `build_grammar` shipped in P2a.

## Context

The §10.1 citation marker contract demands a single grammar-representation format that all our target backends consume. P2a shipped a Lark-format emitter (`rule: production`) on the intuition that Lark is a well-known Python grammar library and the two formats "look similar." In P2b (HFBackend integration) we discovered xgrammar — our default backend — rejects Lark's `:` operator with an EBNF parse error; it expects GBNF (`rule ::= production`), the same format llama.cpp uses natively.

Options considered:

- **Keep Lark, write a Lark→GBNF translator**. Two grammar formats to maintain in lockstep; translation quirks for regex terminals.
- **Keep Lark, use a Lark parser wrapped around xgrammar**. Adds indirection and a third grammar dependency.
- **Switch to GBNF directly**. One format, zero translation, wire-compatible with both xgrammar and llama.cpp.

## Decision

`citeformer.grammar.builder.build_grammar` emits GBNF. `Grammar.gbnf` (renamed from `.ebnf`) carries the string; rule names use kebab-case per GBNF convention (`cite-id`, `cite-group`, `sent-end`); entry rule is `root` (not `start`) since that's xgrammar's default.

The `parse_ok` helper (Lark-based post-hoc check) was removed. Semantic validation now happens at integration time — `test_hf_backend_grammar_compiles` and `test_llamacpp_grammar_compiles_against_llama_cpp_parser` compile the emitted string with the authoritative parsers for each backend.

## Consequences

- One grammar DSL, three backends, no translator layer. Adding a new local backend (e.g. TensorRT-LLM, MLX) only needs the backend-specific wiring — the grammar string is ready.
- `lark` stays a main dependency because it remains a useful sanity parser for user-facing tooling (e.g. future "does this text satisfy my policy?" helpers). If we never build those, we can drop it.
- Our §10.1 snapshot tests moved from Lark syntax to GBNF syntax in one bulk update, which is documented as a §10.1 DSL-swap (minor-level, not breaking — marker semantics unchanged).
