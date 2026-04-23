# Development setup

```bash
git clone https://github.com/random-walks/citeformer
cd citeformer
make dev
```

`make dev` runs `uv sync --all-extras` and installs the pre-commit hooks.

## Common commands

```bash
make test             # full pytest suite
make test-unit        # fast unit tests only
make test-integration # slow tests — loads real models or hits live APIs
make lint             # ruff + mypy
make format           # apply ruff fixes
make docs             # live-reload Sphinx preview at http://127.0.0.1:5190
make docs-build       # one-shot docs build (CI mirror, with -W)
make release-check    # dry-run sdist + wheel build
```

## Environment variables

Copy `.env.example` → `.env` and fill in any you need. Most work flows don't need anything. `HF_TOKEN` is only required for gated models (e.g. Llama 3.2); the default benchmark model (`Qwen/Qwen2.5-0.5B-Instruct`) and the examples' default (`gpt2`) are both open-access. `CITEFORMER_CROSSREF_MAILTO` is optional politeness — set it to your email to route Crossref / arXiv requests through the polite pool.

## Python version

citeformer supports Python 3.11 → 3.14. The local dev default is 3.12 (match `.pre-commit-config.yaml`). CI tests the full matrix.
