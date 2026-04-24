.PHONY: help dev test test-unit test-integration coverage lint format docs docs-build release-check clean hf-space

help:
	@echo "citeformer developer commands:"
	@echo ""
	@echo "  make dev              Install all dev deps + pre-commit hooks"
	@echo "  make test             Run full pytest suite"
	@echo "  make test-unit        Run unit tests only (fast; no model loads)"
	@echo "  make test-integration Run integration tests (loads real HF models; slow)"
	@echo "  make coverage         Run unit tests with coverage; write htmlcov/ + coverage.json"
	@echo "  make lint             Run ruff + mypy"
	@echo "  make format           Apply ruff formatting + fixes"
	@echo "  make docs             Build + serve docs with live reload (http://127.0.0.1:5190)"
	@echo "  make docs-build       Build docs once (for CI)"
	@echo "  make release-check    Dry-run build + version print"
	@echo "  make hf-space SPACE=user/name  Sync hf-space/ to a HuggingFace Space"
	@echo "  make clean            Remove build artifacts and caches"

dev:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:
	uv sync --extra dev --extra hf --extra llamacpp
	uv run pytest -m integration

coverage:
	uv run pytest tests/unit \
	    --cov=citeformer \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    --cov-report=json:coverage.json
	@echo "Coverage report: htmlcov/index.html"

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

format:
	uv run ruff format .
	uv run ruff check --fix .

docs:
	uv run sphinx-autobuild -a --watch src docs docs/_build/html --port 5190

docs-build:
	uv run sphinx-build -W --keep-going -b html docs docs/_build/html

release-check:
	uv build
	uv run python -c "import citeformer; print('citeformer', citeformer.__version__)"

hf-space:
	@if [ -z "$(SPACE)" ]; then \
	    echo "usage: make hf-space SPACE=<hf-user>/<space-name>"; \
	    echo "(one-time prereq: 'pip install huggingface_hub && huggingface-cli login')"; \
	    exit 2; \
	fi
	./hf-space/deploy.sh "$(SPACE)"

clean:
	rm -rf dist/ build/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
	rm -rf docs/_build/ docs/apidocs/
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
