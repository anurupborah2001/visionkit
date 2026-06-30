.DEFAULT_GOAL := help
PYTHON        := python3.11
UV            := uv
VENV          := .venv
PRE_COMMIT    := $(UV) run pre-commit

# ── Help ───────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Bootstrap ─────────────────────────────────────────────────────────────────
.PHONY: install
install: ## Install all deps (prod + dev) via uv
	$(UV) sync --all-groups

.PHONY: install-dev-tools
install-dev-tools: ## Add linting / formatting dev tools to the project
	$(UV) add --dev \
	  pre-commit \
	  black \
	  isort \
	  ruff \
	  flake8 \
	  flake8-bugbear \
	  flake8-comprehensions \
	  flake8-simplify \
	  pyupgrade \
	  pytest \
	  pytest-cov

.PHONY: setup
setup: install install-dev-tools hooks ## Full first-time setup (deps + hooks)

# ── Pre-commit hooks ──────────────────────────────────────────────────────────
.PHONY: hooks
hooks: ## Install pre-commit hooks into .git/hooks
	$(PRE_COMMIT) install --install-hooks
	$(PRE_COMMIT) install --hook-type commit-msg
	$(PRE_COMMIT) install --hook-type pre-push

.PHONY: hooks-update
hooks-update: ## Update all hook revisions to latest
	$(PRE_COMMIT) autoupdate

.PHONY: hooks-run
hooks-run: ## Run all pre-commit hooks against every tracked file
	$(PRE_COMMIT) run --all-files

.PHONY: hooks-uninstall
hooks-uninstall: ## Remove pre-commit hooks from .git/hooks
	$(PRE_COMMIT) uninstall
	$(PRE_COMMIT) uninstall --hook-type commit-msg
	$(PRE_COMMIT) uninstall --hook-type pre-push

# ── Formatting ────────────────────────────────────────────────────────────────
.PHONY: format
format: ## Auto-format with Black + isort
	$(UV) run black openvisionkit/ tests/ main.py
	$(UV) run isort --profile black openvisionkit/ tests/ main.py

.PHONY: format-check
format-check: ## Check formatting without modifying files
	$(UV) run black --check openvisionkit/ tests/ main.py
	$(UV) run isort --profile black --check-only openvisionkit/ tests/ main.py

# ── Linting ───────────────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run Ruff + Flake8
	$(UV) run ruff check openvisionkit/ tests/ main.py
	$(UV) run flake8 --max-line-length=88 --extend-ignore=E203,W503,E501 openvisionkit/ tests/ main.py

.PHONY: lint-fix
lint-fix: ## Run Ruff with auto-fix
	$(UV) run ruff check --fix openvisionkit/ tests/ main.py

# ── Type checking ─────────────────────────────────────────────────────────────
.PHONY: typecheck
typecheck: ## Run mypy on the package
	$(UV) run python -m mypy openvisionkit/ --ignore-missing-imports

# ── Tests ─────────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run full test suite
	$(UV) run pytest tests/ -v

.PHONY: test-smoke
test-smoke: ## Fast smoke check (stop on first failure)
	$(UV) run pytest tests/ -x -q --tb=short

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(UV) run pytest tests/ --cov=openvisionkit --cov-report=term-missing --cov-report=html

# ── Combined quality gate ─────────────────────────────────────────────────────
.PHONY: check
check: format-check lint typecheck test-smoke ## Run all checks (CI equivalent)

# ── Cleanup ───────────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove build artefacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov       -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
