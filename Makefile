.PHONY: help setup setup-runtime fmt lint type type-completeness test test-cov check ci train predict optin-all-restricted quality-gate-full prepush prepush-check prepush-hook import-lint lock-check workflow-lint ci-contracts clean

.DEFAULT_GOAL := help

FILE ?= $(if $(wildcard sample.wav),sample.wav,$(error sample.wav not found; run `make predict FILE=path/to.wav`))

help:
	@echo "Targets:"
	@echo "  setup    - platform-aware setup (full runtime + dev tools)"
	@echo "  setup-runtime - platform-aware setup (runtime deps only)"
	@echo "  fmt      - format code"
	@echo "  lint     - run linters"
	@echo "  type     - run type checks"
	@echo "  type-completeness - enforce pyright verifytypes public API completeness"
	@echo "  test     - run tests"
	@echo "  test-cov - run tests with branch coverage gating"
	@echo "  check    - lint + type + test"
	@echo "  prepush  - run local pre-push quality gates (autofix + verify)"
	@echo "  prepush-check - run canonical pre-push hook command (check-only)"
	@echo "  prepush-hook - run the git pre-push hook workflow (autofix + abort if files change)"
	@echo "  import-lint - run public API boundary import-lint lane"
	@echo "  lock-check - verify uv.lock is fresh without mutating it"
	@echo "  workflow-lint - run actionlint over GitHub Actions workflows"
	@echo "  ci-contracts - run CI/CD policy contract tests"
	@echo "  train    - train model"
	@echo "  predict  - run prediction (set FILE=sample.wav)"
	@echo "  optin-all-restricted - persist consent for all known restricted backends"
	@echo "  quality-gate-full - run full-dataset quality gate suite"
	@echo "  clean    - remove caches"

setup:
	./scripts/setup_compatible_env.sh

setup-runtime:
	SER_SETUP_INCLUDE_DEV=false ./scripts/setup_compatible_env.sh

fmt:
	uv run --frozen --extra dev pyupgrade --py312-plus --exit-zero-even-if-changed $$(rg --files ser tests -g '*.py')
	uv run --frozen --extra dev ruff check --fix ser tests
	uv run --frozen --extra dev isort ser tests
	uv run --frozen --extra dev black ser tests

lint:
	uv run --frozen --extra dev ruff check --ignore TID251 ser tests
	bash ./scripts/run_import_lint.sh
	uv run --frozen --extra dev black --check ser tests
	uv run --frozen --extra dev isort --check-only ser tests

type:
	uv run --frozen --extra dev mypy ser tests
	uv run --frozen --extra dev pyright --pythonversion 3.12 ser tests

type-completeness:
	uv run --frozen --extra dev python scripts/check_type_completeness.py

test:
	uv run --frozen --extra dev pytest -q

test-cov:
	uv run --frozen --extra dev coverage erase
	uv run --frozen --extra dev coverage run -m pytest -q
	uv run --frozen --extra dev coverage combine
	uv run --frozen --extra dev coverage report
	uv run --frozen --extra dev coverage xml
	uv run --frozen --extra dev coverage html

check: lint type test

prepush-check:
	uv run --frozen --extra dev pre-commit run --all-files --hook-stage pre-push

prepush: fmt prepush-check

prepush-hook:
	bash ./scripts/run_prepush_gate.sh

import-lint:
	bash ./scripts/run_import_lint.sh

lock-check:
	uv lock --check

workflow-lint:
	uvx --from actionlint-py==1.7.12.24 actionlint

ci-contracts:
	uv run --frozen --extra dev pytest -q \
		tests/suites/integration/architecture/test_ci_workflow_contracts.py \
		tests/suites/integration/architecture/test_ci_change_classifier.py

train:
	uv run ser --train

predict:
	uv run ser --file $(FILE)

optin-all-restricted:
	uv run ser --accept-all-restricted-backends

quality-gate-full:
	./scripts/run_full_dataset_quality_gate.sh

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build .pkg-smoke htmlcov reports release-evidence
	find . -maxdepth 1 -type f \( -name ".coverage" -o -name ".coverage.*" -o -name "coverage.xml" \) -delete
