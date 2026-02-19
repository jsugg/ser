.PHONY: help setup fmt lint type test check ci train predict prepush clean

.DEFAULT_GOAL := help

FILE ?= $(if $(wildcard sample.wav),sample.wav,$(error sample.wav not found; run `make predict FILE=path/to.wav`))

help:
	@echo "Targets:"
	@echo "  setup    - install dev deps"
	@echo "  fmt      - format code"
	@echo "  lint     - run linters"
	@echo "  type     - run type checks"
	@echo "  test     - run tests"
	@echo "  check    - lint + type + test"
	@echo "  prepush  - run pre-commit pre-push hooks"
	@echo "  train    - train model"
	@echo "  predict  - run prediction (set FILE=sample.wav)"
	@echo "  clean    - remove caches"

setup:
	uv sync --extra dev

fmt:
	uv run ruff check --fix ser tests
	uv run ruff format ser tests
	uv run isort ser tests

lint:
	uv run ruff check ser tests
	uv run black --check ser tests
	uv run isort --check-only ser tests

type:
	uv run mypy ser tests
	uv run pyright ser tests

test:
	uv run pytest -q

check: lint type test

prepush:
	uvx pre-commit run --all-files --hook-stage pre-push

train:
	uv run ser --train

predict:
	uv run ser --file $(FILE)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build .pkg-smoke
