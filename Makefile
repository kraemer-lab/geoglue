OS := $(shell uname -s)
CDO := $(shell which cdo)

tests: lint install install_cdo
	uv run pytest --cov --cov-report=html

install_cdo:
	./scripts/install_cdo.sh

install-uv:
	curl -LsSf https://astral.sh/uv/0.6.0/install.sh | sh

install:
	uv sync --all-extras --dev

lint:
	uvx ruff check src
	uvx ruff check tests

.PHONY: install-uv install lint tests docs
