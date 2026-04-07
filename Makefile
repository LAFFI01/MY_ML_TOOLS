.PHONY: help install install-dev lint format test coverage clean build publish pre-commit-install pre-commit-run

help:
	@cd my_ml_toolkit && make help

install:
	@cd my_ml_toolkit && make install

install-dev:
	@cd my_ml_toolkit && make install-dev

lint:
	@cd my_ml_toolkit && make lint

format:
	@cd my_ml_toolkit && make format

test:
	@cd my_ml_toolkit && make test

coverage:
	@cd my_ml_toolkit && make coverage

clean:
	@cd my_ml_toolkit && make clean

build:
	@cd my_ml_toolkit && make build

publish:
	@cd my_ml_toolkit && make publish

pre-commit-install:
	@cd my_ml_toolkit && make pre-commit-install

pre-commit-run:
	@cd my_ml_toolkit && make pre-commit-run

.DEFAULT_GOAL := help
