.PHONY: help install install-dev lint format test coverage clean build publish pre-commit-install

help:
	@echo "Available commands:"
	@echo "  make install           Install the package"
	@echo "  make install-dev       Install package with dev dependencies"
	@echo "  make lint              Run linting and type checking"
	@echo "  make format            Format code with black and isort"
	@echo "  make test              Run tests with coverage"
	@echo "  make coverage          Generate coverage report"
	@echo "  make clean             Remove build artifacts and cache"
	@echo "  make build             Build distribution packages"
	@echo "  make pre-commit-install Install pre-commit hooks"
	@echo "  make pre-commit-run    Run pre-commit on all files"

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

lint:
	flake8 my_ml_toolkit tests --max-line-length=100 --extend-ignore=E203,W503
	mypy my_ml_toolkit --ignore-missing-imports
	black --check --line-length=100 my_ml_toolkit tests
	isort --check-only --profile=black --line-length=100 my_ml_toolkit tests

format:
	isort --profile=black --line-length=100 my_ml_toolkit tests
	black --line-length=100 my_ml_toolkit tests

test:
	pytest tests/ -v --cov=my_ml_toolkit --cov-report=html --cov-report=term

coverage:
	coverage report -m
	@echo "HTML coverage report generated in htmlcov/index.html"

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist .pytest_cache .coverage htmlcov .mypy_cache .tox

build: clean
	python -m build

publish: build
	@echo "Review the build artifacts before publishing"
	ls -lh dist/
	@echo "Run: python -m twine upload dist/*"

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

.DEFAULT_GOAL := help
