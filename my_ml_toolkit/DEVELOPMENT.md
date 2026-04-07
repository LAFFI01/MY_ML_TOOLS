# Development Guide

Welcome to the development environment for my_ml_toolkit! This guide will help you get started.

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .[dev]
```

### 2. Install Pre-commit Hooks

```bash
make pre-commit-install
```

This ensures code quality checks run automatically before each commit.

### 3. Common Development Tasks

```bash
# Run all checks
make lint

# Format code
make format

# Run tests
make test

# Run tests with coverage
make coverage

# Clean build artifacts
make clean

# View all available commands
make help
```

## Project Structure

```
my_ml_toolkit/
├── my_ml_toolkit/          # Main package
│   ├── __init__.py
│   ├── evaluator.py        # Core evaluation logic
│   └── ...
├── tests/                  # Test suite
│   ├── conftest.py         # Pytest configuration
│   ├── test_evaluator.py   # Evaluator tests
│   └── ...
├── .github/
│   └── workflows/          # CI/CD workflows
├── Makefile                # Development commands
├── pyproject.toml          # Project metadata and dependencies
├── tox.ini                 # Multi-version testing configuration
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── .editorconfig           # Editor settings
├── .gitignore              # Git ignore rules
├── .env.example            # Environment variables template
└── CONTRIBUTING.md         # Contribution guidelines
```

## Tools Used

### Code Quality
- **black** - Code formatting (100 char lines)
- **isort** - Import organization
- **flake8** - Linting
- **mypy** - Static type checking

### Testing
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **tox** - Multi-version testing

### Git
- **pre-commit** - Automated checks before commits

## Code Style Guidelines

### Line Length
Maximum 100 characters (enforced by black)

### Type Hints
Use type hints where practical:

```python
from typing import List, Dict
import pandas as pd

def process_data(
    data: pd.DataFrame,
    columns: List[str]
) -> Dict[str, float]:
    """Process data and return statistics."""
    return {col: data[col].mean() for col in columns}
```

### Docstrings
Use Google-style docstrings:

```python
def train_model(X_train, y_train, model_type='linear'):
    """Train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train
        
    Returns:
        Trained model object
        
    Raises:
        ValueError: If model_type is not supported
    """
    pass
```

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_evaluator.py

# Run with coverage
pytest --cov=my_ml_toolkit tests/

# Run with verbose output
pytest -v
```

### Writing Tests
- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Use pytest fixtures from `conftest.py`

Example:
```python
def test_evaluator_with_valid_data(sample_features, sample_labels):
    """Test evaluator with valid input data."""
    from my_ml_toolkit.evaluator import Evaluator
    
    evaluator = Evaluator()
    result = evaluator.evaluate(sample_features, sample_labels)
    assert result is not None
```

## Continuous Integration

GitHub Actions runs automatically on:
- Commits to `main` and `develop` branches
- Pull requests

Workflows check:
- Tests on Python 3.8-3.12
- Code quality (lint, type checking, formatting)
- Multiple operating systems

## Making Changes

1. **Create a branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and test:**
   ```bash
   make format
   make lint
   make test
   ```

3. **Commit:**
   ```bash
   git add .
   git commit -m "Add my feature"
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/my-feature
   ```

## Troubleshooting

### Pre-commit hooks not running
```bash
make pre-commit-install
```

### Import errors when running tests
```bash
pip install -e .[dev]
```

### Type checking errors
```bash
# Check what modules are available
mypy --version
mypy my_ml_toolkit --show-error-codes
```

### Virtual environment issues
```bash
# Remove and recreate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Getting Help

- Check [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Review existing tests for examples
- Check issue tracker for known issues
- Open a new issue if you find a bug

---

Happy coding! 🚀
