# Contributing to my_ml_toolkit

Thank you for your interest in contributing to my_ml_toolkit! Here's how to get started.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LAFFI01/MY_ML.git
   cd my_ml_toolkit
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e .[dev]
   ```

4. **Install pre-commit hooks:**
   ```bash
   make pre-commit-install
   ```

## Development Workflow

### Code Quality

We use the following tools to maintain code quality:

- **black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
- **mypy** - Type checking
- **pytest** - Testing

Run all checks with:
```bash
make lint
```

Format your code with:
```bash
make format
```

### Running Tests

```bash
make test
```

This runs tests with coverage reporting.

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks before each commit. They will prevent commits that don't meet our standards. To run them manually:

```bash
make pre-commit-run
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clear, descriptive commit messages
   - Add tests for new functionality
   - Update documentation as needed

3. **Run all checks:**
   ```bash
   make lint
   make test
   ```

4. **Push and create a Pull Request:**
   - Provide a clear description of your changes
   - Reference any related issues
   - Ensure all CI checks pass

## Code Style Guidelines

- **Line length:** 100 characters (enforced by black)
- **Python version:** 3.8+
- **Type hints:** Use type hints where practical
- **Docstrings:** Use Google-style docstrings
- **Testing:** Aim for >80% coverage

### Example function:

```python
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric: str = "accuracy"
) -> float:
    """Evaluate model performance on test data.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test labels
        metric: Metric to use for evaluation
        
    Returns:
        Evaluation metric score
    """
    predictions = model.predict(X_test)
    return compute_metric(y_test, predictions, metric)
```

## Reporting Issues

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Python version and environment
- Relevant code snippets

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or reach out to the maintainers!
