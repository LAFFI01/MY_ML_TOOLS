"""Tests for evaluator module."""

import pytest


def test_import_evaluator():
    """Test that evaluator module can be imported."""
    try:
        from my_ml_toolkit import evaluator  # noqa: F401

        assert True
    except ImportError:
        pytest.fail("Failed to import evaluator module")


def test_evaluate_and_plot_models_exists():
    """Test that evaluate_and_plot_models function exists."""
    from my_ml_toolkit import evaluate_and_plot_models

    assert callable(evaluate_and_plot_models)
