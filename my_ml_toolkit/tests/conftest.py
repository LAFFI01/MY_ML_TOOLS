"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_features():
    """Generate sample feature data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        }
    )


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 100))


@pytest.fixture
def sample_regression_labels():
    """Generate sample regression labels for testing."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100))
