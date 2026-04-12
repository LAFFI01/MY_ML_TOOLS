"""
My ML Toolkit - Enterprise-grade machine learning evaluation pipeline
"""

__version__ = "0.1.0"
__author__ = "LAFFI01"
__license__ = "MIT"

from .evaluator import (
    evaluate_and_plot_models,
    balanced_multiclass_accuracy,
    get_balanced_accuracy_scorer,
)

__all__ = [
    "evaluate_and_plot_models",
    "balanced_multiclass_accuracy",
    "get_balanced_accuracy_scorer",
    "__version__",
    "__author__",
    "__license__",
]
