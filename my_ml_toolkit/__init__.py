"""
My ML Toolkit - Enterprise-grade machine learning evaluation pipeline
"""

__version__ = "0.1.0"
__author__ = "LAFFI01"
__license__ = "MIT"

from .evaluator import evaluate_and_plot_models

__all__ = [
    "evaluate_and_plot_models",
    "__version__",
    "__author__",
    "__license__",
]
