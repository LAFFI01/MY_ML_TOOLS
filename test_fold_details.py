"""
Test script demonstrating fold-level accuracy details.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from my_ml_toolkit import evaluate_and_plot_models

# Generate sample imbalanced classification dataset
print("📊 Generating sample imbalanced dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.7, 0.3],  # Imbalanced: 70/30 split
    random_state=42,
)

X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="target")

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
}

# Preprocessing
preprocess_pipeline = Pipeline([("scaler", StandardScaler())])

# Run evaluation WITH fold details
print("\n" + "=" * 80)
print("🚀 Running evaluation WITH fold details (show_fold_details=True)")
print("=" * 80)

results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=preprocess_pipeline,
    X=X,
    y=y,
    task_type="classification",
    primary_metric="balanced_accuracy",  # Use balanced accuracy for imbalance
    cv=5,
    show_fold_details=True,  # ← NEW: Show per-fold metrics
    plot_comparison=False,
    plot_diagnostics=False,
    plot_lc=False,
    plot_importance=False,
    verbose=True,
)

# Display summary with fold statistics
print("\n" + "=" * 80)
print("📊 MODEL SUMMARY WITH FOLD STATISTICS")
print("=" * 80)
print(results["summary_df"].to_string())

print("\n✅ Test completed successfully!")
print("\nKey fold statistics in summary:")
print("  • CV Mean: Average cross-validation score across all folds")
print("  • CV Min/Max: Range of fold scores")
print("  • CV Range: Difference between best and worst folds (fold stability indicator)")
