"""
Example: How to use fold-level tracking with the toolkit
(Similar to your manual stratified k-fold code)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from my_ml_toolkit import evaluate_and_plot_models, balanced_multiclass_accuracy

# ============================================================================
# 1. TOOLKIT APPROACH (Recommended - One function call!)
# ============================================================================

print("=" * 80)
print("TOOLKIT APPROACH: Automatic fold tracking")
print("=" * 80)

# Load your data (or use train/test combined)
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, 
    weights=[0.7, 0.3], random_state=42
)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="target")

models = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
}

preprocess_pipeline = Pipeline([("scaler", StandardScaler())])

# ✨ Call with show_fold_details=True to see per-fold metrics
results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=preprocess_pipeline,
    X=X,
    y=y,
    task_type="classification",
    primary_metric="balanced_accuracy",
    cv=5,  # 5-fold cross-validation
    show_fold_details=True,  # ← Shows fold scores for each model
    plot_comparison=False,
    plot_diagnostics=False,
    plot_lc=False,
    plot_importance=False,
    verbose=True,
)

# ============================================================================
# 2. ACCESS FOLD DETAILS PROGRAMMATICALLY
# ============================================================================

print("\n" + "=" * 80)
print("ACCESSING FOLD DATA PROGRAMMATICALLY")
print("=" * 80)

# Get raw CV scores (array of scores per fold)
raw_cv_scores = results['raw_cv_scores']

for model_name, fold_scores in raw_cv_scores.items():
    print(f"\n{model_name}:")
    print(f"  Individual fold scores: {fold_scores}")
    print(f"  Mean:                   {fold_scores.mean():.4f}")
    print(f"  Std:                    {fold_scores.std():.4f}")
    print(f"  Range:                  {fold_scores.max() - fold_scores.min():.4f}")

# ============================================================================
# 3. COMPARISON: Manual vs Toolkit
# ============================================================================

print("\n" + "=" * 80)
print("MANUAL APPROACH vs TOOLKIT APPROACH")
print("=" * 80)

print("""
YOUR MANUAL CODE:
  - Create StratifiedKFold manually
  - Loop through folds
  - Track OOF predictions for each fold
  - Calculate per-fold accuracy
  - Print fold details manually
  × 50+ lines of code needed

TOOLKIT APPROACH:
  - Call evaluate_and_plot_models()
  - Set show_fold_details=True
  - Get fold details automatically
  - Summary table includes CV Min/Max/Range
  ✓ 1 function call + access results['raw_cv_scores']

KEY BENEFITS:
  1. Automatic cross-validation
  2. Support for custom metrics (balanced_accuracy)
  3. Integrated fold stability analysis
  4. Per-fold metrics in summary table
  5. No data leakage guaranteed (imblearn.pipeline)
""")

print("\n✅ Fold-level accuracy tracking is now part of the toolkit!")
