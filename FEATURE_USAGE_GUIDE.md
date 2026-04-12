# 🚀 New Features Usage Guide

Comprehensive guide for using the latest ML Toolkit enhancements.

---

## 📋 Table of Contents
1. [Fold-Level Accuracy Tracking](#fold-level-accuracy-tracking)
2. [HalvingGridSearchCV](#halvinggridseachcv)
3. [Classification Metrics](#classification-metrics)
4. [Regression Metrics](#regression-metrics)
5. [Checkpoint & Resume](#checkpoint--resume)
6. [Complete Examples](#complete-examples)

---

## Fold-Level Accuracy Tracking

### Overview
Monitor per-fold cross-validation metrics to analyze model stability and identify problematic folds.

### Why Track Folds?
- **Detect Overfitting**: Check if one fold performs much worse than others
- **Fold Stability**: Identify inconsistent model performance across folds
- **Outlier Folds**: Find splits that may contain unusual data patterns
- **Fold Range**: Lower range = more stable model

### Usage

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from my_ml_toolkit import evaluate_and_plot_models
import pandas as pd

# Your data
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv').squeeze()

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

preprocess_pipeline = Pipeline([("scaler", StandardScaler())])

results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=preprocess_pipeline,
    X=X,
    y=y,
    task_type="classification",
    cv=5,
    show_fold_details=True,  # ← Display per-fold scores
    verbose=True
)
```

### Output Example
```
  📊 Fold-Level balanced_accuracy Scores for [Random Forest]:
  Fold 1: 0.845238
  Fold 2: 0.818452
  Fold 3: 0.775298  ← Worst fold
  Fold 4: 0.779762
  Fold 5: 0.857143  ← Best fold
  
  Mean:   0.815179
  Std:    0.033227  ← Standard deviation
  Min:    0.775298
  Max:    0.857143
```

### Programmatic Access
```python
# Get raw CV scores for all models
raw_cv_scores = results['raw_cv_scores']

for model_name, fold_scores in raw_cv_scores.items():
    print(f"{model_name}:")
    print(f"  Fold scores: {fold_scores}")
    print(f"  Mean: {fold_scores.mean():.4f}")
    print(f"  Std:  {fold_scores.std():.4f}")
    print(f"  Range: {fold_scores.max() - fold_scores.min():.4f}")
```

### Summary Table Includes
- `CV Mean (±Std)`: Average and standard deviation across folds
- `CV Min`: Best fold performance
- `CV Max`: Worst fold performance
- `CV Range`: Consistency indicator (lower = more stable)

---

## HalvingGridSearchCV

### Overview
Three hyperparameter search methods available:
- **"grid"**: Exhaustive GridSearchCV (traditional)
- **"random"**: RandomizedSearchCV (random sampling)
- **"halving"**: HalvingGridSearchCV (efficient successive halving)

### When to Use Each

| Method | Use Case | Speed | Coverage |
|--------|----------|-------|----------|
| **grid** | Small parameter space (<50 combinations) | Slow | 100% |
| **random** | Medium parameter space (50-500 combinations) | Medium | Partial |
| **halving** | Large parameter space (>500 combinations) | Fast | Good |

### Basic Usage

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from my_ml_toolkit import evaluate_and_plot_models
import pandas as pd
import numpy as np

# Sample data
X = pd.DataFrame(np.random.randn(200, 10), columns=[f"feature_{i}" for i in range(10)])
y = pd.Series(np.random.randint(0, 2, 200))

models = {
    "RandomForest": RandomForestClassifier(random_state=42)
}

# Halving search - efficient for large parameter spaces
param_grids = {
    "RandomForest": {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
    }
}

results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=None,
    X=X,
    y=y,
    param_grids=param_grids,
    search_type="halving",      # Use HalvingGridSearchCV
    cv=5,
    task_type="classification",
    verbose=True
)
```

---

## Classification Metrics

### Available Metrics

**Core Scalar Metrics:**
- Accuracy
- Precision (macro-averaged for multi-class)
- Recall (macro-averaged for multi-class)
- F1-Score (macro-averaged for multi-class)

**Threshold-Agnostic Metrics:**
- ROC Curve (with AUC)
- Precision-Recall Curve (with AUC)

**Probabilistic Metrics:**
- Log-Loss (cross-entropy)

**Specialized Metrics:**
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa

### Example: Binary Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from my_ml_toolkit import evaluate_and_plot_models

models = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=None,
    X=X_train,
    y=y_train,
    task_type="classification",
    test_size=0.2,
    plot_diagnostics=True,      # Enable confusion matrices
    plot_importance=False,       # No feature importance for classifiers
    verbose=True
)

# Output includes:
# - Core Scalar Metrics (Accuracy, Precision, Recall, F1)
# - Specialized Metrics (MCC, Cohen's Kappa)
# - Log-Loss
# - ROC-AUC
# - ROC & Precision-Recall curves visualizations
```

### Example: Multi-Class Classification

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier

models = {
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=None,
    X=X_train,
    y=y_train,
    task_type="classification",
    target_names=["Class A", "Class B", "Class C"],  # Named classes
    cv=5,
    verbose=True
)

# Metrics automatically computed for all classes
# ROC/PR curves show One-vs-Rest approach for multi-class
# Each class gets its own curve with separate AUC score
```

---

## Regression Metrics

### Available Metrics

**Error/Scale Metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

**Percentage Metrics:**
- Mean Absolute Percentage Error (MAPE)

**Goodness-of-Fit Metrics:**
- R-Squared (R²)
- Adjusted R² (accounts for number of features)

**Logarithmic Metrics:**
- Root Mean Squared Logarithmic Error (RMSLE)

**Diagnostic Plots (4 visualizations):**
1. Residuals vs Predicted
2. Residuals Distribution
3. Q-Q Plot (normality check)
4. Scale-Location Plot (variance homogeneity)

### Example: Regression with Diagnostics

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

param_grids = {
    "Ridge": {"model__alpha": [0.1, 1.0, 10.0]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1]},
    "RandomForest": {"model__max_depth": [5, 10, 15, None]}
}

results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=None,
    X=X_train,
    y=y_train,
    param_grids=param_grids,
    task_type="regression",
    test_size=0.2,
    plot_diagnostics=True,  # Shows 4 regression diagnostic plots
    search_type="grid",
    verbose=True
)

# Output includes:
# - Error/Scale Metrics: MAE, MSE, RMSE
# - Percentage Metrics: MAPE
# - Goodness-of-Fit: R², Adjusted R²
# - Logarithmic: RMSLE
# - 4 diagnostic plots for model assessment
```

### Interpreting Diagnostic Plots

**1. Residuals vs Predicted**
- ✓ Points should scatter randomly around zero
- ✗ Funnel pattern = heteroscedasticity (non-constant variance)
- ✗ Curved pattern = non-linearity

**2. Residuals Distribution**
- ✓ Bell-shaped (normal distribution)
- ✗ Skewed distribution = potential bias
- ✗ Heavy tails = outliers

**3. Q-Q Plot**
- ✓ Points should follow diagonal line
- ✗ Deviations at ends = non-normality

**4. Scale-Location Plot**
- ✓ Horizontal line with random scatter
- ✗ Trend upward/downward = heteroscedasticity

---

## Checkpoint & Resume

### Why Use Checkpoints?
- **Large Datasets**: Evaluation takes hours/days
- **Many Models**: Running 20+ models sequentially
- **Unstable Environments**: Network interruptions possible
- **Iterative Development**: Stop and restart evaluation

### Basic Usage

```python
results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=None,
    X=X_train,
    y=y_train,
    save_dir="./ml_checkpoints",  # Enable checkpointing
    resume=False,                   # Start fresh
    verbose=True
)

# Files created:
# - ./ml_checkpoints/model1_best_20260407_143022.pkl
# - ./ml_checkpoints/model2_best_20260407_143022.pkl
# - ./ml_checkpoints/eval_checkpoint.pkl (state file)
```

### Resume After Interruption

```python
# First run (interrupted after 5/10 models)
results = evaluate_and_plot_models(
    models={"model1": ..., "model2": ..., ..., "model10": ...},
    X=X_train,
    y=y_train,
    save_dir="./ml_checkpoints",
    resume=False,  # First run
    verbose=True
)
# Stops after processing 5 models

# Resume run (continues from model 6)
results = evaluate_and_plot_models(
    models={"model1": ..., "model2": ..., ..., "model10": ...},
    X=X_train,
    y=y_train,
    save_dir="./ml_checkpoints",
    resume=True,   # Resume from checkpoint
    verbose=True
)
# Resumes from model 6, skips models 1-5
```

### What Gets Saved

```
eval_checkpoint.pkl contains:
├── results_score: List of CV score arrays
├── test_scores_list: Test scores for each model
├── successful_names: Models that completed
├── summary_data: Performance summaries
└── best_estimators: Trained model pipelines

Individual model files:
├── model1_best_20260407_143022.pkl
├── model2_best_20260407_143022.pkl
└── ...
```

---

## Complete Examples

### Example 1: Classification with Everything

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from my_ml_toolkit import evaluate_and_plot_models
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
X = pd.DataFrame(
    np.random.randn(500, 20),
    columns=[f"feature_{i}" for i in range(20)]
)
y = pd.Series(np.random.randint(0, 3, 500))  # 3 classes

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
}

# Define preprocessing pipelines
preprocess = Pipeline([
    ("scaler", StandardScaler())
])

# Define parameter grids
param_grids = {
    "LogisticRegression": {
        "model__C": [0.1, 1.0, 10.0],
        "model__penalty": ["l2"]
    },
    "RandomForest": {
        "model__max_depth": [5, 10, 15],
        "model__min_samples_split": [2, 5]
    },
    "GradientBoosting": {
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [3, 5]
    }
}

# Run evaluation
results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=preprocess,
    X=X,
    y=y,
    param_grids=param_grids,
    task_type="classification",
    target_names=["Class_0", "Class_1", "Class_2"],
    test_size=0.2,
    val_size=0.1,
    split_method="random",
    cv=5,
    search_type="halving",          # Use HalvingGridSearchCV
    primary_metric="f1_macro",
    top_k=2,                        # Screening: keep top 2
    plot_lc=True,
    plot_diagnostics=True,
    plot_importance=False,
    plot_comparison=True,
    save_dir="./models",
    resume=False,
    verbose=True
)

# Access results
print("\nBest Model:", results["ultimate_winner"])
print("\nSummary:")
print(results["summary_df"])
print("\nData Splits:")
print(f"Train: {results['data_splits']['X_train'].shape}")
print(f"Val: {results['data_splits']['X_val'].shape}")
print(f"Test: {results['data_splits']['X_test'].shape}")
```

### Example 2: Regression with Advanced Diagnostics

```python
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from my_ml_toolkit import evaluate_and_plot_models

# Create regression data
np.random.seed(42)
X = pd.DataFrame(
    np.random.randn(300, 15),
    columns=[f"X_{i}" for i in range(15)]
)
y = pd.Series(X.iloc[:, 0] * 2 + X.iloc[:, 1] * 3 + np.random.randn(300) * 0.5)

# Preprocessing
preprocess = Pipeline([
    ("scaler", StandardScaler())
])

# Models
models = {
    "Ridge": Ridge(random_state=42),
    "ElasticNet": ElasticNet(random_state=42)
}

# Parameter grids
param_grids = {
    "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0]},
    "ElasticNet": {
        "model__alpha": [0.01, 0.1, 1.0],
        "model__l1_ratio": [0.1, 0.5, 0.9]
    }
}

# Run with diagnostic plots
results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=preprocess,
    X=X,
    y=y,
    param_grids=param_grids,
    task_type="regression",
    test_size=0.2,
    cv=5,
    search_type="grid",
    primary_metric="r2",
    plot_lc=True,
    plot_diagnostics=True,  # Shows all 4 residual plots
    plot_importance=True,    # Feature importance
    save_dir="./reg_models",
    verbose=True
)

# Analyze results
best_model = results["ultimate_winner"]
summary = results["summary_df"]
print(summary)

# Access metrics from summary
print("\nBest Model Metrics:")
print(f"R²: {summary.iloc[0]['Test R²']}")
print(f"RMSE: {summary.iloc[0]['Test RMSE']}")
```

### Example 3: Sequential Split (Time Series)

```python
# Time series data
time_series_X = pd.DataFrame(
    np.random.randn(1000, 5),
    columns=[f"feature_{i}" for i in range(5)]
)
time_series_y = pd.Series(
    np.cumsum(np.random.randn(1000))  # Trending data
)

models = {
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=None,
    X=time_series_X,
    y=time_series_y,
    task_type="regression",
    test_size=0.2,
    split_method="sequential",      # No shuffling (preserve temporal order)
    cv=3,
    plot_diagnostics=True,
    verbose=True
)

# Output shows sequential split information
print(results["data_splits"]["X_train"].iloc[-5:])  # Last 5 training samples
print(results["data_splits"]["X_test"].iloc[:5])    # First 5 test samples
```

---

## 📊 Output Interpretation

### Classification Output Example
```
======================================================================
Classification Metrics for LogisticRegression:
======================================================================
Core Scalar Metrics:
  • Accuracy:  0.9450
  • Precision: 0.9623
  • Recall:    0.9372
  • F1-Score:  0.9496
Specialized Metrics:
  • MCC (Matthews Correlation Coefficient): 0.8901
  • Cohen's Kappa: 0.8756
Probabilistic Metrics:
  • Log-Loss: 0.1845
Threshold-Agnostic Metrics:
  • ROC-AUC: 0.9876
======================================================================
```

### Regression Output Example
```
======================================================================
Regression Metrics for Ridge:
======================================================================
Error/Scale Metrics:
  • MAE (Mean Absolute Error):        0.4521
  • MSE (Mean Squared Error):         0.2847
  • RMSE (Root Mean Squared Error):   0.5335
Percentage Metrics:
  • MAPE (Mean Absolute % Error):     3.2145
Goodness-of-Fit Metrics:
  • R² Score:                         0.8932
  • Adjusted R²:                      0.8891
Logarithmic Metrics:
  • RMSLE (Root Mean Squared Log Error): 0.0456
======================================================================
```

---

## 🎯 Best Practices

### 1. **Choose Right Search Method**
```python
# Small space: Use Grid
if len(param_grid) < 50:
    search_type = "grid"
# Medium space: Use Random
elif len(param_grid) < 500:
    search_type = "random"
# Large space: Use Halving
else:
    search_type = "halving"
```

### 2. **Enable Checkpointing for Long Runs**
```python
# Any evaluation > 30 minutes
results = evaluate_and_plot_models(
    ...,
    save_dir="./checkpoints",
    resume=False,
    verbose=True
)
```

### 3. **Use Feature Names for Better Output**
```python
results = evaluate_and_plot_models(
    ...,
    feature_names=X.columns.tolist(),
    target_names=["negative", "positive"],  # For classification
    plot_importance=True
)
```

### 4. **Set Primary Metric Appropriately**
```python
# Classification
primary_metric = "f1_macro"      # Multi-class
primary_metric = "roc_auc"       # Binary, probability focus

# Regression
primary_metric = "r2"            # Explained variance
primary_metric = "neg_mean_squared_error"  # Error minimization
```

### 5. **Interpret Adjusted R² Correctly**
```python
# Adjusted R² penalizes additional features
# Use for comparing models with different feature counts
# If Adjusted R² < R²: model is overfitting
```

---

## 🔧 Troubleshooting

### Issue: "HalvingGridSearchCV doesn't work"
**Solution**: Ensure `scipy` is installed
```bash
pip install scipy
```

### Issue: "Log-Loss calculation failed"
**Solution**: Model doesn't support `predict_proba`
**Fix**: Add probability support or skip the metric (automatic fallback)

### Issue: "RMSLE calculation failed"
**Solution**: Target or predictions have negative values
**Fix**: RMSLE only works with positive values

### Issue: "Checkpoint file corrupted"
**Solution**: Delete `eval_checkpoint.pkl` and restart
```bash
rm -f ./checkpoints/eval_checkpoint.pkl
```

---

## 📚 Summary

| Feature | When to Use | Key Benefit |
|---------|------------|------------|
| **HalvingGridSearchCV** | Large parameter spaces | 5-10x faster tuning |
| **Classification Metrics** | Any classification task | Comprehensive evaluation |
| **Regression Metrics** | Any regression task | Deep diagnostics & analysis |
| **Checkpoint & Resume** | Long-running evaluations | Fault tolerance & efficiency |

---

**Happy modeling! 🚀**
