"""
VISUALIZATION_GUIDE.md - Industry-Grade Model Comparison & Visualization

Comprehensive guide for using the visualization module to create professional
ML model evaluation dashboards and comparison reports.
"""

# 🎨 VISUALIZATION MODULE GUIDE

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Visualization Types](#visualization-types)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)

---

## Overview

The visualization module provides **industry-grade plotting** for machine learning models:

✅ **Individual Model Dashboards** - 6-subplot performance analysis per model
✅ **Multi-Model Comparison** - Bar charts, heatmaps, radar charts
✅ **Classification Diagnostics** - ROC, PR curves, confusion matrices, calibration
✅ **Regression Diagnostics** - Residuals, Q-Q plots, error distributions
✅ **Feature Importance** - Both sklearn and SHAP-based importance
✅ **Learning Curves** - Training vs validation dynamics
✅ **Batch Reports** - Generate full comparison PDFs for all models

---

## Installation

```bash
# Visualization module is already included in my_ml_toolkit
from my_ml_toolkit.visualizations import (
    plot_model_performance_dashboard,
    plot_model_comparison,
    plot_model_comparison_heatmap,
    plot_model_comparison_radar,
    plot_learning_curves,
    plot_feature_importance,
    generate_comparison_report,
)
```

**Requirements:**
- matplotlib >= 3.5
- seaborn >= 0.12
- numpy
- pandas
- scikit-learn

---

## Quick Start

### Single Model Dashboard (🎯 6 subplots)

```python
from my_ml_toolkit.visualizations import plot_model_performance_dashboard
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Calculate custom metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
}

# Generate dashboard
plot_model_performance_dashboard(
    model_name="Random Forest Classifier",
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    task_type="classification",
    metrics_dict=metrics,
    save_path="plots/model_dashboard.png"
)
```

**Output:** 6-subplot dashboard with:
- Confusion matrix heatmap
- ROC-AUC curve
- Precision-Recall curve
- Calibration curve
- Class distribution comparison
- Metrics summary table

---

### Multi-Model Comparison (🏆 Ranked)

```python
from my_ml_toolkit.visualizations import plot_model_comparison
import pandas as pd

# Create results DataFrame
results_df = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "Logistic Regression"],
    "Test Accuracy": [0.95, 0.94, 0.88],
    "Precision": [0.96, 0.95, 0.89],
    "Recall": [0.94, 0.93, 0.87],
    "F1-Score": [0.95, 0.94, 0.88],
})

# Generate comparison
plot_model_comparison(
    results_df,
    metric_column="Test Accuracy",
    title="Breast Cancer - Model Comparison",
    figsize=(14, 6),
    save_path="plots/comparison.png"
)
```

**Output:**
- Left: Ranked horizontal bar chart
- Right: Medal-based ranking table (🥇🥈🥉)

---

### Comprehensive Heatmap (🔥 All Metrics at Once)

```python
from my_ml_toolkit.visualizations import plot_model_comparison_heatmap

plot_model_comparison_heatmap(
    results_df,
    metric_columns=["Test Accuracy", "Precision", "Recall", "F1-Score"],
    figsize=(12, 6),
    save_path="plots/heatmap.png"
)
```

**Output:**
- Normalized heatmap (0-1 scale)
- Actual values shown in cells
- Color gradient: Red (low) → Yellow (medium) → Green (high)

---

### Radar Chart Comparison (⭐ Multi-Dimensional)

```python
from my_ml_toolkit.visualizations import plot_model_comparison_radar

plot_model_comparison_radar(
    results_df,
    metric_columns=["Test Accuracy", "Precision", "Recall", "F1-Score"],
    top_n=5,
    figsize=(10, 10),
    save_path="plots/radar.png"
)
```

**Output:**
- Polar plot with top 5 models
- Each axis = 1 metric
- Area fill shows model polygon
- Great for visual comparison of model trade-offs

---

### Feature Importance Plot

```python
from my_ml_toolkit.visualizations import plot_feature_importance

# Extract importances from model
importances = model.feature_importances_ * 100
feature_names = ["Feature_1", "Feature_2", ...]
importance_dict = dict(zip(feature_names, importances))

plot_feature_importance(
    importance_dict,
    model_name="Random Forest",
    top_k=15,
    importance_type="Importance %",
    save_path="plots/feature_importance.png"
)
```

**Output:**
- Top 15 features with importance percentages
- Color gradient bars (low to high importance)
- Value labels on each bar

---

### Learning Curves

```python
from my_ml_toolkit.visualizations import plot_learning_curves
from sklearn.model_selection import learning_curve
import numpy as np

# Calculate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

# Plot
plot_learning_curves(
    train_scores=train_mean,
    val_scores=val_mean,
    train_sizes=train_sizes,
    model_name="Random Forest",
    metric_name="Accuracy",
    save_path="plots/learning_curve.png"
)
```

**Output:**
- Training score line (blue)
- Validation score line (red)
- Gray shaded area between curves (gap = overfitting)
- Annotations for final scores

---

### Full Comparison Report (📊 Batch Export)

```python
from my_ml_toolkit.visualizations import generate_comparison_report

# Prepare results dictionary
results_dict = {
    "Random Forest": {
        "y_true": y_test,
        "y_pred": rf_pred,
        "y_proba": rf_proba,
        "metrics": {"Accuracy": 0.95, "Precision": 0.96}
    },
    "Gradient Boosting": {
        "y_true": y_test,
        "y_pred": gb_pred,
        "y_proba": gb_proba,
        "metrics": {"Accuracy": 0.94, "Precision": 0.95}
    }
}

# Generate all plots at once
generate_comparison_report(
    results_dict,
    save_dir="model_reports/",
    task_type="classification"
)
```

**Output in `model_reports/`:**
```
model_reports/
├── 01_dashboard_Random_Forest.png
├── 02_dashboard_Gradient_Boosting.png
├── 02_comparison_bar.png
├── 03_comparison_heatmap.png
├── 04_comparison_radar.png
└── comparison_summary.csv
```

---

## Visualization Types

### Classification Dashboard (6 Subplots)

| Plot | Purpose | Use When |
|------|---------|----------|
| **Confusion Matrix** | Shows TP/FP/TN/FN counts | Diagnosing misclassification patterns |
| **ROC-AUC Curve** | AUC metric + curve visualization | Evaluating probability threshold trade-offs |
| **Precision-Recall** | PR metric + curve | When dataset is imbalanced |
| **Calibration Curve** | Probability reliability check | Ensuring confidence scores match reality |
| **Class Distribution** | Compare true vs pred counts | Identifying systematic biases |
| **Metrics Table** | Summary statistics | Quick reference of all metrics |

### Regression Dashboard (6 Subplots)

| Plot | Purpose | Use When |
|------|---------|----------|
| **Actual vs Predicted** | Scatter plot regression line | Checking prediction quality visually |
| **Residual Plot** | Errors vs predictions | Detecting heteroscedasticity |
| **Residual Histogram** | Error distribution | Checking normality (Gaussian assumption) |
| **Q-Q Plot** | Quantile-quantile comparison | Formal normality test |
| **Error Distribution** | Absolute error histogram | Understanding typical error magnitude |
| **Metrics Summary** | MAE, RMSE, etc. | Quick performance reference |

### Comparison Plots

| Plot | Best For |
|------|----------|
| **Ranked Bar Chart** | Simple 1-metric comparison |
| **Heatmap** | All models × All metrics simultaneously |
| **Radar Chart** | Visual identification of model strengths/weaknesses |
| **Learning Curves** | Diagnosing bias-variance tradeoff |
| **Feature Importance** | Understanding model decisions |

---

## API Reference

### plot_model_performance_dashboard()

```python
def plot_model_performance_dashboard(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task_type: str = "classification",
    metrics_dict: Optional[Dict[str, float]] = None,
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None,
) -> None:
```

**Parameters:**
- `model_name` (str): Display name of the model
- `y_true` (np.ndarray): Ground truth labels/values
- `y_pred` (np.ndarray): Model predictions
- `y_proba` (np.ndarray, optional): Predicted probabilities (classification only)
- `task_type` (str): "classification" or "regression"
- `metrics_dict` (dict, optional): Custom metrics to display
- `figsize` (tuple): Figure size (width, height)
- `save_path` (str, optional): Path to save PNG file

---

### plot_model_comparison()

```python
def plot_model_comparison(
    results_df: pd.DataFrame,
    metric_column: str = "Test Accuracy",
    title: str = "Model Performance Comparison",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> None:
```

**Parameters:**
- `results_df` (pd.DataFrame): Models (col 0) × Metrics (remaining cols)
- `metric_column` (str): Which metric to rank by
- `title` (str): Plot title
- `figsize` (tuple): Figure dimensions
- `save_path` (str, optional): Output file path

---

### plot_model_comparison_heatmap()

```python
def plot_model_comparison_heatmap(
    results_df: pd.DataFrame,
    metric_columns: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> None:
```

**Parameters:**
- `results_df` (pd.DataFrame): Models × Metrics table
- `metric_columns` (list, optional): Specific metrics to include (auto-detects numeric if None)
- `figsize` (tuple): Figure size
- `save_path` (str, optional): Output path

---

### plot_model_comparison_radar()

```python
def plot_model_comparison_radar(
    results_df: pd.DataFrame,
    metric_columns: Optional[List[str]] = None,
    top_n: int = 5,
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None,
) -> None:
```

**Parameters:**
- `results_df` (pd.DataFrame): Models × Metrics
- `metric_columns` (list, optional): Metrics to include (max 6 for readability)
- `top_n` (int): Show top N models by average score
- `figsize` (tuple): Figure size
- `save_path` (str, optional): Output path

---

### plot_learning_curves()

```python
def plot_learning_curves(
    train_scores: List[float],
    val_scores: List[float],
    train_sizes: Optional[List[int]] = None,
    model_name: str = "Model",
    metric_name: str = "Accuracy",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> None:
```

---

### plot_feature_importance()

```python
def plot_feature_importance(
    importance_dict: Dict[str, float],
    model_name: str = "Model",
    top_k: int = 20,
    importance_type: str = "Percentage",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> None:
```

---

### generate_comparison_report()

```python
def generate_comparison_report(
    results_dict: Dict[str, Dict[str, Any]],
    save_dir: str = "./model_reports",
    task_type: str = "classification",
) -> None:
```

**Parameters:**
- `results_dict` (dict): `{model_name: {y_true, y_pred, y_proba, metrics}}`
- `save_dir` (str): Output directory
- `task_type` (str): "classification" or "regression"

---

## Advanced Usage

### Customizing Colors & Styles

```python
import matplotlib.pyplot as plt

# Change figure style
plt.style.use('seaborn-v0_8-darkgrid')

# Update color palette
import seaborn as sns
sns.set_palette("husl")

# Generate plots (will use new style)
plot_model_comparison(results_df)
```

### Saving High-Resolution Figures

```python
# All functions accept save_path parameter
plot_model_performance_dashboard(
    ...,
    save_path="plots/dashboard_600dpi.png"  # Automatically saves at 300 DPI
)

# For publication-quality output
plt.savefig("figure.pdf", dpi=600, bbox_inches="tight", format="pdf")
```

### Combining Multiple Comparisons

```python
# Generate all comparison types programmatically
import os

models = ["Model_A", "Model_B", "Model_C"]

for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    plot_model_comparison(
        results_df,
        metric_column=metric,
        title=f"Comparison by {metric}",
        save_path=f"plots/{metric.lower()}.png"
    )

print(f"✅ Generated {len(models)} comparison plots")
```

---

## Best Practices

### 1️⃣ Always Include Metrics Dictionary
```python
# ✅ GOOD
metrics = {
    "Accuracy": 0.95,
    "Precision": 0.96,
    "Recall": 0.94,
    "F1-Score": 0.95,
}
plot_model_performance_dashboard(..., metrics_dict=metrics)

# ❌ AVOID
plot_model_performance_dashboard(...)  # Less informative
```

### 2️⃣ Use Descriptive Model Names
```python
# ✅ GOOD
"Random Forest (n_estimators=100)"
"XGBoost (max_depth=5, lr=0.1)"

# ❌ AVOID
"Model 1"
"RF"
```

### 3️⃣ Save Plots for Reports
```python
# ✅ Always save
plot_model_comparison(..., save_path="plots/comparison.png")

# ✅ Organize output
os.makedirs("reports", exist_ok=True)
for model_name in models:
    plot_model_performance_dashboard(..., save_path=f"reports/{model_name}.png")
```

### 4️⃣ Use Batch Reports for Final Output
```python
# ✅ Professional: One function call generates entire report
generate_comparison_report(results_dict, save_dir="final_report")

# ❌ Tedious: Manual plot generation
plot_model_comparison(...)
plot_model_comparison_heatmap(...)
plot_model_comparison_radar(...)
# ... repeated for each model and metric
```

### 5️⃣ Verify Data Before Plotting
```python
# ✅ Check dimensions
assert len(y_true) == len(y_pred), "Mismatched array lengths"
assert len(y_true) > 0, "Empty predictions"

# ✅ Check probability shape
if y_proba is not None:
    assert y_proba.shape[0] == len(y_pred), "Probability shape mismatch"

plot_model_performance_dashboard(...)
```

---

## Troubleshooting

### Plot Not Showing
```python
# Make sure matplotlib is in interactive mode
import matplotlib.pyplot as plt
plt.ion()  # Interactive mode ON

# Or use Jupyter notebook (shows plots automatically)
```

### Memory Issues with Large Datasets
```python
# Plotting thousands of points can be slow
# Solution: Subsample for visualization

sample_size = 5000
if len(y_test) > sample_size:
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    y_true_sample = y_test[indices]
    y_pred_sample = y_pred[indices]
else:
    y_true_sample, y_pred_sample = y_test, y_pred

plot_model_performance_dashboard(
    ..., y_true=y_true_sample, y_pred=y_pred_sample
)
```

### Fonts Too Small
```python
import matplotlib.pyplot as plt

# Increase default font size
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12

# Then generate plots
plot_model_performance_dashboard(...)
```

---

## Integration with evaluate_and_plot_models()

The visualization module complements the main `evaluate_and_plot_models()` function:

```python
from my_ml_toolkit.evaluator import evaluate_and_plot_models
from my_ml_toolkit.visualizations import generate_comparison_report

# Run main evaluation
results = evaluate_and_plot_models(
    models=my_models,
    X=X,
    y=y,
    plot_comparison=True,  # Built-in comparison
    plot_importance=True,
)

# Generate additional detailed reports
custom_report_data = {
    model_name: {
        "y_true": results["test_labels"],
        "y_pred": results["predictions"][model_name],
        "metrics": results["scores"][model_name]
    }
    for model_name in my_models.keys()
}

generate_comparison_report(custom_report_data, save_dir="detailed_reports")
```

---

## Examples

See `example_visualization_guide.py` for 8 complete working examples:

```bash
python example_visualization_guide.py
# Choose: 1-7 (specific examples) or 0 (run all)
```

---

## License & Attribution

Part of the `my_ml_toolkit` - Modern ML Evaluation Framework

---

## Questions?

For issues or feature requests, check the [main README](README.md).
