# Examples

Collection of practical examples showing how to use ML Toolkit.

---

## 📝 Available Examples

### Example 1: Basic Classification
**File**: `example1_basic_classification.py`

Simplest example to get started. Shows:
- Data generation
- Model creation
- Model evaluation
- Results display

**Run**:
```bash
python examples/example1_basic_classification.py
```

**Learn**: Basic workflow with ML Toolkit

---

### Example 2: Regression
**File**: `example2_regression.py`

Shows how to use ML Toolkit for regression tasks. Demonstrates:
- Regression data generation
- Multiple regression models
- Regression-specific metrics (MAE, MSE, RMSE, R²)
- Regression visualization

**Run**:
```bash
python examples/example2_regression.py
```

**Learn**: Regression task setup and evaluation

---

### Example 3: Hyperparameter Tuning
**File**: `example3_hyperparameter_tuning.py`

Advanced example showing hyperparameter optimization:
- Defining parameter grids
- Automatic tuning with GridSearch
- Finding best parameters
- Performance comparison

**Run**:
```bash
python examples/example3_hyperparameter_tuning.py
```

**Learn**: Hyperparameter tuning workflow (⏳ Takes 1-2 minutes)

---

### Example 4: Real Dataset
**File**: `example4_real_dataset.py`

Working with actual datasets (Iris):
- Loading real data
- Feature scaling
- Multi-model comparison
- Results interpretation

**Run**:
```bash
python examples/example4_real_dataset.py
```

**Learn**: Real-world usage patterns

---

## 🚀 Quick Start

### Run All Examples

```bash
cd /path/to/MY_ML

# Example 1 (fastest)
python examples/example1_basic_classification.py

# Example 2 (fast)
python examples/example2_regression.py

# Example 3 (slower - does tuning)
python examples/example3_hyperparameter_tuning.py

# Example 4 (medium)
python examples/example4_real_dataset.py
```

### Run Single Example

```bash
python examples/example1_basic_classification.py
```

---

## 📚 Learning Path

**Beginner** → Start with Example 1
```bash
python examples/example1_basic_classification.py
```

**Intermediate** → Try Examples 2 & 4
```bash
python examples/example2_regression.py
python examples/example4_real_dataset.py
```

**Advanced** → Use Example 3
```bash
python examples/example3_hyperparameter_tuning.py
```

---

## 🔍 What Each Example Shows

| Example | Task | Models | Features |
|---------|------|--------|----------|
| 1 | Classification | 2 | Basic |
| 2 | Regression | 3 | Multiple models |
| 3 | Classification | 3 | Hyperparameter tuning |
| 4 | Classification | 3 | Real data, scaling |

---

## 💡 Modification Ideas

### Add Your Own Data
```python
# Replace the data generation with your own
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1)
y = df["target"]
```

### Try Different Models
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
    "AdaBoost": AdaBoostClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
}
```

### Adjust Parameters
```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    cv_folds=10,              # More folds
    primary_metric="f1",      # Different metric
    random_state=123,         # Different seed
    verbose=False,            # Silent mode
)
```

---

## 📊 Output Examples

Each example produces:

1. **Console Output**
   - Step-by-step progress
   - Model evaluation metrics
   - Best model highlighted

2. **Visualizations** (if enabled)
   - Model comparison plots
   - Confusion matrices
   - ROC curves

3. **Metrics Tables**
   - Accuracy, Precision, Recall
   - F1-Score, AUC
   - Training time

---

## 🐛 Troubleshooting Examples

### Example 1 runs but no plots show?
```python
# Update to force display
plot_comparison=True  # Ensure enabled
verbose=True         # See details
```

### Example 3 is too slow?
```python
# Reduce parameter grid
hyperparameter_grids = {
    "RandomForest": {
        "n_estimators": [50, 100],  # Fewer options
        "max_depth": [5, 10],
    }
}
```

### Example 4 fails with scaling?
```python
# Use unscaled data
results = evaluate_and_plot_models(
    X_train=X_train,      # Not X_train_scaled
    X_test=X_test,        # Not X_test_scaled
    ...
)
```

---

## 📖 Related Documentation

- [QUICKSTART.md](../QUICKSTART.md) - 5-minute guide
- [API_REFERENCE.md](../API_REFERENCE.md) - Complete API docs
- [INSTALL.md](../INSTALL.md) - Installation guide
- [FAQ.md](../FAQ.md) - Common questions

---

## 💬 Create Your Own Example!

Have a great use case? Create a new example file:

```python
"""
Example 5: Your Custom Use Case

Description of what this example demonstrates.
Run: python example5_your_example.py
"""

from my_ml_toolkit import evaluate_and_plot_models
# ... your code ...

if __name__ == "__main__":
    main()
```

Then submit a Pull Request! 🤝

---

## ⏱️ Execution Times (Approximate)

| Example | Time | Note |
|---------|------|------|
| 1 | 5-10s | Fast |
| 2 | 5-10s | Fast |
| 3 | 60-120s | Tuning included |
| 4 | 10-20s | Medium |

*Times vary by system and parameters*

---

Happy learning! 🎉

For questions, see [FAQ.md](../FAQ.md) or open an issue.
