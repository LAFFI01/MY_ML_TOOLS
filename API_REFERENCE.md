# API Reference

Complete reference for ML Toolkit functions and parameters.

---

## 📌 Main Function: `evaluate_and_plot_models`

Evaluate and compare multiple machine learning models.

### Signature

```python
def evaluate_and_plot_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    task_type: str = "classification",
    primary_metric: str = "accuracy",
    cv_folds: int = 5,
    test_size: float = None,
    verbose: bool = True,
    plot_comparison: bool = True,
    plot_confusion_matrix: bool = True,
    plot_roc_curve: bool = True,
    enable_hyperparameter_tuning: bool = False,
    hyperparameter_grids: Dict[str, Dict] = None,
    top_k: int = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]
```

---

## 📖 Parameters

### **Core Parameters**

#### `models` (required)
- **Type**: `Dict[str, Any]`
- **Description**: Dictionary of model names to sklearn estimator instances
- **Example**:
  ```python
  models = {
      "LogisticRegression": LogisticRegression(),
      "RandomForest": RandomForestClassifier(),
      "SVM": SVC()
  }
  ```

#### `X_train` (required)
- **Type**: `pd.DataFrame` or `np.ndarray`
- **Description**: Training features
- **Shape**: (n_samples, n_features)
- **Example**:
  ```python
  X_train.shape  # (800, 10)
  ```

#### `y_train` (required)
- **Type**: `pd.Series` or `np.ndarray`
- **Description**: Training labels/targets
- **Shape**: (n_samples,)

#### `X_test` (required)
- **Type**: `pd.DataFrame` or `np.ndarray`
- **Description**: Test features
- **Shape**: (n_samples, n_features)

#### `y_test` (required)
- **Type**: `pd.Series` or `np.ndarray`
- **Description**: Test labels/targets
- **Shape**: (n_samples,)

---

### **Configuration Parameters**

#### `task_type`
- **Type**: `str`
- **Default**: `"classification"`
- **Options**: `"classification"` or `"regression"`
- **Description**: Type of machine learning task
- **Example**:
  ```python
  task_type="classification"  # For classification
  task_type="regression"      # For regression
  ```

#### `primary_metric`
- **Type**: `str`
- **Default**: `"accuracy"`
- **Classification Options**: `"accuracy"`, `"precision"`, `"recall"`, `"f1"`, `"auc"`
- **Regression Options**: `"r2"`, `"mae"`, `"mse"`, `"rmse"`
- **Description**: Main metric for model ranking
- **Example**:
  ```python
  primary_metric="f1"  # Use F1-score as primary
  ```

#### `cv_folds`
- **Type**: `int`
- **Default**: `5`
- **Range**: 2-10 (typical)
- **Description**: Number of cross-validation folds
- **Example**:
  ```python
  cv_folds=10  # 10-fold cross-validation
  ```

#### `verbose`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Print evaluation progress and results
- **Example**:
  ```python
  verbose=False  # Silent mode
  ```

---

### **Visualization Parameters**

#### `plot_comparison`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Show model comparison plots
- **Example**:
  ```python
  plot_comparison=False  # Skip comparison plot
  ```

#### `plot_confusion_matrix`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Show confusion matrix for each model
- **Example**:
  ```python
  plot_confusion_matrix=True
  ```

#### `plot_roc_curve`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Show ROC curves (classification only)
- **Example**:
  ```python
  plot_roc_curve=True
  ```

---

### **Hyperparameter Tuning Parameters**

#### `enable_hyperparameter_tuning`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable automatic hyperparameter tuning
- **Example**:
  ```python
  enable_hyperparameter_tuning=True
  ```

#### `hyperparameter_grids`
- **Type**: `Dict[str, Dict]`
- **Default**: `None`
- **Description**: Parameter grids for GridSearchCV
- **Example**:
  ```python
  hyperparameter_grids={
      "RandomForest": {
          "n_estimators": [50, 100, 200],
          "max_depth": [5, 10, None]
      },
      "SVM": {
          "kernel": ["linear", "rbf"],
          "C": [0.1, 1, 10]
      }
  }
  ```

#### `top_k`
- **Type**: `int`
- **Default**: `None` (all models)
- **Description**: Return only top K models
- **Example**:
  ```python
  top_k=3  # Return best 3 models
  top_k=1  # Return only best model
  ```

---

### **Advanced Parameters**

#### `test_size`
- **Type**: `float`
- **Default**: `None`
- **Range**: 0.0-1.0
- **Description**: Test set size (if splitting data)
- **Example**:
  ```python
  test_size=0.2  # 20% test set
  ```

#### `random_state`
- **Type**: `int`
- **Default**: `42`
- **Description**: Random seed for reproducibility
- **Example**:
  ```python
  random_state=42  # Reproducible results
  ```

---

## 📤 Return Value

### Type
```python
Dict[str, Dict[str, float]]
```

### Structure
```python
{
    "model_name_1": {
        "accuracy": 0.92,
        "precision": 0.90,
        "recall": 0.92,
        "f1": 0.91,
        "auc": 0.95,
        "training_time": 0.234
    },
    "model_name_2": {
        "accuracy": 0.88,
        "precision": 0.85,
        "recall": 0.88,
        "f1": 0.86,
        "auc": 0.91,
        "training_time": 0.156
    }
}
```

### Access Results
```python
results = evaluate_and_plot_models(...)

# Get specific model metrics
model_metrics = results["RandomForest"]
accuracy = model_metrics["accuracy"]

# Get best model
best_model = max(results, key=lambda x: results[x]["accuracy"])
best_metrics = results[best_model]
```

---

## 🎨 Classification Metrics

### Metrics Returned

| Metric | Range | Best Value | Use Case |
|--------|-------|-----------|----------|
| **Accuracy** | 0-1 | 1.0 | Overall correctness |
| **Precision** | 0-1 | 1.0 | False positive penalty |
| **Recall** | 0-1 | 1.0 | False negative penalty |
| **F1-Score** | 0-1 | 1.0 | Balanced metric |
| **AUC (ROC)** | 0-1 | 1.0 | Threshold independence |

### When to Use Each

```python
# Class imbalance → Use F1 or AUC
primary_metric="f1"

# Cost of false positives high → Use Precision
primary_metric="precision"

# Cost of false negatives high → Use Recall
primary_metric="recall"

# Need overall performance → Use Accuracy
primary_metric="accuracy"
```

---

## 📉 Regression Metrics

### Metrics Returned

| Metric | Lower is Better | Use Case |
|--------|-----------------|----------|
| **MAE** | Yes | Average error magnitude |
| **MSE** | Yes | Penalize large errors |
| **RMSE** | Yes | Same units as target |
| **R² Score** | No | Variance explained (0-1) |

---

## 💡 Usage Examples

### Basic Classification
```python
from my_ml_toolkit import evaluate_and_plot_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

results = evaluate_and_plot_models(
    models={
        "RF": RandomForestClassifier(),
        "LR": LogisticRegression()
    },
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="classification"
)
```

### Regression with Tuning
```python
results = evaluate_and_plot_models(
    models={
        "RF": RandomForestRegressor(),
        "SVR": SVR()
    },
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="regression",
    enable_hyperparameter_tuning=True,
    hyperparameter_grids={
        "RF": {"n_estimators": [50, 100, 200]},
        "SVR": {"kernel": ["linear", "rbf"]}
    }
)
```

### Get Top 3 Models
```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    top_k=3,
    verbose=True
)
```

### Custom Metrics Focus
```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    primary_metric="f1",  # Rank by F1-score
    cv_folds=10,         # 10-fold CV
    random_state=123     # Reproducible
)
```

---

## 🔧 Error Handling

### Common Errors

```python
# Error: Model not compatible
# Solution: Use sklearn-compatible models

# Error: Shape mismatch
# Solution: Check X and y shapes match

# Error: No negative y values in regression
# Solution: Ensure y contains all valid values
```

---

## 📚 See Also

- [Quick Start Guide](QUICKSTART.md)
- [Installation Guide](INSTALL.md)
- [Examples Folder](../examples/)

---

## 💬 Questions?

- Check [CONTRIBUTING.md](CONTRIBUTING.md)
- Open issue on [GitHub](https://github.com/YOUR_USERNAME/MY_ML/issues)
