# API Reference

Complete reference for ML Toolkit functions and parameters.

---

## 📌 Metrics & Scoring Functions

### `balanced_multiclass_accuracy(y_true, y_pred) → float`

Balanced accuracy for multi-class classification (macro-averaged per-class recall).

**Why use this?** Standard accuracy can be misleading with imbalanced classes. This metric treats all classes equally.

**Formula:**
```
Accuracy = (1/C) × Σ(TP_i / All_samples_of_class_i)
```

**Parameters:**
- `y_true`: True labels (numpy array or pandas Series)
- `y_pred`: Predicted labels (numpy array or pandas Series)

**Returns:** Float between 0 and 1

**Example:**
```python
from my_ml_toolkit import balanced_multiclass_accuracy
import numpy as np

y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 1, 0, 1, 2])

score = balanced_multiclass_accuracy(y_true, y_pred)
print(f"Balanced Accuracy: {score:.4f}")  # Output: 0.7222
```

---

### `get_balanced_accuracy_scorer() → Scorer`

Create a scikit-learn compatible scorer for `balanced_multiclass_accuracy`.

**Returns:** `_Scorer` object compatible with `GridSearchCV`, `cross_val_score`, etc.

**Example:**
```python
from my_ml_toolkit import get_balanced_accuracy_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

scorer = get_balanced_accuracy_scorer()
rf = RandomForestClassifier()

scores = cross_val_score(rf, X, y, cv=5, scoring=scorer)
print(f"Mean CV Score: {scores.mean():.4f}")
```

---

## 📌 Main Function: `evaluate_and_plot_models`

Enterprise machine learning evaluation pipeline for both classification and regression tasks.

### Signature

```python
def evaluate_and_plot_models(
    models: Dict[str, Any],
    preprocess_pipeline: Union[Any, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.0,
    split_method: str = "random",
    stratify: bool = True,
    random_seed: int = 42,
    task_type: str = "classification",
    target_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    param_grids: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Dict[str, Any]]] = None,
    sampler: Optional[Any] = None,
    cv: int = 4,
    search_type: str = "grid",
    primary_metric: Optional[str] = None,
    top_k: Optional[int] = None,
    quick_test_fraction: float = 0.2,
    plot_lc: bool = True,
    plot_diagnostics: bool = True,
    plot_importance: bool = True,
    plot_comparison: bool = True,
    top_n_features: int = 20,
    save_dir: Optional[str] = None,
    resume: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]
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

## 📖 Parameters

### **Core Data Parameters**

#### `models` (required)
- **Type**: `Dict[str, Any]`
- **Description**: Dictionary mapping model names to Scikit-Learn estimator instances
- **Example**:
  ```python
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  
  models = {
      "LogisticRegression": LogisticRegression(),
      "RandomForest": RandomForestClassifier(random_state=42),
  }
  ```

#### `preprocess_pipeline` (required)
- **Type**: `Any` or `Dict[str, Any]`
- **Description**: Sklearn Pipeline or dict mapping model names to preprocessing pipelines. Use "default" key as fallback for unlisted models.
- **Example**:
  ```python
  from sklearn.preprocessing import StandardScaler, MinMaxScaler
  from sklearn.pipeline import Pipeline
  
  # Single pipeline for all models
  preprocess_pipeline = Pipeline([("scaler", StandardScaler())])
  
  # Per-model preprocessing (dict)
  preprocess_pipeline = {
      "LogisticRegression": Pipeline([("scaler", StandardScaler())]),
      "RandomForest": None,  # No preprocessing
      "default": Pipeline([("scaler", MinMaxScaler())])
  }
  ```

#### `X` (required)
- **Type**: `pd.DataFrame`
- **Description**: Feature matrix with shape (n_samples, n_features)
- **Example**: `X.shape  # (1000, 20)`

#### `y` (required)
- **Type**: `pd.Series`
- **Description**: Target variable with shape (n_samples,)
- **Example**: `y.shape  # (1000,)`

---

### **Data Splitting Parameters**

#### `test_size`
- **Type**: `float`
- **Default**: `0.2`
- **Range**: (0.0, 1.0)
- **Description**: Fraction of data reserved for final test set
- **Example**: `test_size=0.25  # Reserve 25% for testing`

#### `val_size`
- **Type**: `float`
- **Default**: `0.0`
- **Range**: [0.0, 1.0)
- **Description**: Fraction of training data reserved for validation (used by boosting models for early stopping)
- **Example**: `val_size=0.1  # 10% of training data for validation`

#### `split_method`
- **Type**: `str`
- **Default**: `"random"`
- **Options**: `"random"` or `"sequential"`
- **Description**: Split strategy. Use "sequential" for time-series or financial data.
- **Example**: `split_method="sequential"`

#### `stratify`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Stratify splits by target class (classification only, ignored for regression)
- **Example**: `stratify=False  # Don't stratify`

#### `random_seed`
- **Type**: `int`
- **Default**: `42`
- **Description**: Random state for reproducibility
- **Example**: `random_seed=123`

---

### **Task Configuration Parameters**

#### `task_type`
- **Type**: `str`
- **Default**: `"classification"`
- **Options**: `"classification"` or `"regression"`
- **Description**: Type of ML task; auto-switches metrics, cross-validation splitter, and plots
- **Example**: `task_type="regression"`

#### `target_names`
- **Type**: `Optional[List[str]]`
- **Default**: `None`
- **Description**: Names of target classes (classification only) for reports
- **Example**: `target_names=["negative", "positive"]`

#### `feature_names`
- **Type**: `Optional[List[str]]`
- **Default**: `None`
- **Description**: Original feature names for importance plots
- **Example**: `feature_names=["age", "income", "credit_score"]`

---

### **Model Fitting Parameters**

#### `param_grids`
- **Type**: `Optional[Dict[str, Any]]`
- **Default**: `None`
- **Description**: Parameter grids for hyperparameter tuning keyed by model name
- **Example**:
  ```python
  param_grids={
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

#### `fit_params`
- **Type**: `Optional[Dict[str, Dict[str, Any]]]`
- **Default**: `None`
- **Description**: Additional fit keyword arguments per model (e.g., early stopping)
- **Example**:
  ```python
  fit_params={
      "XGBoost": {"early_stopping_rounds": 10, "eval_metric": "logloss"}
  }
  ```

#### `sampler`
- **Type**: `Optional[Any]`
- **Default**: `None`
- **Description**: Imbalanced data sampler (e.g., SMOTE). Ignored for regression.
- **Example**:
  ```python
  from imblearn.over_sampling import SMOTE
  sampler=SMOTE(random_state=42)
  ```

#### `cv`
- **Type**: `int`
- **Default**: `4`
- **Range**: 2+
- **Description**: Number of cross-validation folds
- **Example**: `cv=5`

#### `search_type`
- **Type**: `str`
- **Default**: `"grid"`
- **Options**: `"grid"`, `"random"`, or `"halving"`
- **Description**: Hyperparameter search method. "halving" uses successive halving for large-scale tuning.
- **Example**: `search_type="halving"`

---

### **Evaluation & Ranking Parameters**

#### `primary_metric`
- **Type**: `Optional[str]` or `_Scorer`
- **Default**: `None` (auto-selected: "accuracy" for classification, "r2" for regression)
- **Classification**: `"accuracy"`, `"precision"`, `"recall"`, `"f1"`, `"auc"`, `"kappa"`, `"log_loss"`, **`"balanced_accuracy"`**
- **Regression**: `"r2"`, `"mae"`, `"mse"`, `"rmse"`, `"mape"`, `"msle"`
- **Description**: Main metric for model ranking and hyperparameter tuning
- **When to use `"balanced_accuracy"`:** Use this for **imbalanced classification tasks** where classes have unequal sample counts. It calculates macro-averaged per-class recall, treating all classes equally regardless of their frequency.
- **Example - Imbalanced Data**:
  ```python
  # For imbalanced classification (e.g., fraud detection)
  results = evaluate_and_plot_models(
      models=models,
      preprocess_pipeline=preprocess,
      X=X, y=y,
      task_type="classification",
      primary_metric="balanced_accuracy",  # ← Use this for imbalance!
      cv=5
  )
  ```
- **Example - Balanced Data**:
  ```python
  # For well-balanced classification
  primary_metric="f1"  # Good general-purpose metric
  ```

#### `top_k`
- **Type**: `Optional[int]`
- **Default**: `None` (evaluate all)
- **Description**: Quick screening: evaluate top-k models on quick_test_fraction before full tuning
- **Example**: `top_k=3  # Quickly screen to best 3, then tune fully`

#### `quick_test_fraction`
- **Type**: `float`
- **Default**: `0.2`
- **Range**: (0.0, 1.0)
- **Description**: Fraction of training data for quick screening phase
- **Example**: `quick_test_fraction=0.1`

---

### **Visualization Parameters**

#### `plot_lc`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Generate learning curves
- **Example**: `plot_lc=False`

#### `plot_diagnostics`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Generate confusion matrices (classification) or residual plots (regression)
- **Example**: `plot_diagnostics=False`

#### `plot_importance`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Generate feature importance / coefficient plots
- **Example**: `plot_importance=False`

#### `plot_comparison`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Generate cross-validation score comparison boxplot
- **Example**: `plot_comparison=False`

#### `top_n_features`
- **Type**: `int`
- **Default**: `20`
- **Description**: Number of top features to display in importance plot
- **Example**: `top_n_features=10`

---

### **Checkpoint & Persistence Parameters**

#### `save_dir`
- **Type**: `Optional[str]`
- **Default**: `None`
- **Description**: Directory to save fitted models and checkpoint files. If None, no models saved.
- **Example**: `save_dir="./models"`

#### `resume`
- **Type**: `bool`
- **Default**: `False`
- **Description**: If True and checkpoint exists, resume training from last completed model. If False, start fresh.
- **Example**: `resume=True`

---

### **Logging Parameter**

#### `verbose`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Print progress messages during evaluation
- **Example**: `verbose=False`

---

## 📤 Return Value

Returns `Dict[str, Any]` containing:

- **`summary_df`**: DataFrame ranking all models by primary_metric
- **`best_models`**: Dict mapping model names to fitted pipeline objects
- **`raw_cv_scores`**: Dict mapping model names to cross-validation score arrays
- **`ultimate_winner`**: Best performing model's fitted pipeline
- **`data_splits`**: Dict with data splits:
  ```
  {
      "X_train": pd.DataFrame,
      "y_train": pd.Series,
      "X_val": pd.DataFrame or None,
      "y_val": pd.Series or None,
      "X_test": pd.DataFrame,
      "y_test": pd.Series
  }
  ```

---

## 🔍 Usage Examples

### Classification with SMOTE

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from my_ml_toolkit import evaluate_and_plot_models

# Load data
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
}

# Preprocessing  
preprocess_pipeline = Pipeline([("scaler", StandardScaler())])

# Run evaluation
results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=preprocess_pipeline,
    X=X,
    y=y,
    test_size=0.2,
    task_type="classification",
    sampler=SMOTE(random_state=42),
    cv=5,
    verbose=True,
)

print(results["summary_df"])
best_model = results["ultimate_winner"]
```

### Regression with Hyperparameter Tuning

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from my_ml_toolkit import evaluate_and_plot_models

# Load data
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Define models and hyperparameter grids
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

param_grids = {
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
    },
    "GradientBoosting": {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [50, 100],
    },
}

# Run evaluation
results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=None,
    X=X,
    y=y,
    task_type="regression",
    param_grids=param_grids,
    search_type="grid",
    cv=5,
)

print(results["summary_df"])
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
