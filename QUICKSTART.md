# Quick Start Guide

Get started with ML Toolkit in 5 minutes!

---

## 1️⃣ Installation (30 seconds)

```bash
pip install git+https://github.com/LAFFI01/MY_ML_TOOLS.git
```

---

## 2️⃣ Simple Example (2 minutes)

Create a file called `demo.py`:

```python
"""Simple ML Toolkit demo."""

from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Step 1: Generate sample data
print("📊 Generating sample data...")
X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_informative=8,
    n_classes=2,
    random_state=42
)

# Step 2: Split into train/test
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Create models to compare
print("🤖 Creating models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Step 4: Evaluate using ML Toolkit
print("📈 Evaluating models...\n")
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="classification",
    verbose=True,
    plot_comparison=True
)

# Step 5: Print results
print("\n✅ Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  • {metric}: {value:.4f}")
```

**Run it:**
```bash
python demo.py
```

**Output:**
```
📊 Generating sample data...
🔀 Splitting data...
🤖 Creating models...
📈 Evaluating models...

[Model evaluation details...]

✅ Results:
Logistic Regression:
  • accuracy: 0.8500
  • precision: 0.8333
  • recall: 0.8571

Random Forest:
  • accuracy: 0.9000
  • precision: 0.9000
  • recall: 0.9000
```

---

## 3️⃣ Regression Example

```python
from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Create regression data
X, y = make_regression(n_samples=200, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models for regression
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
}

# Evaluate
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="regression",
    verbose=True,
    plot_comparison=True
)
```

---

## 4️⃣ Real Dataset Example

```python
from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Multiple models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='rbf'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

# Evaluate and compare
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="classification",
    verbose=True,
    plot_comparison=True,
    top_k=2  # Show top 2 models
)
```

---

## 5️⃣ With Hyperparameter Tuning

```python
from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

X, y = make_classification(n_samples=300, n_features=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define models with parameter grids
models = {
    "RF": RandomForestClassifier(n_estimators=100, random_state=42),
    "GB": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# Evaluate with tuning
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="classification",
    verbose=True,
    plot_comparison=True,
    enable_hyperparameter_tuning=True,  # Enable tuning
    hyperparameter_grids={
        "RF": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None]
        },
        "GB": {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [50, 100]
        }
    }
)
```

---

## 📊 Common Tasks

### Task 1: Compare 2 Models
```python
results = evaluate_and_plot_models(
    models={"Model1": clf1, "Model2": clf2},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="classification"
)
```

### Task 2: Get Best Model
```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    top_k=1  # Only return best model
)

best_model_name = list(results.keys())[0]
best_metrics = results[best_model_name]
```

### Task 3: Silent Mode (No Plots)
```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    verbose=False,
    plot_comparison=False
)
```

### Task 4: Only Get Metrics
```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    plot_confusion_matrix=False,
    plot_comparison=False
)
```

---

## 🎯 Key Features

✨ **Automatic Metrics**: accuracy, precision, recall, F1-score, confusion matrix
✨ **Visualizations**: Confusion matrix, ROC curve, comparison plots
✨ **Cross-Validation**: Built-in k-fold validation
✨ **Hyperparameter Tuning**: GridSearch & RandomSearch support
✨ **Imbalanced Data**: Handles class imbalance automatically
✨ **Multi-Model Compare**: Evaluate and rank multiple models at once

---

## 📚 Next Steps

- See [API_REFERENCE.md](API_REFERENCE.md) for all parameters
- Check [examples/](../examples/) for more examples
- Read [DEVELOPMENT.md](DEVELOPMENT.md) to contribute

---

## ❓ FAQ

**Q: How do I use my own data?**
A: Replace `make_classification()` with your data loading code.

**Q: Can I use it with pandas DataFrames?**
A: Yes! Just convert to numpy: `X.values, y.values`

**Q: What metrics are computed?**
A: Classification: accuracy, precision, recall, F1, AUC
   Regression: MAE, MSE, RMSE, R²

**Q: Can I save the trained model?**
A: Yes, use `joblib.dump()` on your trained model.

---

Need help? Check [INSTALL.md](INSTALL.md) or open a GitHub issue! 🤝
