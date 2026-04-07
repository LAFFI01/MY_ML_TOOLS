# Frequently Asked Questions (FAQ)

Common questions and answers about ML Toolkit.

---

## 📦 Installation & Setup

### Q1: How do I install the package?

**A:** Three ways:

```bash
# From GitHub (recommended)
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git

# Clone & install locally
git clone https://github.com/YOUR_USERNAME/MY_ML.git
cd MY_ML
pip install -e .

# From PyPI (after publishing)
pip install my-ml-toolkit
```

**See**: [INSTALL.md](INSTALL.md)

---

### Q2: What if I get "permission denied"?

**A:** Use one of these solutions:

```bash
# Option 1: Install for current user only
pip install --user git+https://github.com/YOUR_USERNAME/MY_ML.git

# Option 2: Use virtual environment (RECOMMENDED)
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git

# Option 3: Use conda
conda create -n myenv python=3.10
conda activate myenv
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git
```

---

### Q3: What Python versions are supported?

**A:** Python 3.8, 3.9, 3.10, 3.11, and 3.12

Check your version:
```bash
python --version
```

Upgrade if needed:
- **Windows**: https://www.python.org/downloads/
- **macOS**: `brew install python@3.11`
- **Linux**: `apt-get install python3.11`

---

### Q4: Do dependencies install automatically?

**A:** **Yes!** When you install the package, it automatically installs:
- pandas, numpy, matplotlib
- scikit-learn, imbalanced-learn
- joblib

No manual installation needed!

---

## 🚀 Usage & Features

### Q5: How do I use the package?

**A:** See [QUICKSTART.md](QUICKSTART.md) for quick examples.

Basic usage:

```python
from my_ml_toolkit import evaluate_and_plot_models
from sklearn.ensemble import RandomForestClassifier

results = evaluate_and_plot_models(
    models={"RandomForest": RandomForestClassifier()},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="classification"
)
```

---

### Q6: Can I use my own data?

**A:** **Yes!** The package works with any data format:

```python
# From pandas
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# From numpy arrays
X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([0, 1])

# From scikit-learn datasets
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
```

---

### Q7: What's the difference between regression and classification?

**A:** 

| Aspect | Classification | Regression |
|--------|-----------------|-----------|
| **Task** | Predict categories | Predict numbers |
| **Example** | Spam/Not spam | House price |
| **Metrics** | Accuracy, F1, AUC | MAE, MSE, R² |
| **Usage** | `task_type="classification"` | `task_type="regression"` |

---

### Q8: How do I compare multiple models?

**A:** Pass a dictionary of models:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf')
}

results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)
```

---

### Q9: How do I get only the best model?

**A:** Use `top_k=1`:

```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    top_k=1  # Only best model
)

best_model_name = list(results.keys())[0]
best_metrics = results[best_model_name]
print(f"Best model: {best_model_name}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
```

---

### Q10: How do I use hyperparameter tuning?

**A:** Enable tuning and provide parameter grids:

```python
results = evaluate_and_plot_models(
    models={
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC()
    },
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    enable_hyperparameter_tuning=True,
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
)
```

---

## 📊 Data & Metrics

### Q11: What metrics are computed?

**A:** Depends on task type:

**Classification:**
- Accuracy, Precision, Recall, F1-Score, AUC (ROC)

**Regression:**
- MAE, MSE, RMSE, R² Score

```python
# View results
results = evaluate_and_plot_models(...)

for metric, value in results["ModelName"].items():
    print(f"{metric}: {value:.4f}")
```

---

### Q12: How do I handle imbalanced data?

**A:** The package handles it automatically! But you can also:

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Use SMOTE to balance data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Use balanced data
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train_balanced,
    y_train=y_train_balanced,
    X_test=X_test,
    y_test=y_test
)
```

Or use class weights:
```python
models = {
    "RF": RandomForestClassifier(class_weight='balanced')
}
```

---

### Q13: How do I handle missing data?

**A:** Clean before using:

```python
import pandas as pd

# Remove rows with missing values
df = df.dropna()

# Or fill with mean
df = df.fillna(df.mean())

# Then use with package
X = df.drop("target", axis=1)
y = df["target"]
```

---

### Q14: What should my train/test split be?

**A:** Common ratios:
- **80/20**: Most common (80% train, 20% test)
- **70/30**: Larger test set
- **90/10**: Large training set

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 🐛 Troubleshooting

### Q15: "ModuleNotFoundError: No module named 'my_ml_toolkit'"

**A:** Package not properly installed.

```bash
# Check installation
pip list | grep my-ml-toolkit

# Reinstall
pip install --force-reinstall git+https://github.com/YOUR_USERNAME/MY_ML.git

# Or check Python path
python -c "import sys; print(sys.path)"
```

---

### Q16: "Shape mismatch" error

**A:** X and y shapes don't match.

```python
# Check shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# X_train shape should be (n_samples, n_features)
# y_train shape should be (n_samples,)

# If X is 1D, reshape it
X_train = X_train.reshape(-1, 1)
```

---

### Q17: "Negative y values in regression" error

**A:** Regression models can't have negative targets in some classes.

```python
# Check for negative values
if (y < 0).any():
    print("⚠️ Negative values found!")
    
# Solution: Scale or normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))
```

---

### Q18: Program runs very slowly

**A:** Try these optimizations:

```python
# Reduce cross-validation folds
results = evaluate_and_plot_models(
    ...,
    cv_folds=3,  # Reduce from 5
    verbose=False  # Skip verbose output
)

# Disable unnecessary plots
results = evaluate_and_plot_models(
    ...,
    plot_comparison=False,
    plot_confusion_matrix=False,
    plot_roc_curve=False
)

# Use fewer models
models = {"BestModel": best_clf}  # Only best model
```

---

## 🔧 Advanced Topics

### Q19: How do I save a trained model?

**A:** Use joblib:

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save
joblib.dump(model, "my_model.pkl")

# Load
loaded_model = joblib.load("my_model.pkl")

# Use loaded model
predictions = loaded_model.predict(X_test)
```

---

### Q20: Can I use this with cross-validation?

**A:** **Yes!** The package uses k-fold cross-validation internally:

```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    cv_folds=10  # 10-fold cross-validation
)
```

---

### Q21: How do I set random seed for reproducibility?

**A:** Use `random_state` parameter:

```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    random_state=42  # Same seed = same results
)
```

---

### Q22: Can I use GPU for training?

**A:** Some models support GPU:

```python
# XGBoost with GPU
import xgboost as xgb

models = {
    "XGBoost": xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0)
}

results = evaluate_and_plot_models(models=models, ...)
```

---

## 📚 Resources

| Topic | Link |
|-------|------|
| **Getting Started** | [QUICKSTART.md](QUICKSTART.md) |
| **Installation** | [INSTALL.md](INSTALL.md) |
| **API Details** | [API_REFERENCE.md](API_REFERENCE.md) |
| **Examples** | `examples/` folder |
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **GitHub** | https://github.com/YOUR_USERNAME/MY_ML |

---

## 💬 Still Have Questions?

1. **Check documentation**: Read relevant .md files
2. **Search issues**: https://github.com/YOUR_USERNAME/MY_ML/issues
3. **Create issue**: Ask on GitHub
4. **Email**: your.email@example.com

---

**Last Updated**: April 7, 2026

