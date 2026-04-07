# 🚀 ML Toolkit - Installation & Usage Manual

**Enterprise-grade machine learning evaluation pipeline for classification and regression tasks.**

---

## 📦 Quick Installation

```bash
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git
```

That's it! All dependencies install automatically.

---

## ⚡ 30-Second Quick Start

```python
from my_ml_toolkit import evaluate_and_plot_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Create data
X, y = make_classification(n_samples=200, n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Compare models
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

# View results
for model, metrics in results.items():
    print(f"{model}: {metrics['accuracy']:.4f}")
```

---

## 📚 Documentation

| Document | Purpose | Time |
|----------|---------|------|
| **[INSTALL.md](INSTALL.md)** | Installation & troubleshooting | 5 min |
| **[QUICKSTART.md](QUICKSTART.md)** | 5-minute hands-on guide | 5 min |
| **[API_REFERENCE.md](API_REFERENCE.md)** | Complete API documentation | 20 min |
| **[FAQ.md](FAQ.md)** | Common questions & answers | 10 min |
| **[examples/](examples/)** | Working code examples | 10 min |

---

## ✨ Features

✔️ **Multiple Model Comparison**
- Compare 2+ models at once
- Automatic ranking by metric

✔️ **Comprehensive Metrics**
- Classification: Accuracy, Precision, Recall, F1, AUC
- Regression: MAE, MSE, RMSE, R²

✔️ **Built-in Visualizations**
- Confusion matrices
- ROC curves
- Model comparison plots

✔️ **Cross-Validation**
- K-fold validation built-in
- Customizable folds

✔️ **Hyperparameter Tuning**
- Automatic GridSearch
- Multiple parameter configurations

✔️ **Automatic Dependency Installation**
- No manual setup needed
- Works with pandas, numpy, scikit-learn, matplotlib, etc.

---

## 🎯 Use Cases

### 1. Compare Models Quickly
```python
results = evaluate_and_plot_models(
    models={"RF": RandomForestClassifier(), "SVM": SVC()},
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test
)
```

### 2. Find Best Model
```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    top_k=1  # Only best model
)
```

### 3. Regression Tasks
```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    task_type="regression"
)
```

### 4. Optimize Parameters
```python
results = evaluate_and_plot_models(
    models=models,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    enable_hyperparameter_tuning=True,
    hyperparameter_grids={"RF": {"n_estimators": [50, 100, 200]}}
)
```

---

## 🔧 What Gets Installed

**Core Dependencies** (automatic):
- pandas ≥2.0.0
- numpy ≥1.24.0
- matplotlib ≥3.7.0
- scikit-learn ≥1.5.0
- imbalanced-learn ≥0.11.0
- joblib ≥1.0.0

**Optional** (development):
```bash
pip install git+....[dev]  # Testing & linting tools
```

---

## 📖 Installation Methods

### Method 1: From GitHub (Recommended)
```bash
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git
```

### Method 2: Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/MY_ML.git
cd MY_ML
pip install -e .
```

### Method 3: From PyPI (After Publishing)
```bash
pip install my-ml-toolkit
```

---

## 🎓 Learning Path

### Beginner (5 minutes)
1. Read [INSTALL.md](INSTALL.md)
2. Run Example 1: `python examples/example1_basic_classification.py`

### Intermediate (15 minutes)
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run Examples 2 & 4

### Advanced (30 minutes)
1. Read [API_REFERENCE.md](API_REFERENCE.md)
2. Run Example 3 (hyperparameter tuning)
3. Build custom models

---

## 📂 Project Structure

```
MY_ML/
├── my_ml_toolkit/          # Main package
│   ├── __init__.py
│   └── evaluator.py        # Core functions
├── tests/                  # Test suite
├── examples/               # Working examples
├── INSTALL.md             # Installation guide ← READ THIS FIRST
├── QUICKSTART.md          # 5-minute guide
├── API_REFERENCE.md       # Complete API docs
├── FAQ.md                 # Q&A
└── CONTRIBUTING.md        # Contributing guide
```

---

## 🚀 First Steps

### Step 1: Verify Installation
```bash
python -c "from my_ml_toolkit import evaluate_and_plot_models; print('✅ Ready!')
```

### Step 2: Run First Example
```bash
python examples/example1_basic_classification.py
```

### Step 3: Try with Your Data
Follow guides:
- [QUICKSTART.md](QUICKSTART.md) - Hands-on guide
- [FAQ.md](FAQ.md) - Common questions
- [examples/example4_real_dataset.py](examples/example4_real_dataset.py) - Real data example

---

## ❓ Common Issues

### "pip: command not found"
```bash
# Try python3
python3 -m pip install git+https://...

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install git+https://...
```

### "No module named my_ml_toolkit"
```bash
# Reinstall
pip install --force-reinstall git+https://...

# Or check installation
pip list | grep my-ml-toolkit
```

### Need help?
See [FAQ.md](FAQ.md) for solutions to common problems.

---

## 📞 Support & Resources

| Resource | Link |
|----------|------|
| **Issues** | https://github.com/YOUR_USERNAME/MY_ML/issues |
| **Discussion** | https://github.com/YOUR_USERNAME/MY_ML/discussions |
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Development** | [DEVELOPMENT.md](my_ml_toolkit/DEVELOPMENT.md) |

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style
- Testing requirements
- Pull request process

---

## 📊 Quick Comparison: Classification

```python
from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

results = evaluate_and_plot_models(
    models={"RF": RandomForestClassifier(), "SVM": SVC()},
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    task_type="classification"
)
```

**Output:**
```
RandomForest:   Accuracy: 1.0000
SVM:            Accuracy: 0.9333
```

---

## 💡 Pro Tips

1. **Always scale features** for distance-based models:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   ```

2. **Use stratified split** for imbalanced data:
   ```python
   train_test_split(..., stratify=y)
   ```

3. **Save best model**:
   ```python
   import joblib
   joblib.dump(best_model, "model.pkl")
   ```

4. **Check reproducibility**:
   ```python
   results = evaluate_and_plot_models(..., random_state=42)
   ```

---

## 📈 Next Steps

1. ✅ Install package
2. ✅ Run examples
3. ✅ Read documentation
4. ✅ Try with your data
5. ✅ Contribute improvements

---

## 📄 License

MIT License - See [LICENSE](my_ml_toolkit/LICENSE) for details

---

## 👤 Author

**LAFFI01**

- GitHub: https://github.com/LAFFI01
- Repository: https://github.com/LAFFI01/MY_ML

---

## 🎉 Ready to Get Started?

**Start here:**
1. [INSTALL.md](INSTALL.md) - Installation guide
2. [QUICKSTART.md](QUICKSTART.md) - 5-minute tutorial
3. [examples/](examples/) - Working code examples

Happy modeling! 🚀

---

**Last Updated**: April 7, 2026
**Version**: 0.1.0
