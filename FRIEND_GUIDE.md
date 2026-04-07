# 🚀 ML Toolkit - Setup Guide for Friends

Welcome! This guide will help you install and use my ML Toolkit package. It's super simple!

---

## ⚡ 5-Minute Setup

### Step 1: Install the Package (1 minute)

Open your terminal and run:

```bash
pip install git+https://github.com/LAFFI01/MY_ML_TOOLS.git
```

**That's it!** The package and all its dependencies are automatically installed.

### Step 2: Verify Installation (30 seconds)

Run this command to confirm everything works:

```bash
python -c "from my_ml_toolkit import evaluate_and_plot_models; print('✅ Ready to go!')"
```

If you see `✅ Ready to go!`, you're all set!

---

## 📝 Your First Program

Create a new file called `my_first_ml_project.py`:

```python
"""My first ML project using ML Toolkit."""

from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Step 1: Create sample data
print("📊 Creating sample data...")
X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_informative=8,
    n_classes=2,
    random_state=42
)

# Step 2: Split data into training and testing sets
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Create two models to compare
print("🤖 Creating models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Step 4: Evaluate models using ML Toolkit
print("📈 Evaluating and comparing models...\n")
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

# Step 5: Display results
print("\n✅ Final Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  • {metric}: {value:.4f}")
```

### Run Your Program

```bash
python my_first_ml_project.py
```

You'll see:
- 📊 Sample data being created
- 🔀 Data being split
- 🤖 Models being trained
- 📈 Models being evaluated
- A comparison plot showing both models' performance
- ✅ Detailed metrics for each model

---

## 💡 What You Can Do

The `evaluate_and_plot_models()` function helps you:

1. **Compare multiple models** - See which one performs best
2. **Get detailed metrics** - Accuracy, precision, recall, F1-score, etc.
3. **Visualize performance** - See comparison plots automatically
4. **Handle classification AND regression** - Works for both task types

### For Classification:

```python
results = evaluate_and_plot_models(
    models=your_models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="classification",  # ← Use this for classification
    verbose=True,
    plot_comparison=True
)
```

### For Regression:

```python
results = evaluate_and_plot_models(
    models=your_models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type="regression",  # ← Use this for regression
    verbose=True,
    plot_comparison=True
)
```

---

## 🐛 Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'my_ml_toolkit'`

**Solution:** Make sure you installed the package:
```bash
pip install git+https://github.com/LAFFI01/MY_ML_TOOLS.git
```

### Problem: `git command not found`

**Solution:** Install Git from https://git-scm.com, then try again.

### Problem: `pip: command not found`

**Solution:** Make sure Python is installed. Download from https://python.org

Check your Python version:
```bash
python --version
```

Should be Python 3.8 or higher.

---

## 📚 More Examples

Check out the `examples/` folder in the repository for more advanced examples:
- Classification on real datasets
- Regression examples
- Hyperparameter tuning
- And more!

---

## 🎓 Key Features

✅ **Easy to use** - One function does all the heavy lifting  
✅ **Automatic metrics** - Get metrics without manual calculation  
✅ **Beautiful plots** - Comparison visualizations included  
✅ **Handles imbalanced data** - Built-in support  
✅ **Works with any scikit-learn model** - Compatible with all sklearn models  

---

## 🤝 Questions?

If something doesn't work:
1. Check that Python 3.8+ is installed
2. Make sure pip installed the package correctly
3. Try the installation command again if there were errors
4. Check that you copied the code correctly

---

## 🎉 You're Ready!

Now you have a powerful ML evaluation toolkit at your fingertips. Start experimenting with different models and datasets!

Happy coding! 🚀
