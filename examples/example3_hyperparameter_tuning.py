"""
Example 3: Hyperparameter Tuning

Example showing how to use hyperparameter tuning with ML Toolkit.
Run: python example3_hyperparameter_tuning.py
"""

from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def main():
    print("=" * 60)
    print("Example 3: Hyperparameter Tuning")
    print("=" * 60)

    # Generate data
    print("\n📊 Generating classification data...")
    X, y = make_classification(
        n_samples=300, n_features=15, n_informative=10, n_classes=2, random_state=42
    )

    # Split data
    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    print("\n🤖 Creating models with base parameters...")
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf"),
    }

    # Define hyperparameter grids for tuning
    print("\n⚙️  Defining hyperparameter grids...")
    hyperparameter_grids = {
        "RandomForest": {
            "n_estimators": [50, 100, 150],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
        "GradientBoosting": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        },
        "SVM": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10],
        },
    }
    print("  ✓ Hyperparameter grids configured")

    # Evaluate with tuning
    print("\n📈 Evaluating models with hyperparameter tuning...\n")
    print(
        "  ⏳ This may take a moment as it tests multiple parameter combinations...\n"
    )

    results = evaluate_and_plot_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type="classification",
        primary_metric="f1",
        enable_hyperparameter_tuning=True,  # ← Enable tuning
        hyperparameter_grids=hyperparameter_grids,  # ← Pass grids
        cv_folds=5,
        verbose=True,
        plot_comparison=True,
    )

    # Display results
    print("\n✅ Results After Tuning:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  • {metric:.<25} {value:.4f}")

    # Highlight best model
    best_model = max(results, key=lambda x: results[x]["f1"])
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   F1-Score: {results[best_model]['f1']:.4f}")


if __name__ == "__main__":
    main()
