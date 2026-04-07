"""
Example 2: Regression Task

Example showing how to use ML Toolkit for regression problems.
Run: python example2_regression.py
"""

from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def main():
    print("=" * 60)
    print("Example 2: Regression Task")
    print("=" * 60)

    # Generate regression data
    print("\n📊 Generating regression data...")
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=8, random_state=42
    )
    print(f"  ✓ Data shape: {X.shape}")
    print(f"  ✓ Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Split data
    print("\n🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models for regression
    print("\n🤖 Creating regression models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Support Vector Regressor": SVR(kernel="rbf"),
    }
    print(f"  ✓ Models: {list(models.keys())}")

    # Evaluate models
    print("\n📈 Evaluating models...\n")
    results = evaluate_and_plot_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type="regression",  # ← Key difference: regression task
        primary_metric="r2",     # ← Use R² as primary metric
        verbose=True,
        plot_comparison=True,
    )

    # Display results
    print("\n✅ Regression Results:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  • {metric:.<25} {value:.4f}")


if __name__ == "__main__":
    main()
