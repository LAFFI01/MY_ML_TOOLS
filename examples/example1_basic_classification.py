"""
Example 1: Basic Classification

Simplest example to get started with ML Toolkit.
Run: python example1_basic_classification.py
"""

from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def main():
    print("=" * 60)
    print("Example 1: Basic Classification")
    print("=" * 60)

    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating sample data...")
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_classes=2,
        random_state=42,
    )
    print(f"  ✓ Data shape: {X.shape}")
    print(f"  ✓ Classes: {len(set(y))}")

    # Step 2: Split into train and test
    print("\n🔀 Step 2: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  ✓ Training set: {X_train.shape}")
    print(f"  ✓ Test set: {X_test.shape}")

    # Step 3: Create models
    print("\n🤖 Step 3: Creating models...")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    print(f"  ✓ Models: {list(models.keys())}")

    # Step 4: Evaluate models
    print("\n📈 Step 4: Evaluating models...\n")
    results = evaluate_and_plot_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type="classification",
        verbose=True,
        plot_comparison=True,
    )

    # Step 5: Display results
    print("\n✅ Final Results:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  • {metric:.<25} {value:.4f}")


if __name__ == "__main__":
    main()
