"""
Example 4: Real Dataset (Iris)

Example using the famous Iris dataset for classification.
Run: python example4_real_dataset.py
"""

from my_ml_toolkit import evaluate_and_plot_models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd


def main():
    print("=" * 60)
    print("Example 4: Real Dataset (Iris)")
    print("=" * 60)

    # Load Iris dataset
    print("\n📊 Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Create DataFrame for better visualization
    feature_names = iris.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    print(f"  ✓ Dataset shape: {X.shape}")
    print(f"  ✓ Features: {feature_names}")
    print(f"  ✓ Classes: {iris.target_names}")
    print(f"  ✓ Data preview:")
    print(df.head())

    # Split data
    print("\n🔀 Splitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Training samples: {len(X_train)}")
    print(f"  ✓ Test samples: {len(X_test)}")

    # Scale features (important for some algorithms)
    print("\n🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  ✓ Features scaled using StandardScaler")

    # Define multiple models
    print("\n🤖 Creating multiple models...")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }
    print(f"  ✓ Models: {list(models.keys())}")

    # Evaluate models
    print("\n📈 Evaluating models...\n")
    results = evaluate_and_plot_models(
        models=models,
        X_train=X_train_scaled,  # Use scaled features
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        task_type="classification",
        primary_metric="accuracy",
        cv_folds=5,
        verbose=True,
        plot_comparison=True,
        plot_confusion_matrix=True,
    )

    # Display results
    print("\n✅ Iris Dataset Results:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  • {metric:.<25} {value:.4f}")

    # Get best model
    best_model = max(results, key=lambda x: results[x]["accuracy"])
    best_accuracy = results[best_model]["accuracy"]

    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")


if __name__ == "__main__":
    main()
