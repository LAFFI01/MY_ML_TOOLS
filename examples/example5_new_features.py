"""
Example 5: Using New Features - HalvingGridSearchCV, Comprehensive Metrics & Checkpoint/Resume

This example demonstrates:
1. HalvingGridSearchCV for efficient hyperparameter tuning
2. Comprehensive classification metrics (12 metrics + ROC/PR curves)
3. Comprehensive regression metrics (7 metrics + 4 diagnostic plots)
4. Checkpoint and resume functionality
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from my_ml_toolkit import evaluate_and_plot_models


# ============================================================================
# EXAMPLE 1: Classification with HalvingGridSearchCV and All Metrics
# ============================================================================
def example_classification_halving():
    """
    Comprehensive classification example using HalvingGridSearchCV.
    
    Features demonstrated:
    - HalvingGridSearchCV (efficient search)
    - 12 classification metrics
    - ROC and PR curves (binary & multi-class)
    - Formatted metric output
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Classification with HalvingGridSearchCV")
    print("="*80)
    
    # Generate multi-class classification data
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    # Define preprocessing
    preprocess = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    # Define parameter grids
    param_grids = {
        "LogisticRegression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l2"]
        },
        "RandomForest": {
            "model__max_depth": [3, 5, 7, 9],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        },
        "GradientBoosting": {
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
            "model__n_estimators": [50, 100]
        }
    }
    
    # Run evaluation with HalvingGridSearchCV
    results = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=preprocess,
        X=X_train,
        y=y_train,
        param_grids=param_grids,
        task_type="classification",
        target_names=["Class_0", "Class_1", "Class_2"],
        test_size=0.2,
        cv=5,
        search_type="halving",  # ← HalvingGridSearchCV
        primary_metric="f1_macro",
        plot_lc=True,
        plot_diagnostics=True,  # Confusion matrices + ROC/PR curves
        plot_importance=False,
        plot_comparison=True,
        verbose=True
    )
    
    # Display results
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    print("\nModel Performance Summary:")
    print(results["summary_df"])
    print("\nBest Model:", results["ultimate_winner"])
    
    return results


# ============================================================================
# EXAMPLE 2: Regression with All Diagnostic Metrics
# ============================================================================
def example_regression_diagnostics():
    """
    Comprehensive regression example with all metrics and diagnostics.
    
    Features demonstrated:
    - 7 regression metrics (MAE, MSE, RMSE, MAPE, R², Adj-R², RMSLE)
    - 4 diagnostic plots (residuals, distribution, Q-Q, scale-location)
    - Parameter tuning with halving search
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Regression with Comprehensive Diagnostics")
    print("="*80)
    
    # Generate regression data
    X, y = make_regression(
        n_samples=300,
        n_features=15,
        n_informative=10,
        noise=10,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"X_{i}" for i in range(15)])
    y = pd.Series(y)
    
    # Define models
    models = {
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    # Preprocessing
    preprocess = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    # Parameter grids
    param_grids = {
        "Ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        "Lasso": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0]
        },
        "RandomForest": {
            "model__max_depth": [5, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    }
    
    # Run with diagnostics
    results = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=preprocess,
        X=X,
        y=y,
        param_grids=param_grids,
        task_type="regression",
        test_size=0.2,
        cv=5,
        search_type="halving",  # ← Efficient for large parameter spaces
        primary_metric="r2",
        plot_lc=True,
        plot_diagnostics=True,  # Shows 4 diagnostic plots
        plot_importance=True,
        plot_comparison=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("REGRESSION RESULTS")
    print("="*80)
    print("\nModel Performance Summary:")
    print(results["summary_df"])
    
    return results


# ============================================================================
# EXAMPLE 3: Checkpoint and Resume
# ============================================================================
def example_checkpoint_resume():
    """
    Demonstrate checkpoint and resume functionality.
    
    Features demonstrated:
    - Saving model checkpoints
    - Resuming from checkpoint
    - Fault tolerance
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Checkpoint and Resume")
    print("="*80)
    
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(20)])
    y = pd.Series(y)
    
    models = {
        "Model_1": LogisticRegression(random_state=42),
        "Model_2": RandomForestClassifier(random_state=42),
        "Model_3": GradientBoostingClassifier(random_state=42)
    }
    
    param_grids = {
        "Model_1": {"model__C": [0.1, 1.0, 10.0]},
        "Model_2": {"model__max_depth": [5, 10, 15]},
        "Model_3": {"model__learning_rate": [0.01, 0.1]}
    }
    
    # First run - will create checkpoint
    print("\n▶️ FIRST RUN - Create checkpoint")
    results_1 = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=None,
        X=X,
        y=y,
        param_grids=param_grids,
        task_type="classification",
        save_dir="./checkpoint_example",  # ← Enable checkpointing
        resume=False,                      # ← Fresh start
        cv=3,
        verbose=True
    )
    
    # Resume run - will continue from checkpoint
    print("\n⏸️ RESUME - Continue from checkpoint")
    results_2 = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=None,
        X=X,
        y=y,
        param_grids=param_grids,
        task_type="classification",
        save_dir="./checkpoint_example",  # ← Same directory
        resume=True,                       # ← Resume from checkpoint
        cv=3,
        verbose=True
    )
    
    print("\n✅ Checkpoint/Resume Example Complete")
    print("Files saved to: ./checkpoint_example/")
    
    return results_1, results_2


# ============================================================================
# EXAMPLE 4: Time Series with Sequential Split
# ============================================================================
def example_time_series():
    """
    Time series regression with sequential split (no shuffling).
    
    Features demonstrated:
    - Sequential split method for time series
    - Regression metrics
    - Proper temporal ordering
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Time Series with Sequential Split")
    print("="*80)
    
    # Create time series data
    n_samples = 200
    trend = np.arange(n_samples) * 0.5
    noise = np.random.randn(n_samples) * 2
    y = trend + noise
    
    X = pd.DataFrame({
        "lag_1": np.roll(y, 1),
        "lag_2": np.roll(y, 2),
        "lag_3": np.roll(y, 3),
        "trend": np.arange(n_samples)
    })
    y = pd.Series(y)
    
    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=42)
    }
    
    # Sequential split preserves temporal order
    results = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=None,
        X=X,
        y=y,
        task_type="regression",
        test_size=0.2,
        split_method="sequential",  # ← No shuffling
        cv=3,
        plot_diagnostics=True,
        verbose=True
    )
    
    # Verify temporal ordering
    train_indices = results["data_splits"]["X_train"].index
    test_indices = results["data_splits"]["X_test"].index
    
    print(f"\nTraining samples: indices {train_indices.min()} to {train_indices.max()}")
    print(f"Test samples: indices {test_indices.min()} to {test_indices.max()}")
    print("✅ Sequential ordering preserved!")
    
    return results


# ============================================================================
# EXAMPLE 5: Comparing Search Methods
# ============================================================================
def example_compare_search_methods():
    """
    Compare performance and speed of different search methods.
    
    Features demonstrated:
    - Grid search (exhaustive)
    - Random search (sampling)
    - Halving search (efficient)
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Comparing Search Methods")
    print("="*80)
    
    X, y = make_classification(n_samples=300, n_features=10, n_classes=2, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    models = {"LogisticRegression": LogisticRegression(max_iter=500, random_state=42)}
    
    param_grids = {
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__penalty": ["l2"],
            "model__max_iter": [500]
        }
    }
    
    import time
    
    for search_method in ["grid", "random", "halving"]:
        print(f"\n▶️ Using {search_method} search...")
        start_time = time.time()
        
        results = evaluate_and_plot_models(
            models=models,
            preprocess_pipeline=None,
            X=X,
            y=y,
            param_grids=param_grids,
            search_type=search_method,
            cv=3,
            verbose=False
        )
        
        elapsed = time.time() - start_time
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Best Score: {results['summary_df'].iloc[0]['CV Mean (accuracy)']}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Run all examples."""
    
    print("\n" + "="*80)
    print("NEW FEATURES EXAMPLES - ML Toolkit")
    print("="*80)
    print("\n📚 Running Examples...")
    
    # Run examples
    example_classification_halving()
    example_regression_diagnostics()
    example_checkpoint_resume()
    example_time_series()
    example_compare_search_methods()
    
    print("\n" + "="*80)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\n📖 For more information, see FEATURE_USAGE_GUIDE.md")


if __name__ == "__main__":
    main()
