"""
California Housing Regression Example

Demonstrates how to use ML Toolkit for regression on the California Housing dataset.
This is a real-world regression problem predicting median house prices.

Dataset:
- 20,640 samples
- 8 features (longitude, latitude, housing_median_age, total_rooms, etc.)
- Target: median_house_value (in $100,000s)

Run: python example_california_housing.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from my_ml_toolkit import evaluate_and_plot_models


def main():
    print("=" * 70)
    print("🏠 California Housing Regression Example")
    print("=" * 70)

    # ========================================================================
    # 1. LOAD CALIFORNIA HOUSING DATASET
    # ========================================================================
    print("\n📊 Loading California Housing Dataset...")
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="median_house_value")
    
    print(f"  ✓ Dataset shape: {X.shape}")
    print(f"  ✓ Features: {list(X.columns)}")
    print(f"  ✓ Target range: ${y.min():.2f}M - ${y.max():.2f}M (in $100,000s)")
    print(f"  ✓ Target mean: ${y.mean():.2f}M")
    print(f"  ✓ Target median: ${y.median():.2f}M")
    
    # ========================================================================
    # 2. DEFINE REGRESSION MODELS (3 models for speed)
    # ========================================================================
    print("\n🤖 Setting up regression models...")
    
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbosity=0
        )
    }
    
    print(f"  ✓ Models: {list(models.keys())}")
    
    # ========================================================================
    # 3. OPTIONAL: DEFINE PREPROCESSING PIPELINES
    # ========================================================================
    print("\n🔄 Setting up preprocessing...")
    
    # Standard scaling for most models
    standard_preprocessing = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Model-specific preprocessing
    preprocess_pipeline = {
        "Ridge Regression": standard_preprocessing,
        "Random Forest": None,  # No scaling needed for tree models
        "XGBoost": None,        # XGBoost handles scaling internally
        "default": None
    }
    
    print("  ✓ Preprocessing configured (StandardScaler for Ridge)")
    
    # ========================================================================
    # 4. OPTIONAL: DEFINE HYPERPARAMETER GRIDS FOR TUNING
    # ========================================================================
    print("\n🎯 Setting up hyperparameter grids (for future reference)...")
    
    param_grids = {
        "Ridge Regression": {
            'model__alpha': [0.1, 1.0, 10.0]
        },
        "Random Forest": {
            'model__n_estimators': [30, 50],
            'model__max_depth': [8, 10]
        },
        "XGBoost": {
            'model__n_estimators': [30, 50],
            'model__learning_rate': [0.05, 0.1]
        }
    }
    
    print("  ✓ Hyperparameter grids prepared (tuning can be enabled in code)")
    
    # ========================================================================
    # 5. RUN EVALUATION WITH MULTIPLE CONFIGURATIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("📈 Scenario 1: Quick Evaluation (No Tuning)")
    print("=" * 70)
    
    results_quick = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=preprocess_pipeline,
        X=X,
        y=y,
        test_size=0.2,
        val_size=0.0,
        task_type='regression',
        primary_metric='r2',
        cv=2,  # Reduced for speed (default is 4)
        param_grids=None,
        search_type='grid',
        plot_lc=False,  # Disable learning curves for speed
        plot_diagnostics=True,
        plot_importance=True,
        plot_comparison=True,
        show_fold_details=True,
        verbose=True,
        save_dir='./california_housing_results'
    )
    
    # ========================================================================
    # 6. GRID SEARCH TUNING (OPTIONAL - Commented out for speed)
    # ========================================================================
    # Uncomment below to perform hyperparameter tuning (takes ~5-10 minutes)
    
    print("\n" + "=" * 70)
    print("📈 Scenario 2: Grid Search Hyperparameter Tuning (Optional)")
    print("=" * 70)
    print("⏭️  Tuning skipped for demo (would take 5-10 minutes)")
    print("To enable tuning, uncomment the code below in the script")
    print("=" * 70)
    
    """
    results_tuned = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=preprocess_pipeline,
        X=X,
        y=y,
        test_size=0.2,
        val_size=0.1,  # Use validation set for early stopping
        task_type='regression',
        primary_metric='r2',
        param_grids=param_grids,
        search_type='grid',  # Grid search
        cv=5,
        plot_lc=True,
        plot_diagnostics=True,
        plot_importance=True,
        plot_comparison=True,
        show_fold_details=True,
        verbose=True,
        save_dir='./california_housing_results_tuned'
    )
    """
    results_tuned = results_quick  # Use quick results instead
    
    # ========================================================================
    # 7. DISPLAY SUMMARY RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ QUICK EVALUATION COMPLETE")
    print("=" * 70)
    
    if 'summary_df' in results_quick:
        print("\n📊 Model Performance Summary:")
        print("-" * 70)
        print(results_quick['summary_df'].to_string())
    
    print("\n💡 Key Metrics Explained:")
    print("  • R² Score: Proportion of variance explained (0-1, higher is better)")
    print("  • MAE: Average prediction error in $100,000s")
    print("  • RMSE: Root Mean Squared Error (penalizes larger errors)")
    print("  • MAPE: Mean Absolute Percentage Error (relative error %)")
    
    print("\n📁 Results saved to:")
    print("  • ./california_housing_results/")
    
    print("\n🎯 Next Steps:")
    print("  1. Check feature importance plots to understand price drivers")
    print("  2. Review learning curves to detect overfitting")
    print("  3. Compare R² scores across models")
    print("  4. Use top-performing model for predictions")


if __name__ == "__main__":
    main()
