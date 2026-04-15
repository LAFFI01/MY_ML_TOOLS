"""
Advanced Optuna Hyperparameter Tuning Example

Demonstrates comprehensive Optuna usage with ALL available parameters:
- optuna_n_trials: Number of optimization trials
- optuna_timeout: Time limit for tuning
- optuna_storage: Persistent storage (SQLite/Redis)
- optuna_study_name: Study naming for resumability
- distributed_tuning: Enable parallel distributed tuning
- search_type: Set to 'optuna'

Run: python example_optuna_advanced.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from my_ml_toolkit import evaluate_and_plot_models


def example_1_regression_basic_optuna():
    """
    Example 1: Basic Optuna Tuning for Regression
    
    Demonstrates straightforward Optuna usage without persistence.
    Best for quick local experiments.
    """
    print("\n" + "=" * 80)
    print("📈 Example 1: Basic Optuna Tuning (Regression)")
    print("=" * 80)

    # Generate regression dataset
    X, y = make_regression(n_samples=300, n_features=15, n_informative=10, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Define models
    models = {
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0)
    }

    # Define hyperparameter grids for Optuna to explore
    param_grids = {
        "Ridge": {
            'model__alpha': [0.001, 1000],  # Optuna will sample between these bounds
        },
        "Random Forest": {
            'model__n_estimators': [10, 200],
            'model__max_depth': [3, 50],
            'model__min_samples_split': [2, 20]
        },
        "XGBoost": {
            'model__n_estimators': [10, 200],
            'model__learning_rate': [0.001, 0.3],
            'model__max_depth': [2, 15]
        }
    }

    # Setup preprocessing
    preprocessing = Pipeline([('scaler', StandardScaler())])
    
    # ✅ BASIC OPTUNA TUNING
    print("\n🎯 Running Optuna Optimization (50 trials)...")
    results = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=preprocessing,
        X=X,
        y=y,
        test_size=0.2,
        task_type='regression',
        primary_metric='r2',
        param_grids=param_grids,
        
        # ===== OPTUNA PARAMETERS =====
        search_type='optuna',              # Use Optuna instead of GridSearch/RandomSearch
        optuna_n_trials=50,                # Run 50 hyperparameter combinations
        optuna_timeout=None,               # No time limit (runs all 50 trials)
        optuna_storage=None,               # In-memory only (no persistence)
        optuna_study_name=None,            # Auto-generated study name
        distributed_tuning=False,          # Single machine (not distributed)
        # ==============================
        
        cv=3,
        plot_comparison=True,
        verbose=True
    )

    print("\n✅ Example 1 Complete!")
    if 'summary_df' in results:
        print("\n📊 Top 3 Models:")
        print(results['summary_df'].head(3).to_string())


def example_2_classification_optuna_timeout():
    """
    Example 2: Optuna with Time Limit
    
    Demonstrates timeout parameter to stop tuning after X seconds.
    Useful when you need results quickly instead of exhaustive search.
    """
    print("\n" + "=" * 80)
    print("🎯 Example 2: Optuna with Timeout (Classification)")
    print("=" * 80)

    # Generate classification dataset
    X, y = make_classification(
        n_samples=400, n_features=20, n_informative=15, 
        n_classes=3, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(y.unique())} classes")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, verbosity=0)
    }

    # Hyperparameter grids
    param_grids = {
        "Logistic Regression": {
            'model__C': [0.001, 100],
            'model__penalty': ['l2']
        },
        "Random Forest": {
            'model__n_estimators': [20, 300],
            'model__max_depth': [3, 30],
            'model__min_samples_split': [2, 10]
        },
        "XGBoost": {
            'model__n_estimators': [20, 300],
            'model__learning_rate': [0.01, 0.5],
            'model__max_depth': [2, 15],
            'model__subsample': [0.5, 1.0]
        }
    }

    preprocessing = Pipeline([('scaler', StandardScaler())])
    
    # ✅ OPTUNA WITH TIMEOUT
    print("\n⏱️  Running Optuna with 30 second timeout (will stop early)...")
    results = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=preprocessing,
        X=X,
        y=y,
        test_size=0.2,
        task_type='classification',
        primary_metric='accuracy',
        param_grids=param_grids,
        
        # ===== OPTUNA PARAMETERS =====
        search_type='optuna',
        optuna_n_trials=1000,              # Request 1000 trials
        optuna_timeout=30,                 # BUT stop after 30 seconds ⏱️
        optuna_storage=None,               # In-memory
        optuna_study_name=None,
        distributed_tuning=False,
        # ==============================
        
        cv=3,
        plot_comparison=True,
        verbose=True
    )

    print("\n✅ Example 2 Complete!")
    if 'summary_df' in results:
        print("\n📊 Top 3 Models:")
        print(results['summary_df'].head(3).to_string())


def example_3_optuna_persistent_sqlite():
    """
    Example 3: Persistent Optuna Storage with SQLite
    
    Demonstrates persistent storage so you can:
    - Resume interrupted tuning
    - Compare studies across runs
    - Run parallel experiments (multiple notebooks share same DB)
    
    Perfect for Kaggle notebooks!
    """
    print("\n" + "=" * 80)
    print("💾 Example 3: Persistent Optuna with SQLite")
    print("=" * 80)

    # Generate dataset
    X, y = make_regression(n_samples=400, n_features=20, n_informative=15, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print("💾 Persistent storage: ./optuna_studies/my_study.db")

    # Define models
    models = {
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
    }

    # Hyperparameter grids
    param_grids = {
        "Ridge": {
            'model__alpha': [0.001, 1000],
        },
        "Random Forest": {
            'model__n_estimators': [20, 150],
            'model__max_depth': [3, 30],
            'model__min_samples_split': [2, 15]
        },
        "XGBoost": {
            'model__n_estimators': [20, 150],
            'model__learning_rate': [0.01, 0.3],
            'model__max_depth': [2, 12]
        }
    }

    preprocessing = Pipeline([('scaler', StandardScaler())])
    
    # ✅ OPTUNA WITH PERSISTENT SQLITE STORAGE
    print("\n📝 Setup:")
    print("  • Storage: SQLite database (persistent)")
    print("  • Study Name: 'regression_study_v1'")
    print("  • Trials: 40 (this run)")
    print("  • If study exists: Resume from last trial")
    
    results = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=preprocessing,
        X=X,
        y=y,
        test_size=0.2,
        task_type='regression',
        primary_metric='r2',
        param_grids=param_grids,
        
        # ===== OPTUNA PARAMETERS WITH PERSISTENCE =====
        search_type='optuna',
        optuna_n_trials=40,
        optuna_timeout=None,
        optuna_storage='sqlite:///./optuna_studies/my_study.db',  # 💾 Persistent!
        optuna_study_name='regression_study_v1',                  # Named study for resumability
        distributed_tuning=False,
        # =============================================
        
        cv=3,
        plot_comparison=True,
        verbose=True,
        save_dir='./optuna_regression_results'
    )

    print("\n✅ Example 3 Complete!")
    print("💡 To resume this study later, use the SAME storage_url and study_name")


def example_4_optuna_distributed_kaggle():
    """
    Example 4: Distributed Optuna for Parallel Kaggle Notebooks
    
    THREE parallel notebooks use the SAME SQLite database:
    - Each notebook runs 30 trials simultaneously
    - All notebooks share trial history (no duplicate work)
    - Result: 3x speedup! 🚀
    
    Copy this code to 3 Kaggle notebooks and run them all at once.
    """
    print("\n" + "=" * 80)
    print("🚀 Example 4: Distributed Optuna (Parallel Notebooks)")
    print("=" * 80)

    # Generate dataset
    X, y = make_regression(n_samples=500, n_features=25, n_informative=20, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print("\n📋 SETUP FOR 3 PARALLEL NOTEBOOKS:")
    print("  Copy this ENTIRE block to 3 different Kaggle notebooks")
    print("  Run all 3 notebooks simultaneously")
    print("  Each searches 30 trials in parallel → Done in 1/3 time! ✨")

    # Define models
    models = {
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=1),  # n_jobs=1 to avoid thread contention
        "XGBoost": XGBRegressor(random_state=42, verbosity=0, n_jobs=1)
    }

    # Hyperparameter grids
    param_grids = {
        "Ridge": {
            'model__alpha': [0.001, 1000],
        },
        "Random Forest": {
            'model__n_estimators': [20, 300],
            'model__max_depth': [3, 50],
            'model__min_samples_split': [2, 20],
            'model__max_features': ['sqrt', 'log2']
        },
        "XGBoost": {
            'model__n_estimators': [20, 300],
            'model__learning_rate': [0.001, 0.5],
            'model__max_depth': [2, 20],
            'model__subsample': [0.5, 1.0],
            'model__colsample_bytree': [0.3, 1.0]
        }
    }

    preprocessing = Pipeline([('scaler', StandardScaler())])
    
    # ✅ OPTUNA WITH DISTRIBUTED PARALLEL TUNING
    print("\n🔗 Running distributed Optuna (30 trials per notebook, shared study)...")
    results = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=preprocessing,
        X=X,
        y=y,
        test_size=0.2,
        task_type='regression',
        primary_metric='r2',
        param_grids=param_grids,
        
        # ===== OPTUNA DISTRIBUTED PARAMETERS =====
        search_type='optuna',
        optuna_n_trials=30,                                           # 30 per notebook
        optuna_timeout=300,                                           # 5 min timeout per notebook
        optuna_storage='sqlite:///./kaggle_optuna_shared.db',        # 💾 SHARED DATABASE
        optuna_study_name='kaggle_distributed_tuning',               # 🔗 SHARED STUDY NAME
        distributed_tuning=True,                                     # Enable distributed mode
        # ========================================
        
        cv=3,
        plot_comparison=True,
        verbose=True,
        save_dir='./kaggle_optuna_results'
    )

    print("\n✅ Notebook completed its 30 trials")
    print("💡 Simultaneously running 3 notebooks will find optimal params 3x faster!")


def example_5_optuna_all_parameters():
    """
    Example 5: COMPLETE Example with ALL Optuna Parameters
    
    Shows every single parameter at once for reference.
    """
    print("\n" + "=" * 80)
    print("🎓 Example 5: ALL Optuna Parameters Reference")
    print("=" * 80)

    # Generate smaller dataset for demo
    X, y = make_regression(n_samples=200, n_features=12, n_informative=10, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Define models
    models = {
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0)
    }

    # Hyperparameter grids
    param_grids = {
        "Ridge": {
            'model__alpha': [0.001, 1000],
        },
        "Random Forest": {
            'model__n_estimators': [20, 200],
            'model__max_depth': [3, 30],
            'model__min_samples_split': [2, 15]
        },
        "XGBoost": {
            'model__n_estimators': [20, 200],
            'model__learning_rate': [0.01, 0.3],
            'model__max_depth': [2, 15],
            'model__subsample': [0.6, 1.0]
        }
    }

    preprocessing = Pipeline([('scaler', StandardScaler())])
    
    print("\n" + "=" * 80)
    print("📋 ALL OPTUNA PARAMETERS BREAKDOWN:")
    print("=" * 80)
    
    # ✅ ALL PARAMETERS
    results = evaluate_and_plot_models(
        # Dataset
        models=models,
        preprocess_pipeline=preprocessing,
        X=X,
        y=y,
        test_size=0.2,
        val_size=0.1,
        
        # Task
        task_type='regression',
        primary_metric='r2',
        
        # Hyperparameter tuning
        param_grids=param_grids,
        
        # ===== OPTUNA-SPECIFIC PARAMETERS =====
        search_type='optuna',                      # ✅ Use Optuna optimization
        optuna_n_trials=20,                        # ✅ Run 20 trials (hyperparameter combinations)
        optuna_timeout=60,                         # ✅ Stop after 60 seconds (if not finished)
        optuna_storage='sqlite:///./demo_optuna.db',  # ✅ Save to SQLite (persistent)
        optuna_study_name='demo_study_all_params',    # ✅ Named study for resumability
        distributed_tuning=False,                  # ✅ Single machine tuning (set True for parallel)
        # =====================================
        
        # Cross-validation
        cv=3,
        
        # Visualizations
        plot_lc=True,
        plot_diagnostics=True,
        plot_importance=True,
        plot_comparison=True,
        
        # Other
        build_ensemble=True,
        verbose=True,
        save_dir='./optuna_all_params_results'
    )

    print("\n✅ Example 5 Complete!")
    print("\n📚 PARAMETER REFERENCE:")
    print("""
    search_type='optuna'              → Use Optuna instead of GridSearch/Random
    optuna_n_trials=20                → Try 20 combinations
    optuna_timeout=60                 → Stop after 60 seconds
    optuna_storage='sqlite:///...'    → Save results to database
    optuna_study_name='study_name'    → Name for resuming studies
    distributed_tuning=False          → Set True for parallel notebooks
    """)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 OPTUNA COMPREHENSIVE EXAMPLES - ALL PARAMETERS")
    print("=" * 80)
    
    # Run examples
    example_1_regression_basic_optuna()
    example_2_classification_optuna_timeout()
    example_3_optuna_persistent_sqlite()
    example_4_optuna_distributed_kaggle()
    example_5_optuna_all_parameters()
    
    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\n📚 SUMMARY:")
    print("  1. Basic Optuna (in-memory)")
    print("  2. Timeout-based tuning")
    print("  3. Persistent SQLite storage")
    print("  4. Distributed parallel tuning (Kaggle)")
    print("  5. All parameters reference")
    print("\n💡 Choose the example that matches your use case!")
