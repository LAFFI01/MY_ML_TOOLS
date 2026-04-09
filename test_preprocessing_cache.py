"""
Test script to demonstrate the preprocessor caching optimization.
Verifies that shared preprocessors are fitted ONCE, not multiple times.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import umap

from my_ml_toolkit.evaluator import evaluate_and_plot_models

# ==============================================================================
# SETUP: Create sample regression data
# ==============================================================================
print("Creating sample regression dataset...")
X, y = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y)

RANDOM_STATE = 42

# ==============================================================================
# SETUP: Define preprocessing pipelines (ONCE)
# ==============================================================================
print("\n📋 Defining preprocessing pipelines...")

num_pipeline_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('umap', umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=RANDOM_STATE))
])

num_pipeline_scaled = Pipeline([
    ('scaler', StandardScaler()),
    ('umap', umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=RANDOM_STATE))
])

print(f"✓ num_pipeline_poly object id: {id(num_pipeline_poly)}")
print(f"✓ num_pipeline_scaled object id: {id(num_pipeline_scaled)}")

# ==============================================================================
# SETUP: Define models
# ==============================================================================
print("\n🤖 Defining models...")
models = {
    'SGDRegressor(SCALED)': SGDRegressor(random_state=42, max_iter=1000),
    'SGDRegressor(POLY)': SGDRegressor(random_state=42, max_iter=1000),
    'RandomForest(SCALED)': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'RandomForest(POLY)': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'XgBoost(SCALED)': XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'XgBoost(POLY)': XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
}

print(f"✓ Defined {len(models)} models")

# ==============================================================================
# SETUP: Map models to preprocessing pipelines
# ==============================================================================
print("\n🔗 Mapping models to preprocessing pipelines...")
preprocess_pipeline = {
    'SGDRegressor(SCALED)': num_pipeline_scaled,
    'SGDRegressor(POLY)': num_pipeline_poly,
    'RandomForest(SCALED)': num_pipeline_scaled,
    'RandomForest(POLY)': num_pipeline_poly,
    'XgBoost(SCALED)': num_pipeline_scaled,
    'XgBoost(POLY)': num_pipeline_poly,
}

print(f"✓ 6 models mapped to 2 unique preprocessor objects")
print(f"  - num_pipeline_scaled: 3 models [SGDRegressor, RandomForest, XgBoost]")
print(f"  - num_pipeline_poly: 3 models [SGDRegressor, RandomForest, XgBoost]")

# ==============================================================================
# RUN: Call evaluate_and_plot_models with preprocessor caching
# ==============================================================================
print("\n" + "=" * 80)
print("🚀 RUNNING evaluate_and_plot_models WITH PREPROCESSOR CACHING")
print("=" * 80)
print("\n⏱️  Watch the output - should see:")
print("   1. 'Building Preprocessor Cache' message")
print("   2. '✓ Cache built: 2 unique preprocessor(s) fitted' ")
print("   3. Models using cached preprocessors (no refitting)")
print("\n")

results = evaluate_and_plot_models(
    models=models,
    preprocess_pipeline=preprocess_pipeline,
    X=X,
    y=y,
    task_type="regression",
    test_size=0.2,
    val_size=0.0,
    cv=3,
    verbose=True,
    plot_lc=False,
    plot_diagnostics=False,
    plot_importance=False,
    plot_comparison=False,
)

# ==============================================================================
# RESULTS
# ==============================================================================
print("\n" + "=" * 80)
print("✅ TEST COMPLETE!")
print("=" * 80)
print("\n📊 Summary Results:")
print(results["summary_df"])
print(f"\n🏆 Best Model: {results['summary_df'].iloc[0]['Model Name']}")
print(f"\n💾 Models counted: {len(results['best_models'])}")
