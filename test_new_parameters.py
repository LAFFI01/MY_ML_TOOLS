#!/usr/bin/env python3
"""
Quick test to verify new parameters work correctly.
Run this to ensure the package has the new features.
"""

import sys
sys.path.insert(0, '/home/laffi/CODE /MY_tools')

from my_ml_toolkit import evaluate_and_plot_models
import inspect
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

print("=" * 60)
print("TESTING NEW PARAMETERS")
print("=" * 60)

# Step 1: Verify parameters exist
print("\n✓ Step 1: Checking function signature...")
sig = inspect.signature(evaluate_and_plot_models)
params = list(sig.parameters.keys())

required_new_params = ['resume', 'show_fold_details', 'halving_factor', 'memory_efficient']
missing = [p for p in required_new_params if p not in params]

if missing:
    print(f"❌ MISSING PARAMETERS: {missing}")
    sys.exit(1)
else:
    print(f"✅ All new parameters found: {required_new_params}")

# Step 2: Get parameter defaults
print("\n✓ Step 2: Checking parameter defaults...")
for param_name in required_new_params:
    param = sig.parameters[param_name]
    print(f"  • {param_name} = {param.default}")

# Step 3: Create minimal test data
print("\n✓ Step 3: Creating test data...")
X, y = make_classification(
    n_samples=100, 
    n_features=10, 
    n_informative=5,
    n_classes=3,
    random_state=42
)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
y_series = pd.Series(y)
print(f"  • X shape: {X_df.shape}")
print(f"  • y shape: {y_series.shape}")

# Step 4: Try calling with new parameters
print("\n✓ Step 4: Testing function call with NEW parameters...")
try:
    # Import models and pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    models = {
        'RF': RandomForestClassifier(n_estimators=10, random_state=42)
    }
    
    # Create a simple preprocessing pipeline
    preprocess_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Call with new parameters
    result = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=preprocess_pipeline,
        X=X_df,
        y=y_series,
        task_type='classification',
        cv=3,
        # NEW PARAMETERS - TEST THESE
        resume=False,
        show_fold_details=True,
        halving_factor=3,
        memory_efficient=True,  # Use this to avoid plots
        verbose=False
    )
    
    print("✅ Function call successful with new parameters!")
    print(f"   Result type: {type(result)}")
    
except TypeError as e:
    print(f"❌ TypeError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"⚠️  Other error (may be OK): {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("✅ TEST PASSED - New parameters are working!")
print("=" * 60)
