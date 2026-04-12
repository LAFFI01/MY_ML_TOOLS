"""
COMPLETE REFERENCE: All evaluate_and_plot_models() Parameters

Updated with all new features:
- resume: Checkpoint resumption
- show_fold_details: Per-fold metrics display
- halving_factor: HalvingGridSearchCV control
- memory_efficient: Memory optimization mode
"""

from my_ml_toolkit import evaluate_and_plot_models

# ============================================================================
# EXAMPLE 1: REGRESSION - Basic Setup (No Tuning)
# ============================================================================

results_no_tuning = evaluate_and_plot_models(
    # ---------------------------------------------------------
    # 1. CORE COMPONENTS
    # ---------------------------------------------------------
    models = models,
    # The dictionary or single pipeline dictating how data is scaled/encoded
    preprocess_pipeline = MY_PIPELINES,  # Can be single pipeline or dict per model
    
    # Passing raw X and y tells the library to handle splitting internally
    X = X, 
    y = Y, 
    test_size = 0.2,          # Reserves 20% of data for final test score
    val_size = 0.0,           # 0.0 = no validation set (useful for XGBoost early stopping with val_size > 0)
    split_method = 'random',  # 'random' (default) or 'sequential' (time-series)
    stratify = False,         # No stratification for regression
    random_seed = RANDOM_STATE,  # Master seed for reproducibility
    
    # ---------------------------------------------------------
    # 2. TASK & METADATA
    # ---------------------------------------------------------
    task_type = 'regression',            # Triggers regression metrics (MAE, RMSE, R²)
    target_names = ['price'],            # Label for target variable
    feature_names = X.columns.tolist(),  # Feature names for importance charts
    
    # ---------------------------------------------------------
    # 3. TUNING & SAMPLING (Disabled in this example)
    # ---------------------------------------------------------
    param_grids = None,        # No hyperparameter tuning
    fit_params = None,         # No special .fit() parameters
    sampler = None,            # No sampling (only for imbalanced classification)
    cv = 5,                    # 5-fold cross-validation on training data
    search_type = 'grid',      # 'grid', 'random', or 'halving' (ignored without param_grids)
    primary_metric = 'r2',     # Metric to optimize (auto-selected for regression)
    
    # ---------------------------------------------------------
    # 4. PHASE 1 SCREENING (Optional Quick Filter)
    # ---------------------------------------------------------
    top_k = None,              # None = disable Phase 1, evaluate all models
                               # Set to 3-5 to quickly filter before expensive tuning
    quick_test_fraction = 0.2, # Fraction of data used in Phase 1 screening
    
    # ---------------------------------------------------------
    # 5. VISUALIZATION & OUTPUT
    # ---------------------------------------------------------
    plot_lc = True,            # Learning curves (shows overfitting)
    plot_diagnostics = True,   # Residual plots for regression / confusion matrix for classification
    plot_importance = True,    # Feature importance bar chart
    plot_comparison = True,    # Boxplot comparing all models
    top_n_features = 4,        # Limit feature importance chart to top 4 features
    save_dir = 'model_outputs', # Save trained models here
    
    # ---------------------------------------------------------
    # 6. STATE & LOGGING (NEW FEATURES)
    # ---------------------------------------------------------
    resume = False,            # NEW: Resume from checkpoint if interrupted
                               #   False = start fresh
                               #   True = load from save_dir/eval_checkpoint.pkl
    show_fold_details = True,  # NEW: Display per-fold CV scores
                               #   True = show Fold 1, Fold 2, ... Fold N scores
                               #   False = show only mean ± std
    
    # ---------------------------------------------------------
    # 7. OPTIMIZATION (NEW FEATURES)
    # ---------------------------------------------------------
    halving_factor = 3,        # NEW: HalvingGridSearchCV elimination rate
                               #   2 = fastest (50% elimination) - use for huge param spaces
                               #   3 = balanced (67% elimination) - DEFAULT, recommended
                               #   4+ = thorough (75%+ elimination) - slower but more thorough
                               #   Only used when search_type='halving'
    memory_efficient = False,  # NEW: Enable memory efficiency for large datasets
                               #   False = generate all plots (default)
                               #   True = disable learning curves, importance, diagnostics
                               #        = reduces RAM by 60-70%, training 2-3x faster
    
    # ---------------------------------------------------------
    # 8. LOGGING
    # ---------------------------------------------------------
    verbose = True             # Print progress to terminal
)

print(results_no_tuning['summary_df'])


# ============================================================================
# EXAMPLE 2: CLASSIFICATION - With Hyperparameter Tuning & Memory Efficiency
# ============================================================================

results_with_tuning = evaluate_and_plot_models(
    # ---------------------------------------------------------
    # 1. CORE COMPONENTS
    # ---------------------------------------------------------
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    },
    preprocess_pipeline = Pipeline([("scaler", StandardScaler())]),
    
    X = X_train,
    y = y_train,
    test_size = 0.2,
    val_size = 0.1,           # Reserve 10% for validation (useful for XGBoost early stopping)
    split_method = 'random',
    stratify = True,          # Stratify for classification (maintains class ratios)
    random_seed = 42,
    
    # ---------------------------------------------------------
    # 2. TASK & METADATA
    # ---------------------------------------------------------
    task_type = 'classification',
    target_names = ['negative', 'positive'],  # Binary classification
    feature_names = X_train.columns.tolist(),
    
    # ---------------------------------------------------------
    # 3. TUNING & SAMPLING (ENABLED)
    # ---------------------------------------------------------
    param_grids = {
        "LogisticRegression": {
            "model__C": [0.001, 0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "sag"],
        },
        "RandomForest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10],
        },
        "GradientBoosting": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 4, 5],
        },
    },
    
    fit_params = {
        "GradientBoosting": {
            "model__eval_set": [(X_val, y_val)],  # Early stopping
        }
    },
    
    sampler = SMOTE(random_state=42),  # Handle class imbalance
    cv = 5,
    search_type = 'halving',           # Use efficient halving search
    primary_metric = 'balanced_accuracy',  # NEW: Use balanced accuracy for imbalanced data
    
    # ---------------------------------------------------------
    # 4. PHASE 1 SCREENING (Quick Filter)
    # ---------------------------------------------------------
    top_k = 2,                 # Quickly filter to top 2 models before full tuning
    quick_test_fraction = 0.3, # Use 30% of data for Phase 1
    
    # ---------------------------------------------------------
    # 5. VISUALIZATION & OUTPUT
    # ---------------------------------------------------------
    plot_lc = True,
    plot_diagnostics = True,   # Confusion matrices for classification
    plot_importance = True,
    plot_comparison = True,
    top_n_features = 10,
    save_dir = 'best_models',  # Save all models and checkpoint
    
    # ---------------------------------------------------------
    # 6. STATE & LOGGING (NEW)
    # ---------------------------------------------------------
    resume = True,             # Resume from checkpoint if exists
    show_fold_details = True,  # Show per-fold balanced accuracy
    
    # ---------------------------------------------------------
    # 7. OPTIMIZATION (NEW)
    # ---------------------------------------------------------
    halving_factor = 3,        # Balanced halving (recommended)
    memory_efficient = False,  # Use all visualizations (balanced speed/memory)
    
    verbose = True,
)

print(results_with_tuning['summary_df'])


# ============================================================================
# EXAMPLE 3: LARGE DATASET - Memory-Efficient Mode
# ============================================================================

results_memory_efficient = evaluate_and_plot_models(
    # ---------------------------------------------------------
    # 1. CORE COMPONENTS
    # ---------------------------------------------------------
    models = {
        "RandomForest": RandomForestClassifier(n_jobs=-1, random_state=42),
        "XGBClassifier": XGBClassifier(tree_method='hist', random_state=42),
    },
    preprocess_pipeline = Pipeline([("scaler", StandardScaler())]),
    
    X = X_large,    # 10+ million rows
    y = y_large,
    test_size = 0.2,
    val_size = 0.0,
    split_method = 'random',
    stratify = True,
    random_seed = 42,
    
    # ---------------------------------------------------------
    # 2. TASK & METADATA
    # ---------------------------------------------------------
    task_type = 'classification',
    target_names = None,  # Let toolkit auto-detect
    feature_names = X_large.columns.tolist(),
    
    # ---------------------------------------------------------
    # 3. TUNING & SAMPLING
    # ---------------------------------------------------------
    param_grids = {
        "RandomForest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 15],
        },
        "XGBClassifier": {
            "model__max_depth": [5, 7],
            "model__learning_rate": [0.1, 0.2],
        },
    },
    
    fit_params = None,
    sampler = None,
    cv = 3,                # Fewer folds for speed
    search_type = 'halving',
    primary_metric = 'balanced_accuracy',
    
    # ---------------------------------------------------------
    # 4. PHASE 1 (Disabled for memory efficiency)
    # ---------------------------------------------------------
    top_k = None,
    quick_test_fraction = 0.2,
    
    # ---------------------------------------------------------
    # 5. VISUALIZATION & OUTPUT (All disabled by memory_efficient)
    # ---------------------------------------------------------
    plot_lc = True,          # Will be disabled by memory_efficient=True
    plot_diagnostics = True, # Will be disabled by memory_efficient=True
    plot_importance = True,  # Will be disabled by memory_efficient=True
    plot_comparison = True,  # Will be disabled by memory_efficient=True
    top_n_features = 10,
    save_dir = 'large_models',
    
    # ---------------------------------------------------------
    # 6. STATE & LOGGING (NEW)
    # ---------------------------------------------------------
    resume = False,         # Start fresh
    show_fold_details = True,  # Show fold stability even with memory efficiency
    
    # ---------------------------------------------------------
    # 7. OPTIMIZATION (NEW) ⚡⚡⚡
    # ---------------------------------------------------------
    halving_factor = 2,        # AGGRESSIVE: 50% elimination per round
                               # For large datasets, speed is critical
    memory_efficient = True,   # 🚀 ENABLE: Disables all plots
                               # Terminal output: ⚡ MEMORY-EFFICIENT MODE ENABLED
                               # Expected: 60-70% less RAM, 2-3x faster training
    
    verbose = True,
)

# You still get full metrics despite memory efficiency!
print(results_memory_efficient['summary_df'])
print(results_memory_efficient['ultimate_winner'])


# ============================================================================
# EXAMPLE 4: RESUMABLE TRAINING - Checkpoint Support
# ============================================================================

# First run - might be interrupted
results_run1 = evaluate_and_plot_models(
    models = models,
    preprocess_pipeline = preprocess,
    X = X,
    y = y,
    task_type = 'classification',
    cv = 5,
    search_type = 'grid',
    param_grids = param_grids,          # Complex tuning, might take hours
    
    save_dir = 'checkpoints',           # Save checkpoint after each model
    resume = False,                     # Start fresh
    show_fold_details = True,
    halving_factor = 3,
    memory_efficient = False,
    
    verbose = True,
)

# ✅ If interrupted, second run picks up where it left off!
results_run2 = evaluate_and_plot_models(
    models = models,
    preprocess_pipeline = preprocess,
    X = X,
    y = y,
    task_type = 'classification',
    cv = 5,
    search_type = 'grid',
    param_grids = param_grids,
    
    save_dir = 'checkpoints',           # Same directory
    resume = True,                      # 🔄 RESUME from checkpoint!
                                        # Terminal: "⏭️ Skipping [RandomForest] (Already completed...)"
    show_fold_details = True,
    halving_factor = 3,
    memory_efficient = False,
    
    verbose = True,
)


# ============================================================================
# PARAMETER REFERENCE TABLE
# ============================================================================

"""
┌─────────────────────────────────────┬──────────────┬────────────┬──────────────────────┐
│ Parameter                           │ Type         │ Default    │ Key Values           │
├─────────────────────────────────────┼──────────────┼────────────┼──────────────────────┤
│ CORE DATA                                                                              │
│ ├─ models                           │ Dict         │ REQUIRED   │ Model instances      │
│ ├─ preprocess_pipeline              │ Pipeline/Dict│ REQUIRED   │ Scaler, encoder      │
│ ├─ X                                │ DataFrame    │ REQUIRED   │ Features (n×m)       │
│ ├─ y                                │ Series       │ REQUIRED   │ Target (n,)          │
│ ├─ test_size                        │ float        │ 0.2        │ [0.0, 1.0)           │
│ ├─ val_size                         │ float        │ 0.0        │ [0.0, 1.0)           │
│ ├─ split_method                     │ str          │ 'random'   │ 'random','sequential'│
│ ├─ stratify                         │ bool         │ True       │ True/False           │
│ └─ random_seed                      │ int          │ 42         │ Any positive int     │
│                                                                                         │
│ TASK & METADATA                                                                        │
│ ├─ task_type                        │ str          │ N/A        │ 'classification'     │
│ │                                   │              │            │ 'regression'         │
│ ├─ target_names                     │ List[str]    │ None       │ Class/target names   │
│ └─ feature_names                    │ List[str]    │ None       │ Column names         │
│                                                                                         │
│ TUNING & CV                                                                            │
│ ├─ param_grids                      │ Dict/None    │ None       │ Parameter ranges     │
│ ├─ fit_params                       │ Dict/None    │ None       │ Custom .fit() kwargs │
│ ├─ sampler                          │ Sampler/None │ None       │ SMOTE, RandomOver... │
│ ├─ cv                               │ int          │ 4          │ [2, 3, 5, 10]        │
│ ├─ search_type                      │ str          │ 'grid'     │ 'grid'               │
│ │                                   │              │            │ 'random'             │
│ │                                   │              │            │ 'halving'            │
│ └─ primary_metric                   │ str          │ Auto       │ 'accuracy'           │
│                                     │              │            │ 'f1'                 │
│                                     │              │            │ 'balanced_accuracy'  │
│                                     │              │            │ 'r2', 'mae', 'rmse' │
│                                                                                         │
│ PHASE 1 SCREENING                                                                      │
│ ├─ top_k                            │ int/None     │ None       │ 2, 3, 5 (or None)    │
│ └─ quick_test_fraction              │ float        │ 0.2        │ [0.1, 0.5]           │
│                                                                                         │
│ VISUALIZATION                                                                          │
│ ├─ plot_lc                          │ bool         │ True       │ True/False           │
│ ├─ plot_diagnostics                 │ bool         │ True       │ True/False           │
│ ├─ plot_importance                  │ bool         │ True       │ True/False           │
│ ├─ plot_comparison                  │ bool         │ True       │ True/False           │
│ ├─ top_n_features                   │ int          │ 20         │ [5, 10, 20, 30]      │
│ └─ save_dir                         │ str/None     │ None       │ 'path/to/dir'        │
│                                                                                         │
│ STATE & LOGGING (NEW)                                                                  │
│ ├─ resume                           │ bool         │ False      │ True/False           │
│ └─ show_fold_details                │ bool         │ True       │ True/False           │
│                                                                                         │
│ OPTIMIZATION (NEW)                                                                     │
│ ├─ halving_factor                   │ int          │ 3          │ 2, 3, 4, 5           │
│ ├─ memory_efficient                 │ bool         │ False      │ True/False           │
│                                                                                         │
│ LOGGING                                                                                │
│ └─ verbose                          │ bool         │ True       │ True/False           │
└─────────────────────────────────────┴──────────────┴────────────┴──────────────────────┘
"""

