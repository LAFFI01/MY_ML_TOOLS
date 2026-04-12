"""
QUICKSTART: Most Important Parameters for Common Use Cases

Copy-paste these templates to get started quickly.
"""

from my_ml_toolkit import evaluate_and_plot_models

# ============================================================================
# QUICK TEMPLATE 1: Basic Classification (Recommended Starting Point)
# ============================================================================

results = evaluate_and_plot_models(
    models = models,
    preprocess_pipeline = preprocess,
    X = X,
    y = y,
    task_type = 'classification',
    cv = 5,                        # 5-fold cross-validation
    search_type = 'halving',       # Fast & efficient
    halving_factor = 3,            # Balanced (DEFAULT)
    primary_metric = 'balanced_accuracy',  # Good for imbalanced data
    show_fold_details = True,      # See per-fold stability
    save_dir = 'models',           # Save trained models
    verbose = True,
)


# ============================================================================
# QUICK TEMPLATE 2: Large Dataset (Memory Efficient)
# ============================================================================

results = evaluate_and_plot_models(
    models = models,
    preprocess_pipeline = preprocess,
    X = X_large,   # 1M+ rows
    y = y_large,
    task_type = 'classification',
    cv = 3,                        # Fewer folds = faster
    search_type = 'halving',
    halving_factor = 2,            # FAST (aggressive)
    memory_efficient = True,       # ⚡ 60-70% less RAM, 2-3x faster
    save_dir = 'models',
    verbose = True,
)


# ============================================================================
# QUICK TEMPLATE 3: Hyperparameter Tuning (Thorough)
# ============================================================================

results = evaluate_and_plot_models(
    models = models,
    preprocess_pipeline = preprocess,
    X = X,
    y = y,
    task_type = 'classification',
    param_grids = param_grids,    # Enable tuning
    cv = 5,
    search_type = 'grid',         # Exhaustive (slower but thorough)
    primary_metric = 'f1',        # Your metric
    show_fold_details = True,
    save_dir = 'tuned_models',
    verbose = True,
)


# ============================================================================
# QUICK TEMPLATE 4: Resumable Training (Long-Running)
# ============================================================================

results = evaluate_and_plot_models(
    models = models,
    preprocess_pipeline = preprocess,
    X = X,
    y = y,
    task_type = 'classification',
    param_grids = param_grids,    # Complex tuning
    cv = 5,
    search_type = 'halving',
    save_dir = 'checkpoints',
    resume = True,               # NEW: Resume if interrupted!
    show_fold_details = True,
    verbose = True,
)


# ============================================================================
# NEW PARAMETERS QUICK REFERENCE
# ============================================================================

"""
NEW PARAMETERS (Added in Latest Update):

1. resume (bool, default=False)
   ├─ False: Start fresh training
   └─ True:  Resume from checkpoint (if save_dir contains eval_checkpoint.pkl)
   
   Example:
   resume = True  # Picks up from last completed model

2. show_fold_details (bool, default=True)
   ├─ True:  Display Fold 1, Fold 2, ..., Fold N scores + stats
   └─ False: Show only Mean ± Std
   
   Output when True:
   📊 Fold-Level balanced_accuracy Scores for [RandomForest]:
   Fold 1: 0.845238
   Fold 2: 0.818452
   Mean:   0.815179
   Std:    0.033227

3. halving_factor (int, default=3)
   ├─ 2: Fastest (50% elimination per round)
   ├─ 3: Balanced (67% elimination) ← RECOMMENDED
   ├─ 4: Thorough (75% elimination)
   └─ 5+: Very thorough (80%+ elimination)
   
   Only used when search_type='halving'
   
   Example:
   halving_factor = 2   # For huge param spaces (>500 combinations)
   halving_factor = 3   # Default, works for most cases
   halving_factor = 4   # When accuracy matters more than speed

4. memory_efficient (bool, default=False)
   ├─ False: Generate all plots (default)
   └─ True:  Disable plots to save RAM (60-70% reduction)
   
   Automatically disables when True:
   ✗ Learning curves
   ✗ Feature importance plots
   ✗ Diagnostic plots (confusion matrix, residuals)
   ✗ Model comparison boxplots
   
   Still shows in terminal/results:
   ✓ Fold scores
   ✓ All metrics (Accuracy, F1, etc.)
   ✓ Model rankings
   ✓ Summary table
   
   Example:
   memory_efficient = True  # For datasets > 1M rows

PARAMETER COMBINATIONS FOR DIFFERENT SCENARIOS:

┌────────────────────────┬──────────────┬─────────────┬──────────────────┬────┐
│ Scenario               │ search_type  │halving_fact.│memory_efficient  │ cv │
├────────────────────────┼──────────────┼─────────────┼──────────────────┼────┤
│ Quick Screening        │ halving      │ 2           │ True             │ 3  │
│ Balanced (Default)     │ halving      │ 3           │ False            │ 5  │
│ Thorough Tuning        │ grid         │ N/A         │ False            │ 10 │
│ Large Dataset (1M+)    │ halving      │ 2           │ True             │ 3  │
│ Production Pipeline    │ halving      │ 3           │ True             │ 5  │
│ Critical Model (Slow)  │ grid         │ N/A         │ False            │ 10 │
└────────────────────────┴──────────────┴─────────────┴──────────────────┴────┘
"""

# ============================================================================
# BEFORE & AFTER: What Terminal Output Looks Like
# ============================================================================

"""
=== SEARCH TYPE CLARITY (NEW) ===

BEFORE (Unclear):
  [RandomForest] Running HalvingGridSearchCV optimizing for 'accuracy'...

AFTER (Crystal Clear):
  [RandomForest] Running 🚀 HalvingGridSearchCV (factor=3, efficient successive halving)
         Optimizing for: 'accuracy' | CV Folds: 5

Same for GridSearchCV:
  [Model] Running GridSearchCV (exhaustive search - tests all combinations)

And RandomizedSearchCV:
  [Model] Running RandomizedSearchCV (random sampling of parameter space)

=== FOLD DETAILS (NEW) ===

When show_fold_details=True:
  📊 Fold-Level balanced_accuracy Scores for [RandomForest]:
  Fold 1: 0.845238
  Fold 2: 0.818452
  Fold 3: 0.775298
  Fold 4: 0.779762
  Fold 5: 0.857143
  
  Mean:   0.815179
  Std:    0.033227
  Min:    0.775298
  Max:    0.857143

=== MEMORY EFFICIENCY (NEW) ===

When memory_efficient=True:
  ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡
  ⚡ MEMORY-EFFICIENT MODE ENABLED
  ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡
    ✓ Disabled: Learning curves, feature importance, diagnostic plots
    ✓ Reduced memory footprint for large datasets
    ✓ Training will be significantly faster

=== RESUME/CHECKPOINT (NEW) ===

When resume=True and checkpoint exists:
  💾 RESUMING PREVIOUS RUN FROM CHECKPOINT...
  -> Successfully loaded 2 previously trained models: ['RandomForest', 'SVM']
  ⏭️ Skipping [RandomForest] (Already completed in checkpoint file)
  ⏭️ Skipping [SVM] (Already completed in checkpoint file)
  [LogisticRegression] Starting Full Evaluation...
"""
