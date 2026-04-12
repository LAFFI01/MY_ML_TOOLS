import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Enable experimental HalvingGridSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa: F401

# Imblearn Pipeline alias to protect against Scikit-Learn overrides
from imblearn.pipeline import Pipeline as ImbPipeline

# Universal Metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Model Selection & Splitters
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.metrics import make_scorer

# ==============================================================================
# CUSTOM METRICS FOR IMBALANCED CLASSIFICATION
# ==============================================================================


def balanced_multiclass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced/macro-averaged accuracy for multi-class classification.
    
    Calculates per-class recall and averages them, making it robust to class imbalance.
    Each class is weighted equally regardless of sample count.
    
    Formula: Accuracy = (1/C) * Σ(TP_i / All_samples_of_class_i)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Balanced accuracy score (0 to 1)
        
    Example:
        >>> y_true = [0, 1, 2, 0, 1, 2]
        >>> y_pred = [0, 1, 1, 0, 1, 2]
        >>> balanced_multiclass_accuracy(y_true, y_pred)
        # Returns average of recall for each class
    """
    unique_classes = np.unique(y_true)
    recalls = []
    
    for cls in unique_classes:
        class_mask = y_true == cls
        if np.sum(class_mask) > 0:
            recall = np.sum((y_true == cls) & (y_pred == cls)) / np.sum(class_mask)
            recalls.append(recall)
    
    return float(np.mean(recalls)) if recalls else 0.0


def get_balanced_accuracy_scorer():
    """
    Create a sklearn scorer for balanced multiclass accuracy.
    
    Returns:
        Scorer object compatible with GridSearchCV, cross_val_score, etc.
    """
    return make_scorer(balanced_multiclass_accuracy)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================



def _validate_inputs(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float) -> None:
    """
    Validate input data and parameters.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of rows. Got X shape={X.shape}, y shape={y.shape}"
        )

    if X.isna().any().any():
        raise ValueError(f"X contains NaN values. Found {X.isna().sum().sum()} NaN values")

    if y.isna().any():
        raise ValueError(f"y contains NaN values. Found {y.isna().sum()} NaN values")

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1. Got {test_size}")

    if not 0 <= val_size < 1:
        raise ValueError(f"val_size must be between 0 and 1. Got {val_size}")

    if test_size + val_size >= 1:
        raise ValueError(f"test_size + val_size must be < 1. Got {test_size + val_size}")

    if X.shape[0] < 10:
        raise ValueError(f"Dataset too small. Need at least 10 samples, got {X.shape[0]}")


def _get_preprocessor(
    preprocess_pipeline: Union[Any, Dict[str, Any]], model_name: str
) -> Optional[Any]:
    """
    Extract the preprocessor for a given model.
    """
    if isinstance(preprocess_pipeline, dict):
        return preprocess_pipeline.get(model_name, preprocess_pipeline.get("default"))
    return preprocess_pipeline


def _build_pipeline(preprocessor: Optional[Any], sampler: Optional[Any], model: Any) -> ImbPipeline:
    """
    Build a pipeline with preprocessor, sampler, and model.
    """
    steps = []
    if preprocessor is not None:
        if hasattr(preprocessor, "steps"):
            steps.extend(preprocessor.steps.copy())
        else:
            steps.append(("preprocessor", preprocessor))

    if sampler is not None:
        steps.append(("sampler", sampler))

    steps.append(("model", model))
    return ImbPipeline(steps)


def _build_preprocessor_cache(
    preprocess_pipeline: Union[Any, Dict[str, Any]],
    model_names: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[int, Any]:
    """
    Build and cache fitted preprocessors to avoid redundant refitting.
    
    This function:
    1. Identifies unique preprocessors by object identity
    2. Fits each unique preprocessor only ONCE
    3. Returns a cache mapping object_id -> fitted_preprocessor
    
    Args:
        preprocess_pipeline: Single or dict of preprocessor pipelines
        model_names: List of model names to check for unique preprocessors
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dict mapping preprocessor object id -> fitted preprocessor instance
    """
    cache = {}
    
    # Get all unique preprocessors
    preprocessors_to_fit = {}
    for model_name in model_names:
        preproc = _get_preprocessor(preprocess_pipeline, model_name)
        obj_id = id(preproc)
        if obj_id not in preprocessors_to_fit and preproc is not None:
            preprocessors_to_fit[obj_id] = preproc
    
    # Fit each unique preprocessor once
    for obj_id, preproc in preprocessors_to_fit.items():
        fitted = preproc.fit(X_train, y_train)
        cache[obj_id] = fitted
    
    return cache


def _build_pipeline_from_cache(
    preprocessor: Optional[Any],
    sampler: Optional[Any],
    model: Any,
    fitted_preprocessor_cache: Dict[int, Any],
) -> ImbPipeline:
    """
    Build a pipeline using a FITTED preprocessor from cache, not a fresh one.
    
    This replaces _build_pipeline() when you have a cache of fitted preprocessors.
    
    Args:
        preprocessor: Original (potentially unfitted) preprocessor object
        sampler: Imbalanced data sampler (e.g., SMOTE)
        model: The ML model to add to pipeline
        fitted_preprocessor_cache: Cache mapping object_id -> fitted_preprocessor
        
    Returns:
        ImbPipeline with fitted preprocessor steps + sampler + model
    """
    steps = []
    
    if preprocessor is not None:
        obj_id = id(preprocessor)
        if obj_id in fitted_preprocessor_cache:
            fitted_preproc = fitted_preprocessor_cache[obj_id]
            if hasattr(fitted_preproc, "steps"):
                steps.extend(fitted_preproc.steps.copy())
            else:
                steps.append(("preprocessor", fitted_preproc))
        else:
            # Fallback to unfitted preprocessor if not in cache
            if hasattr(preprocessor, "steps"):
                steps.extend(preprocessor.steps.copy())
            else:
                steps.append(("preprocessor", preprocessor))
    
    if sampler is not None:
        steps.append(("sampler", sampler))
    
    steps.append(("model", model))
    return ImbPipeline(steps)


def _extract_feature_importance(
    pipeline: Any, feature_names: Optional[List[str]] = None, top_k: int = 10
) -> Dict[str, float]:
    """
    Extract feature importance/coefficients from a trained pipeline.
    Returns a dict with feature names and their importance percentages.
    """
    fitted_model = pipeline.named_steps.get("model")
    if fitted_model is None:
        return {}

    # Get feature names after preprocessing
    current_names = list(feature_names) if feature_names else []
    preprocessor = pipeline.named_steps.get("preprocessor")

    # Try to find preprocessing step with get_feature_names_out
    if preprocessor is None:
        for step_name, step in pipeline.named_steps.items():
            if (
                step_name != "model"
                and step_name != "sampler"
                and hasattr(step, "get_feature_names_out")
            ):
                preprocessor = step
                break

    if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
        try:
            current_names = list(preprocessor.get_feature_names_out(feature_names))
        except Exception:
            pass

    # Extract importance scores
    importances_array = None
    if hasattr(fitted_model, "coef_"):
        coefs = np.abs(fitted_model.coef_)
        if coefs.ndim > 1:
            coefs = np.mean(coefs, axis=0)
        else:
            coefs = coefs.flatten()
        importances_array = coefs
    elif hasattr(fitted_model, "feature_importances_"):
        importances_array = fitted_model.feature_importances_

    if importances_array is None:
        return {}

    # Fill missing feature names
    if len(current_names) < len(importances_array):
        missing = len(importances_array) - len(current_names)
        for i in range(missing):
            current_names.append(f"Feature_{i+1}")
    elif len(current_names) > len(importances_array):
        current_names = current_names[: len(importances_array)]

    # Normalize to percentages
    total = np.sum(importances_array)
    if total > 0:
        percentages = (importances_array / total) * 100
    else:
        percentages = np.zeros_like(importances_array)

    # Create dict and sort by importance
    importance_dict = dict(zip(current_names, percentages))
    sorted_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k])

    return sorted_dict


def evaluate_and_plot_models(
    models: Dict[str, Any],
    preprocess_pipeline: Union[Any, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.0,
    split_method: str = "random",  # 'random' or 'sequential'
    stratify: bool = True,
    random_seed: int = 42,
    task_type: str = "classification",
    target_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    param_grids: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Dict[str, Any]]] = None,
    sampler: Optional[Any] = None,
    cv: int = 4,
    search_type: str = "grid",
    primary_metric: Optional[str] = None,
    top_k: Optional[int] = None,
    quick_test_fraction: float = 0.2,
    plot_lc: bool = True,
    plot_diagnostics: bool = True,
    plot_importance: bool = True,
    plot_comparison: bool = True,
    top_n_features: int = 20,
    save_dir: Optional[str] = None,
    resume: bool = False,         # <--- NEW: State check parameter
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Enterprise machine learning evaluation pipeline.
    
    Handles internal data splitting, automated tuning, imbalanced sampling,
    cross-validation, and mid-training state checkpointing for both 
    classification and regression tasks. Supports resumable training with 
    checkpoint persistence.

    Args:
        models: Dictionary of model name -> model instance mappings.
        preprocess_pipeline: Single preprocessing pipeline or dict mapping 
            model names to their respective pipelines. Use "default" key as fallback.
        X: Feature dataframe with shape (n_samples, n_features).
        y: Target series with shape (n_samples,).
        test_size: Fraction of data for test set [0.0, 1.0]. Default: 0.2.
        val_size: Fraction of training data for validation set [0.0, 1.0). 
            Default: 0.0 (no validation set).
        split_method: Either "random" (standard shuffle split) or "sequential" 
            (time series/financial data - no shuffling). Default: "random".
        stratify: Whether to stratify splits by target (classification only). 
            Default: True.
        random_seed: Random state for reproducibility. Default: 42.
        task_type: Either "classification" or "regression". Default: "classification".
        target_names: List of target class names for classification reports. Default: None.
        feature_names: List of original feature names for importance plots. Default: None.
        param_grids: Dictionary mapping model names to parameter grids for hyperparameter 
            tuning. If None, models train with default parameters. Default: None.
        fit_params: Dictionary mapping model names to fit keyword arguments 
            (e.g., early stopping). Default: None.
        sampler: Imbalanced data sampler (e.g., SMOTE). Applied to all models. 
            Ignored for regression. Default: None.
        cv: Number of cross-validation folds. Default: 4.
        search_type: Hyperparameter search method: "grid", "random", or "halving". 
            "halving" uses successive halving for efficient large-scale tuning. Default: "grid".
        primary_metric: Metric to optimize ("accuracy", "f1", "r2", "balanced_accuracy", etc.). 
            Use "balanced_accuracy" for imbalanced classification tasks. 
            Auto-selected if None. Default: None.
        top_k: Filter to top-k models after quick screening phase. 
            If None, all models evaluated. Default: None.
        quick_test_fraction: Fraction of training data used in screening phase. 
            Default: 0.2.
        plot_lc: Whether to generate learning curves. Default: True.
        plot_diagnostics: Whether to generate confusion matrices/residual plots. 
            Default: True.
        plot_importance: Whether to display feature importance/coefficients. 
            Default: True.
        plot_comparison: Whether to generate boxplot comparison of all models. 
            Default: True.
        top_n_features: Number of top features to display in importance plot. 
            Default: 20.
        save_dir: Directory to save models and checkpoint files. 
            If None, no models saved. Default: None.
        resume: If True and checkpoint exists, resume training from last completed 
            model. If False, start fresh. Default: False.
        verbose: Whether to print progress messages. Default: True.

    Returns:
        Dictionary containing:
        - "summary_df": DataFrame with ranked model performance metrics.
        - "best_models": Dict mapping model names to fitted pipeline objects.
        - "raw_cv_scores": Dict mapping model names to cross-validation score arrays.
        - "ultimate_winner": Best performing model's fitted pipeline.
        - "data_splits": Dict with train/val/test splits for post-processing:
            {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}

    Notes:
        - Checkpoint files are auto-saved to save_dir/eval_checkpoint.pkl after 
          each model completes. Use resume=True to continue from interruption.
        - For gradient boosting models (XGBoost, LightGBM, CatBoost), validation 
          set is auto-injected into fit_params if val_size > 0.0.
        - IPython.display is optional; falls back to pandas.to_string() if unavailable.
    """

    def vprint(text):
        if verbose:
            print(text)

    # Validate inputs early
    _validate_inputs(X, y, test_size, val_size)

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==============================================================================
    # 🧠 DYNAMIC TASK SETUP
    # ==============================================================================
    task_type = task_type.lower()
    split_method = split_method.lower()

    if task_type not in ["classification", "regression"]:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    if task_type == "classification":
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)
        if primary_metric is None:
            primary_metric = "accuracy"
        # Convert balanced_accuracy string to scorer object
        if primary_metric == "balanced_accuracy":
            primary_metric = get_balanced_accuracy_scorer()
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
        if primary_metric is None:
            primary_metric = "r2"

        if sampler is not None:
            vprint("⚠️ WARNING: Samplers cannot be used for regression. Disabling sampler...")
            sampler = None

    # ==============================================================================
    # 🔀 INTERNAL DATA SPLITTING ENGINE
    # ==============================================================================
    vprint("\n" + "=" * 70)
    vprint(f"🔀 DATA SPLIT LOGIC (Method: {split_method.capitalize()})")
    vprint("=" * 70)

    if split_method == "sequential":
        # Used for Time Series or Financial Data (No Shuffling allowed!)
        test_idx = int(len(X) * (1 - test_size))
        X_train_full, X_test = X.iloc[:test_idx], X.iloc[test_idx:]
        y_train_full, y_test = y.iloc[:test_idx], y.iloc[test_idx:]
    else:
        # Standard Random Split
        stratify_col = y if (task_type == "classification" and stratify) else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=stratify_col
        )

    # Optional Validation Set Logic
    X_val, y_val = None, None
    X_train, y_train = X_train_full, y_train_full

    if val_size > 0.0:
        val_fraction = val_size / (1.0 - test_size)
        if split_method == "sequential":
            val_idx = int(len(X_train_full) * (1 - val_fraction))
            X_train, X_val = X_train_full.iloc[:val_idx], X_train_full.iloc[val_idx:]
            y_train, y_val = y_train_full.iloc[:val_idx], y_train_full.iloc[val_idx:]
        else:
            stratify_val = y_train_full if (task_type == "classification" and stratify) else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=val_fraction,
                random_state=random_seed,
                stratify=stratify_val,
            )
        vprint(f"Data Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    else:
        vprint(f"Data Shapes -> Train: {X_train.shape}, Test: {X_test.shape}")

    # ==============================================================================
    # 🚀 NEW: BUILD PREPROCESSOR CACHE (Fit each unique preprocessor once)
    # ==============================================================================
    vprint("\n" + "=" * 70)
    vprint("🔧 Building Preprocessor Cache (fitting each unique preprocessor ONCE)")
    vprint("=" * 70)
    fitted_preprocessor_cache = _build_preprocessor_cache(
        preprocess_pipeline, list(models.keys()), X_train, y_train
    )
    vprint(f"✓ Cache built: {len(fitted_preprocessor_cache)} unique preprocessor(s) fitted")

    # ==============================================================================
    # PHASE 1: QUICK SCREENING
    # ==============================================================================
    if top_k is not None and top_k < len(models):
        vprint("\n" + "=" * 70)
        vprint(
            f"🚀 PHASE 1: Quick Screening (using {quick_test_fraction*100:.0f}% of training data)"
        )
        vprint("=" * 70)

        stratify_arg = y_train if (task_type == "classification" and stratify) else None
        X_sub, _, y_sub, _ = train_test_split(
            X_train,
            y_train,
            train_size=quick_test_fraction,
            stratify=stratify_arg,
            random_state=random_seed,
        )

        screening_scores = {}
        for name, model in models.items():
            current_preprocessor = _get_preprocessor(preprocess_pipeline, name)
            base_pipeline = _build_pipeline_from_cache(
                current_preprocessor, sampler, model, fitted_preprocessor_cache
            )
            f_kwargs = fit_params.get(name, {}) if fit_params else {}

            try:
                scores = cross_val_score(
                    base_pipeline,
                    X_sub,
                    y_sub,
                    cv=cv_splitter,
                    scoring=primary_metric,
                    n_jobs=-1,
                    params=f_kwargs,
                )
                screening_scores[name] = np.mean(scores)
                score_msg = (
                    f"   -> [{name}] Preliminary Score ({primary_metric}): "
                    f"{screening_scores[name]:.4f}"
                )
                vprint(score_msg)
            except Exception as e:
                vprint(f"   -> [{name}] Failed during screening ({str(e)}). Skipping.")

        top_model_names = sorted(
            screening_scores,
            key=lambda x: screening_scores[x],  # type: ignore
            reverse=True,
        )[:top_k]

        vprint(f"\n🏆 Top {top_k} models advancing to Phase 2:")
        for rank, name in enumerate(top_model_names, 1):
            vprint(f"   {rank}. {name} ({screening_scores[name]:.4f})")

        models = {name: models[name] for name in top_model_names}

    # ==============================================================================
    # 💾 NEW: CHECKPOINT LOADING ENGINE
    # ==============================================================================
    checkpoint_file = os.path.join(save_dir, "eval_checkpoint.pkl") if save_dir else None

    if resume and checkpoint_file and os.path.exists(checkpoint_file):
        vprint("\n" + "=" * 70)
        vprint(f"💾 RESUMING PREVIOUS RUN FROM CHECKPOINT...")
        vprint("=" * 70)
        state = joblib.load(checkpoint_file)
        results_score = state.get("results_score", [])
        test_scores_list = state.get("test_scores_list", [])
        successful_names = state.get("successful_names", [])
        summary_data = state.get("summary_data", [])
        best_estimators = state.get("best_estimators", {})
        vprint(f"-> Successfully loaded {len(successful_names)} previously trained models: {successful_names}")
    else:
        results_score = []
        test_scores_list = []
        successful_names = []
        summary_data = []
        best_estimators = {}

    # ==============================================================================
    # PHASE 2: FULL EVALUATION AND TUNING
    # ==============================================================================
    for name, model in models.items():
        
        # --- NEW: The Skip Logic ---
        if name in successful_names:
            vprint(f"\n⏭️ Skipping [{name}] (Already completed in checkpoint file).")
            continue

        start_time = time.time()

        vprint("\n" + "=" * 70)
        vprint(f"[{name}] Starting Full Evaluation...")
        vprint("=" * 70)

        try:
            current_preprocessor = _get_preprocessor(preprocess_pipeline, name)
            base_pipeline = _build_pipeline_from_cache(
                current_preprocessor, sampler, model, fitted_preprocessor_cache
            )

            best_params_display = "Default"
            pipeline = base_pipeline
            # Create a copy so we don't mutate the user's original dictionary
            f_kwargs = dict(fit_params.get(name, {})) if fit_params else {}

            # Auto-inject validation set for supported Gradient Boosting models
            if val_size > 0.0 and type(model).__name__ in [
                "XGBClassifier",
                "XGBRegressor",
                "LGBMClassifier",
                "LGBMRegressor",
                "CatBoostClassifier",
                "CatBoostRegressor",
            ]:
                f_kwargs.setdefault("model__eval_set", [(X_val, y_val)])

            if param_grids is not None and name in param_grids:
                search_type_lower = search_type.lower()
                if search_type_lower == "grid":
                    search_cls = GridSearchCV
                    kwargs = {"cv": cv_splitter, "scoring": primary_metric, "n_jobs": -1}
                elif search_type_lower == "halving":
                    search_cls = HalvingGridSearchCV
                    kwargs = {"cv": cv_splitter, "scoring": primary_metric, "n_jobs": -1, "random_state": random_seed}
                else:  # "random"
                    search_cls = RandomizedSearchCV
                    kwargs = {"cv": cv_splitter, "scoring": primary_metric, "n_jobs": -1, "n_iter": 10, "random_state": random_seed}

                vprint(
                    f"[{name}] Running {search_cls.__name__} optimizing for '{primary_metric}'..."
                )
                search = search_cls(base_pipeline, param_grids[name], **kwargs)
                search.fit(X_train, y_train, **f_kwargs)
                pipeline = search.best_estimator_

                clean_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
                best_params_display = str(clean_params)
                vprint(f"[{name}] Best Params Found: {clean_params}")
            else:
                vprint(f"[{name}] No param grid provided. Training base model...")
                pipeline.fit(X_train, y_train, **f_kwargs)

            best_estimators[name] = pipeline

            if save_dir:
                safe_name = name.replace(" ", "_").lower()
                filepath = os.path.join(save_dir, f"{safe_name}_best_{run_timestamp}.pkl")
                joblib.dump(pipeline, filepath)
                vprint(f"[{name}] Model saved to: {filepath}")

            # ✨ Display Feature Importance with Percentages (only if plot_importance=True)
            if plot_importance:
                feature_importance = _extract_feature_importance(pipeline, feature_names, top_k=10)
                if feature_importance:
                    vprint(f"\n📊 Top Features Used by [{name}] for Prediction:")
                    vprint("-" * 60)
                    for rank, (feat_name, importance_pct) in enumerate(
                        feature_importance.items(), 1
                    ):
                        bar_length = int(importance_pct / 2)  # Scale for display
                        bar = "█" * bar_length
                        vprint(f"  {rank:2d}. {feat_name:30s} {importance_pct:6.2f}% {bar}")
                    vprint("-" * 60)

            vprint(f"[{name}] Calculating Final Cross-Validation Scores...")
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_splitter,
                scoring=primary_metric,
                n_jobs=-1,
                params=f_kwargs,
            )
            results_score.append(scores)
            successful_names.append(name)
            mean_cv, std_cv = np.mean(scores), np.std(scores)

            if plot_lc:
                vprint(f"[{name}] Generating Learning Curve...")
                train_sizes, train_scores, val_scores = learning_curve(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_splitter,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring=primary_metric,
                    n_jobs=-1,
                    params=f_kwargs,
                )
                fig = plt.figure(figsize=(8, 4))
                plt.plot(
                    train_sizes,
                    np.mean(train_scores, axis=1),
                    "o-",
                    color="r",
                    label="Training score",
                )
                plt.plot(
                    train_sizes, np.mean(val_scores, axis=1), "o-", color="g", label="CV score"
                )
                plt.title(f"Learning Curve: {name} ({primary_metric})")
                plt.xlabel("Number of Training Examples")
                plt.ylabel(f"Score ({primary_metric})")
                plt.legend(loc="best")
                plt.grid(True)
                plt.tight_layout()
                if verbose:
                    plt.show()
                plt.close(fig)

            y_val_pred = pipeline.predict(X_test)

            if task_type == "classification":
                # --- NEW: Comprehensive Classification Metrics ---
                # Core Scalar Metrics
                accuracy = accuracy_score(y_test, y_val_pred)
                precision = precision_score(y_test, y_val_pred, average="macro", zero_division=0)
                recall = recall_score(y_test, y_val_pred, average="macro", zero_division=0)
                f1 = f1_score(y_test, y_val_pred, average="macro", zero_division=0)
                
                # Specialized Metrics
                mcc = matthews_corrcoef(y_test, y_val_pred)
                kappa = cohen_kappa_score(y_test, y_val_pred)
                
                # Probabilistic Metrics (requires predicted probabilities)
                try:
                    y_proba = pipeline.predict_proba(X_test)
                    logloss = log_loss(y_test, y_proba)
                    
                    # Threshold-Agnostic Metrics (binary classification)
                    if y_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
                except Exception:
                    y_proba = None
                    logloss = None
                    roc_auc = None
                
                # Display metrics
                metric1 = accuracy
                metric2 = f1
                
                vprint(f"\n{'='*70}")
                vprint(f"Classification Metrics for {name}:")
                vprint(f"{'='*70}")
                vprint(f"Core Scalar Metrics:")
                vprint(f"  • Accuracy:  {accuracy:.4f}")
                vprint(f"  • Precision: {precision:.4f}")
                vprint(f"  • Recall:    {recall:.4f}")
                vprint(f"  • F1-Score:  {f1:.4f}")
                vprint(f"Specialized Metrics:")
                vprint(f"  • MCC (Matthews Correlation Coefficient): {mcc:.4f}")
                vprint(f"  • Cohen's Kappa: {kappa:.4f}")
                if logloss is not None:
                    vprint(f"Probabilistic Metrics:")
                    vprint(f"  • Log-Loss: {logloss:.4f}")
                if roc_auc is not None:
                    vprint(f"Threshold-Agnostic Metrics:")
                    vprint(f"  • ROC-AUC: {roc_auc:.4f}")
                vprint(f"{'='*70}")
                
                vprint(f"\nDetailed Classification Report for {name}:")
                vprint(
                    classification_report(
                        y_test, y_val_pred, target_names=target_names, zero_division=0
                    )
                )
            else:
                # --- NEW: Comprehensive Regression Metrics ---
                # Error/Scale Metrics
                mae = mean_absolute_error(y_test, y_val_pred)
                mse = mean_squared_error(y_test, y_val_pred)
                rmse = np.sqrt(mse)
                
                # Percentage Metrics
                try:
                    mape = mean_absolute_percentage_error(y_test, y_val_pred)
                except Exception:
                    mape = None
                
                # Goodness-of-Fit Metrics
                r2 = r2_score(y_test, y_val_pred)
                
                # Adjusted R² (accounts for number of features)
                n_samples = y_test.shape[0]
                n_features = X_test.shape[1]
                adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1) if n_samples > n_features else r2
                
                # Logarithmic Metrics
                try:
                    rmsle = np.sqrt(mean_squared_log_error(y_test, y_val_pred))
                except Exception:
                    rmsle = None
                
                # Display metrics
                metric1 = r2
                metric2 = rmse
                
                vprint(f"\n{'='*70}")
                vprint(f"Regression Metrics for {name}:")
                vprint(f"{'='*70}")
                vprint(f"Error/Scale Metrics:")
                vprint(f"  • MAE (Mean Absolute Error):        {mae:.4f}")
                vprint(f"  • MSE (Mean Squared Error):         {mse:.4f}")
                vprint(f"  • RMSE (Root Mean Squared Error):   {rmse:.4f}")
                if mape is not None:
                    vprint(f"Percentage Metrics:")
                    vprint(f"  • MAPE (Mean Absolute % Error):     {mape:.4f}")
                vprint(f"Goodness-of-Fit Metrics:")
                vprint(f"  • R² Score:                         {r2:.4f}")
                vprint(f"  • Adjusted R²:                      {adj_r2:.4f}")
                if rmsle is not None:
                    vprint(f"Logarithmic Metrics:")
                    vprint(f"  • RMSLE (Root Mean Squared Log Error): {rmsle:.4f}")
                vprint(f"{'='*70}")

            test_scores_list.append(metric1)

            if plot_diagnostics:
                fig, ax = plt.subplots(figsize=(6, 5))
                if task_type == "classification":
                    ConfusionMatrixDisplay.from_estimator(
                        pipeline,
                        X_test,
                        y_test,
                        display_labels=target_names,
                        cmap="Reds",
                        ax=ax,
                        xticks_rotation=45,
                    )
                    plt.title(f"Confusion Matrix: {name}")
                else:
                    ax.scatter(y_test, y_val_pred, alpha=0.6, color="steelblue", edgecolor="k")
                    min_val = min(y_test.min(), y_val_pred.min())
                    max_val = max(y_test.max(), y_val_pred.max())
                    ax.plot(
                        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit (y=x)"
                    )
                    ax.set_xlabel("Actual Values", fontsize=11)
                    ax.set_ylabel("Predicted Values", fontsize=11)
                    ax.set_title(f"Predicted vs. Actual: {name}", fontsize=12, fontweight="bold")
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.6)

                plt.tight_layout()
                if verbose:
                    plt.show()
                plt.close(fig)
                
                # --- NEW: Advanced Regression Visualizations ---
                if task_type == "regression":
                    try:
                        # Residuals and Distribution
                        residuals = y_test - y_val_pred
                        
                        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                        
                        # 1. Residuals vs Predicted Values (Homoscedasticity check)
                        axes[0, 0].scatter(y_val_pred, residuals, alpha=0.6, color="steelblue", edgecolor="k")
                        axes[0, 0].axhline(y=0, color="r", linestyle="--", lw=2)
                        axes[0, 0].set_xlabel("Predicted Values", fontsize=11)
                        axes[0, 0].set_ylabel("Residuals", fontsize=11)
                        axes[0, 0].set_title("Residuals vs Predicted (Homoscedasticity)", fontsize=12, fontweight="bold")
                        axes[0, 0].grid(True, alpha=0.3)
                        
                        # 2. Residuals Distribution (Normality check)
                        axes[0, 1].hist(residuals, bins=30, color="teal", alpha=0.7, edgecolor="black")
                        axes[0, 1].axvline(x=0, color="r", linestyle="--", lw=2)
                        axes[0, 1].set_xlabel("Residuals", fontsize=11)
                        axes[0, 1].set_ylabel("Frequency", fontsize=11)
                        axes[0, 1].set_title("Residuals Distribution", fontsize=12, fontweight="bold")
                        axes[0, 1].grid(True, alpha=0.3, axis="y")
                        
                        # 3. Q-Q Plot (Normality check)
                        from scipy import stats
                        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
                        axes[1, 0].set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight="bold")
                        axes[1, 0].grid(True, alpha=0.3)
                        
                        # 4. Scale-Location Plot (Sqrt of standardized residuals vs fitted)
                        standardized_residuals = residuals / np.std(residuals)
                        axes[1, 1].scatter(y_val_pred, np.sqrt(np.abs(standardized_residuals)), 
                                         alpha=0.6, color="darkgreen", edgecolor="k")
                        axes[1, 1].set_xlabel("Predicted Values", fontsize=11)
                        axes[1, 1].set_ylabel("√|Standardized Residuals|", fontsize=11)
                        axes[1, 1].set_title("Scale-Location Plot", fontsize=12, fontweight="bold")
                        axes[1, 1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        if verbose:
                            plt.show()
                        plt.close(fig)
                    except Exception as e:
                        vprint(f"⚠️ Could not generate regression diagnostics for {name}: {str(e)}")
                
                # --- NEW: ROC & PR Curves for Classification ---
                if task_type == "classification" and y_proba is not None:
                    # ROC Curve
                    try:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                        
                        # Binary classification ROC curve
                        if y_proba.shape[1] == 2:
                            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                            roc_auc_val = auc(fpr, tpr)
                            ax1.plot(fpr, tpr, color="steelblue", lw=2.5, 
                                    label=f"ROC curve (AUC = {roc_auc_val:.3f})")
                        else:
                            # Multi-class: One-vs-Rest
                            colors = plt.cm.Set1(np.linspace(0, 1, y_proba.shape[1]))
                            for i in range(y_proba.shape[1]):
                                fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
                                roc_auc_val = auc(fpr, tpr)
                                ax1.plot(fpr, tpr, color=colors[i], lw=2, 
                                        label=f"Class {i} (AUC = {roc_auc_val:.3f})")
                        
                        ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
                        ax1.set_xlabel("False Positive Rate", fontsize=11)
                        ax1.set_ylabel("True Positive Rate", fontsize=11)
                        ax1.set_title(f"ROC Curve: {name}", fontsize=12, fontweight="bold")
                        ax1.legend(loc="lower right")
                        ax1.grid(True, alpha=0.3)
                        
                        # Precision-Recall Curve
                        if y_proba.shape[1] == 2:
                            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba[:, 1])
                            pr_auc = auc(recall_vals, precision_vals)
                            ax2.plot(recall_vals, precision_vals, color="darkgreen", lw=2.5,
                                    label=f"PR curve (AUC = {pr_auc:.3f})")
                        else:
                            # Multi-class: One-vs-Rest
                            for i in range(y_proba.shape[1]):
                                precision_vals, recall_vals, _ = precision_recall_curve(
                                    (y_test == i).astype(int), y_proba[:, i]
                                )
                                pr_auc = auc(recall_vals, precision_vals)
                                ax2.plot(recall_vals, precision_vals, color=colors[i], lw=2,
                                        label=f"Class {i} (AUC = {pr_auc:.3f})")
                        
                        ax2.set_xlabel("Recall", fontsize=11)
                        ax2.set_ylabel("Precision", fontsize=11)
                        ax2.set_title(f"Precision-Recall Curve: {name}", fontsize=12, fontweight="bold")
                        ax2.legend(loc="best")
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        if verbose:
                            plt.show()
                        plt.close(fig)
                    except Exception as e:
                        vprint(f"⚠️ Could not generate ROC/PR curves for {name}: {str(e)}")

            elapsed_time = time.time() - start_time

            summary_data.append(
                {
                    "Model Name": name,
                    "Train Time": f"{elapsed_time:.1f}s",
                    f"CV Mean ({primary_metric})": f"{mean_cv:.3f} (±{std_cv:.3f})",
                    "Metric 1 Raw": metric1,
                    "Metric 2 Raw": metric2,
                    "Best Params": best_params_display,
                }
            )
            
            # --- NEW: INCREMENTAL STATE SAVING ---
            # Automatically save the progress after every successfully completed model
            if checkpoint_file:
                joblib.dump(
                    {
                        "results_score": results_score,
                        "test_scores_list": test_scores_list,
                        "successful_names": successful_names,
                        "summary_data": summary_data,
                        "best_estimators": best_estimators,
                    },
                    checkpoint_file,
                )
                vprint(f"💾 Checkpoint updated: State saved successfully.")

        except Exception:
            vprint(f"\n[ERROR] Model '{name}' failed during evaluation!")
            vprint(traceback.format_exc())
            vprint(f"Skipping {name} and moving to the next model...\n")
            continue

    if plot_comparison and len(results_score) > 0:
        fig = plt.figure(figsize=(12, 6))
        plt.boxplot(
            results_score, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.4)
        )
        plt.xticks(range(1, len(successful_names) + 1), successful_names, rotation=45, ha="right")

        for i, scores in enumerate(results_score):
            plt.plot(np.random.normal(i + 1, 0.04, size=len(scores)), scores, "b.", alpha=0.4)
        for i, test_val in enumerate(test_scores_list):
            plt.plot(
                i + 1,
                test_val,
                marker="*",
                color="red",
                markersize=14,
                label="Test Set Score" if i == 0 else "",
            )

        plt.title(f"Final Model Comparison (Metric: {primary_metric})")
        plt.ylabel("Score")
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc="lower right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    summary_df = pd.DataFrame(summary_data)
    ultimate_winner = None

    if not summary_df.empty:
        if task_type == "classification":
            summary_df = summary_df.sort_values(
                by=["Metric 1 Raw", "Metric 2 Raw"], ascending=[False, False]
            ).reset_index(drop=True)
            col1_name, col2_name = "Test Acc", "Test F1"
        else:
            summary_df = summary_df.sort_values(
                by=["Metric 1 Raw", "Metric 2 Raw"], ascending=[False, True]
            ).reset_index(drop=True)
            col1_name, col2_name = "Test R²", "Test RMSE"

        summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))
        winner_name = summary_df.iloc[0]["Model Name"]
        ultimate_winner = best_estimators[winner_name]

        summary_df[col1_name] = summary_df["Metric 1 Raw"].apply(lambda x: f"{x:.4f}")
        summary_df[col2_name] = summary_df["Metric 2 Raw"].apply(lambda x: f"{x:.4f}")
        summary_df = summary_df.drop(columns=["Metric 1 Raw", "Metric 2 Raw"])

        cols = [
            "Rank",
            "Model Name",
            "Train Time",
            f"CV Mean ({primary_metric})",
            col1_name,
            col2_name,
            "Best Params",
        ]
        summary_df = summary_df[cols]

        print("\n" + "=" * 100)
        print(f"🏆 RANKED MODEL PERFORMANCE SUMMARY (Winner: {winner_name})")
        print("=" * 100)

        try:
            from IPython.display import display

            styled_df = summary_df.style.set_properties(**{"text-align": "left"}).set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]
            )
            display(styled_df)
        except ImportError:
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                1000,
                "display.max_colwidth",
                None,
            ):
                print(summary_df.to_string(index=False))
        print("=" * 100)

    return {
        "summary_df": summary_df,
        "best_models": best_estimators,
        "raw_cv_scores": dict(zip(successful_names, results_score)),
        "ultimate_winner": ultimate_winner,
        "data_splits": {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        },
    }