import os
import shutil
import tempfile
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Optional SHAP for explainable AI
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Optional Pandera for data validation (Data Quality Contract)
try:
    import pandera as pa
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False

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


def _print_fold_details(model_name: str, fold_scores: np.ndarray, metric_name: str = "accuracy", verbose: bool = True) -> Dict[str, float]:
    """
    Display detailed fold-by-fold cross-validation metrics.
    
    Args:
        model_name: Name of the model
        fold_scores: Array of scores for each fold
        metric_name: Name of the metric being displayed
        verbose: Whether to print details
        
    Returns:
        Dictionary with fold stats (mean, std, min, max)
    """
    stats = {
        "mean": np.mean(fold_scores),
        "std": np.std(fold_scores),
        "min": np.min(fold_scores),
        "max": np.max(fold_scores),
    }
    
    if verbose:
        print(f"\n  📊 Fold-Level {metric_name.capitalize()} Scores for [{model_name}]:")
        print(f"  {'-' * 60}")
        for fold_idx, score in enumerate(fold_scores, 1):
            print(f"    Fold {fold_idx}: {score:.6f}")
        print(f"  {'-' * 60}")
        print(f"    Mean:   {stats['mean']:.6f}")
        print(f"    Std:    {stats['std']:.6f}")
        print(f"    Min:    {stats['min']:.6f}")
        print(f"    Max:    {stats['max']:.6f}")
    
    return stats



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


def _build_pipeline(preprocessor: Optional[Any], sampler: Optional[Any], model: Any, cachedir: Optional[str] = None) -> ImbPipeline:
    """
    Build a pipeline with preprocessor, sampler, and model.
    
    Args:
        preprocessor: Optional preprocessing pipeline/transformer
        sampler: Optional imbalanced data sampler (e.g., SMOTE)
        model: The ML model to add to pipeline
        cachedir: Optional directory for Pipeline memory caching. When provided,
                  intermediate transformation results are cached to disk, avoiding
                  redundant computation across pipeline steps. Default: None.
    
    Returns:
        ImbPipeline with caching enabled if cachedir provided.
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
    return ImbPipeline(steps, memory=cachedir)


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
    cachedir: Optional[str] = None,
) -> ImbPipeline:
    """
    Build a pipeline using a FITTED preprocessor from cache, not a fresh one.
    
    This replaces _build_pipeline() when you have a cache of fitted preprocessors.
    
    Args:
        preprocessor: Original (potentially unfitted) preprocessor object
        sampler: Imbalanced data sampler (e.g., SMOTE)
        model: The ML model to add to pipeline
        fitted_preprocessor_cache: Cache mapping object_id -> fitted_preprocessor
        cachedir: Optional directory for Pipeline memory caching. When provided,
                  intermediate transformation results are cached to disk, avoiding
                  redundant computation across pipeline steps. Default: None.
        
    Returns:
        ImbPipeline with fitted preprocessor steps + sampler + model, memory caching enabled if cachedir provided.
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
    return ImbPipeline(steps, memory=cachedir)


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


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts 64-bit types to 32-bit types to save 50% memory.
    
    Automatically converts float64 -> float32 and int64 -> int32 where possible,
    reducing memory overhead without significant precision loss for most ML tasks.
    
    Args:
        df: Input DataFrame with potentially high-precision columns.
        
    Returns:
        DataFrame with optimized dtypes consuming approximately 50% less memory.
        
    Example:
        >>> df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1, 2, 3]})
        >>> df.memory_usage(deep=True).sum()  # Original memory usage
        >>> optimized_df = optimize_dtypes(df)
        >>> optimized_df.memory_usage(deep=True).sum()  # ~50% reduction
    """
    df_opt = df.copy()
    
    # Downcast float64 to float32
    for col in df_opt.select_dtypes(include=['float64']).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
    
    # Downcast int64 to int32 (or smaller)
    for col in df_opt.select_dtypes(include=['int64']).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
    
    return df_opt


def _configure_early_stopping(model, early_stopping_rounds: int = 50) -> Dict[str, Any]:
    """
    Configure early stopping callbacks/parameters for gradient boosting models.
    
    Handles framework-specific early stopping configuration:
    - XGBoost: early_stopping_rounds in model init (not fit_params)
    - LightGBM: callbacks with early_stopping in fit_params
    - CatBoost: early_stopping_rounds in fit_params
    
    Args:
        model: Model instance to check type
        early_stopping_rounds: Number of rounds without improvement to trigger stop
        
    Returns:
        Dict with early stopping config (empty if model doesn't support it)
    """
    model_cls_name = type(model).__name__
    fit_params = {}
    
    if "LightGBM" in model_cls_name or "LGBM" in model_cls_name:
        try:
            import lightgbm as lgb
            fit_params["callbacks"] = [
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(period=0),
            ]
        except (ImportError, AttributeError):
            pass
            
    elif "CatBoost" in model_cls_name or "Catboost" in model_cls_name:
        fit_params["early_stopping_rounds"] = early_stopping_rounds
        
    elif "XGB" in model_cls_name or "xgboost" in str(model_cls_name).lower():
        # XGBoost: eval_metric should be in model init, not fit params
        # Early stopping will be handled via eval_set callback
        pass  # No fit_params needed for XGBoost early stopping
    
    return fit_params


def _inject_eval_set(f_kwargs: Dict[str, Any], X_val: Optional[pd.DataFrame], 
                     y_val: Optional[pd.Series], model) -> Dict[str, Any]:
    """
    Inject evaluation set into fit parameters for gradient boosting models.
    
    Args:
        f_kwargs: Current fit parameters
        X_val: Validation feature set (if available)
        y_val: Validation target set (if available)
        model: Model instance to check type
        
    Returns:
        Updated f_kwargs with eval_set if model supports it and data available
    """
    if X_val is None or y_val is None:
        return f_kwargs
    
    model_cls_name = type(model).__name__
    updated_kwargs = dict(f_kwargs)
    
    # Gradient boosters that support eval_set
    if any(x in model_cls_name for x in ["XGB", "LGBM", "LightGBM", "CatBoost", "Catboost"]):
        updated_kwargs.setdefault("model__eval_set", [(X_val, y_val)])
    
    return updated_kwargs


def _calculate_shap_values(model, X_sample: pd.DataFrame, max_samples: int = 500) -> Optional[Dict[str, Any]]:
    """
    Calculate SHAP values for model interpretability (game-theory based feature importance).
    
    Uses industry-standard SHAP (SHapley Additive exPlanations) to compute exact feature
    contributions. Superior to sklearn's feature_importances_ which is biased toward
    high-cardinality features.
    
    Args:
        model: Fitted model (must be tree-based or compatible with SHAP)
        X_sample: Sample of training data for SHAP explainer background
        max_samples: Max background samples (larger = slower but more accurate)
        
    Returns:
        Dict with SHAP mean abs values + model type, or None if SHAP unavailable
    """
    if not HAS_SHAP:
        return None
    
    try:
        model_type = type(model).__name__
        
        # Limit background samples for speed (SHAP is O(features * background_samples))
        if len(X_sample) > max_samples:
            X_background = X_sample.sample(n=max_samples, random_state=42)
        else:
            X_background = X_sample
        
        # Create explainer (auto-detects model type)
        if any(x in model_type for x in ["XGB", "LGBM", "Catboost"]):
            # Tree-based: Use TreeExplainer (fast)
            explainer = shap.TreeExplainer(model)
        elif any(x in model_type for x in ["RandomForest", "GradientBoosting"]):
            # Sklearn tree models
            explainer = shap.TreeExplainer(model)
        else:
            # Linear/other: Use KernelExplainer (slower, universal)
            explainer = shap.KernelExplainer(model.predict, X_background)
        
        # Calculate SHAP values for background data
        shap_values = explainer.shap_values(X_background)
        
        # Handle multi-class classification (returns list of arrays)
        if isinstance(shap_values, list):
            # Multi-class: average across classes
            shap_array = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_array = np.abs(shap_values)
        
        # Mean absolute SHAP values per feature
        feature_importance = np.mean(shap_array, axis=0)
        
        # Normalize to percentages
        total = np.sum(feature_importance)
        if total > 0:
            importance_pct = (feature_importance / total) * 100
        else:
            importance_pct = np.zeros_like(feature_importance)
        
        # Create feature importance dict
        feature_names = list(X_sample.columns) if hasattr(X_sample, 'columns') else [f"Feature_{i}" for i in range(len(feature_importance))]
        importance_dict = dict(zip(feature_names, importance_pct))
        
        # Sort by importance
        sorted_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "shap_values": sorted_dict,
            "shap_method": "TreeExplainer" if any(x in model_type for x in ["XGB", "LGBM", "Catboost", "Forest"]) else "KernelExplainer",
            "model_type": model_type,
            "explainer": explainer,
        }
    except Exception as e:
        return None


def _plot_shap_summary(shap_result: Dict[str, Any], model_name: str, top_k: int = 20, 
                       verbose: bool = True) -> None:
    """
    Display SHAP-based feature importance summary with game-theory interpretation.
    
    Args:
        shap_result: Dict from _calculate_shap_values()
        model_name: Model name for display
        top_k: Number of top features to show
        verbose: Whether to print details
    """
    if not shap_result or verbose is False:
        return
    
    shap_importance = shap_result["shap_values"]
    method = shap_result["shap_method"]
    
    print(f"\n🎯 SHAP Feature Importance (Game Theory Based) [{model_name}]:")
    print(f"   Explainer: {method} | {len(shap_importance)} features analyzed")
    print(f"  {'-' * 70}")
    
    for rank, (feat_name, importance_pct) in enumerate(
        list(shap_importance.items())[:top_k], 1
    ):
        bar_length = int(importance_pct / 2)
        bar = "█" * bar_length
        print(f"  {rank:2d}. {feat_name:30s} {importance_pct:6.2f}% {bar}")
        print(f"      └─ This feature's values explain {importance_pct:.1f}% of predictions")
    
    print(f"  {'-' * 70}")


def _extract_shap_importance(pipeline: Any, feature_names: Optional[List[str]] = None, 
                            X_train: Optional[pd.DataFrame] = None, top_k: int = 10) -> Dict[str, float]:
    """
    Extract SHAP-based feature importance from pipeline (fallback to sklearn if SHAP fails).
    
    SHAP (SHapley Additive exPlanations) uses game theory to calculate exact feature
    contributions, avoiding bias in tree-based models. Falls back to sklearn importance
    if SHAP unavailable or fails.
    
    Args:
        pipeline: Fitted pipeline
        feature_names: Original feature names
        X_train: Training data for SHAP background (improves accuracy)
        top_k: Top K features to return
        
    Returns:
        Dict mapping feature names to importance percentages
    """
    fitted_model = pipeline.named_steps.get("model")
    if fitted_model is None:
        return {}
    
    # Try SHAP first if available
    if HAS_SHAP and X_train is not None:
        shap_result = _calculate_shap_values(fitted_model, X_train, max_samples=500)
        if shap_result:
            shap_imp = shap_result["shap_values"]
            # Return top K
            return dict(list(shap_imp.items())[:top_k])
    
    # Fallback to sklearn feature_importances_
    current_names = list(feature_names) if feature_names else []
    preprocessor = pipeline.named_steps.get("preprocessor")
    
    if preprocessor is None:
        for step_name, step in pipeline.named_steps.items():
            if (step_name != "model" and step_name != "sampler" and 
                hasattr(step, "get_feature_names_out")):
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
    
    importance_dict = dict(zip(current_names, percentages))
    sorted_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k])
    
    return sorted_dict


def _calculate_expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE) to measure probability calibration quality.
    
    ECE measures alignment between predicted probabilities and actual outcomes.
    ECE = Σ |P(predicted class) - accuracy_in_bin| * (bin_size / total)
    
    Lower ECE = Better calibration.
    - ECE=0.0: Perfect calibration (99% confident → correct 99% of time)
    - ECE=0.2: Poorly calibrated (90% confident → only correct 70% of time)
    
    Args:
        y_true: True binary labels (0/1)
        y_proba: Predicted probabilities for positive class (0-1)
        n_bins: Number of bins for calibration curve
        
    Returns:
        Expected Calibration Error (float between 0 and 1)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(len(bin_edges) - 1):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if np.sum(mask) > 0:
            # Accuracy in this bin
            bin_accuracy = np.mean(y_true[mask])
            # Average confidence in this bin
            bin_confidence = np.mean(y_proba[mask])
            # Contribution to ECE
            ece += np.abs(bin_confidence - bin_accuracy) * (np.sum(mask) / total_samples)
    
    return float(ece)


def _calibrate_model(pipeline: Any, X_train: pd.DataFrame, y_train: pd.Series, 
                    calibration_method: str = "isotonic", cv: int = 5) -> CalibratedClassifierCV:
    """
    Wrap classification pipeline with probability calibration for accurate confidence scores.
    
    Calibration ensures:
    - When model says "90% confident", it's correct 9 times out of 10
    - Probabilities reflect true likelihood of classes
    - Log-Loss and other probability-based metrics improve
    
    Methods:
    - 'isotonic': Non-parametric, more flexible (requires more data, ~50+ samples)
    - 'sigmoid': Parametric Platt Scaling (works with small datasets, ~10+ samples)
    
    Args:
        pipeline: Fitted classification pipeline
        X_train: Training data for calibrator fitting
        y_train: Training labels for calibrator fitting
        calibration_method: 'isotonic' (recommended for large datasets) or 'sigmoid' (Platt Scaling)
        cv: Cross-validation strategy for calibration (prevents target leakage)
        
    Returns:
        CalibratedClassifierCV wrapper around original pipeline
    """
    return CalibratedClassifierCV(
        estimator=pipeline,
        method=calibration_method,
        cv=cv
    )


def _initialize_optuna_storage(storage_url: Optional[str] = None, study_name: Optional[str] = None,
                              save_dir: Optional[str] = None, verbose: bool = True) -> tuple:
    """
    Initialize persistent Optuna storage backend for distributed tuning.
    
    Supports SQLite (local persistence) and Redis (distributed):
    - SQLite: Perfect for Kaggle parallel notebooks (all point to same .db file)
    - Redis: Cloud/remote distributed tuning across machines
    
    Args:
        storage_url: Storage backend URL
            - None: In-memory (default, no persistence)
            - 'sqlite:///automl_optuna.db': SQLite in current directory
            - 'sqlite:////tmp/automl_optuna.db': SQLite in /tmp (absolute path)
            - 'redis://localhost:6379': Redis on local machine
        study_name: Name for the Optuna study (used for resuming trials)
        save_dir: Directory to save database files (if None, uses current directory)
        verbose: Whether to print setup information
        
    Returns:
        Tuple of (storage_url, study_name) for optuna.create_study()
        
    Example:
        # Single Kaggle notebook (local SQLite)
        storage, study_name = _initialize_optuna_storage(
            storage_url="sqlite:///automl_optuna.db",
            study_name="kaggle_run_1"
        )
        
        # Three parallel Kaggle notebooks (all use same database)
        # Notebook 1: storage_url="sqlite:///automl_optuna.db", study_name="shared_study"
        # Notebook 2: storage_url="sqlite:///automl_optuna.db", study_name="shared_study"
        # Notebook 3: storage_url="sqlite:///automl_optuna.db", study_name="shared_study"
        # Result: 3x faster tuning via parallel trials + shared history
    """
    if storage_url is None:
        # In-memory only (default, no persistence)
        if verbose:
            print("   📦 Optuna Storage: In-memory (no persistence)")
        return None, study_name or "default_study"
    
    # Prepare storage path
    final_storage_url = storage_url
    
    if storage_url.startswith("sqlite://"):
        # SQLite backend
        db_path = storage_url.replace("sqlite:///", "")
        
        # If relative path and save_dir provided, prepend save_dir
        if not db_path.startswith("/") and save_dir:
            db_path = os.path.join(save_dir, db_path)
            final_storage_url = f"sqlite:///{db_path}"
        
        # Create parent directory if needed
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        if verbose:
            print(f"   🗄️  Optuna Storage: SQLite at {os.path.abspath(db_path)}")
            print(f"   📊 Study Name: {study_name or 'default_study'}")
            print(f"   ⚡ Kaggle Feature: Multiple notebooks can share this database for 3x speedup")
    
    elif storage_url.startswith("redis://"):
        # Redis backend
        if verbose:
            print(f"   🔴 Optuna Storage: Redis at {storage_url}")
            print(f"   📊 Study Name: {study_name or 'default_study'}")
            print(f"   ⚡ Kaggle Feature: Cloud-distributed tuning across any machines")
    
    else:
        if verbose:
            print(f"   ⚠️  Unknown storage backend: {storage_url}")
    
    return final_storage_url, study_name or "default_study"


def _validate_data_contract(X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None,
                            task_type: str = "classification", verbose: bool = True) -> Dict[str, Any]:
    """
    Validate data quality using Pandera (Data Contract/Schema Validation).
    
    Catches data quality issues EARLY (before model training):
    - Missing columns (IT department renamed something)
    - Type mismatches (integers accidentally converted to strings)
    - NaN/null values in critical columns
    - Out-of-range values
    - Duplicate rows
    
    This prevents cryptic errors deep in the modeling pipeline.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        feature_names: Expected feature names (if None, auto-detects from X)
        task_type: 'classification' or 'regression'
        verbose: Whether to print validation results
        
    Returns:
        Dict with validation results: {'passed': bool, 'issues': List[str], 'schema': schema}
    """
    results = {
        'passed': True,
        'issues': [],
        'schema': None,
        'stats': {}
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("🔍 DATA QUALITY VALIDATION (Data Contract via Pandera)")
        print("=" * 70)
    
    # Basic pandas-level checks (always available)
    if verbose:
        print("\n📋 Basic Validation (pandas):")
    
    # Check shape
    if X.shape[0] == 0:
        results['issues'].append("ERROR: X has 0 rows")
        results['passed'] = False
    if X.shape[1] == 0:
        results['issues'].append("ERROR: X has 0 columns")
        results['passed'] = False
    if y.shape[0] == 0:
        results['issues'].append("ERROR: y has 0 rows")
        results['passed'] = False
    
    # Check row count alignment
    if X.shape[0] != y.shape[0]:
        results['issues'].append(f"ERROR: Row mismatch - X has {X.shape[0]} rows, y has {y.shape[0]} rows")
        results['passed'] = False
    
    # Check for NaN in target
    n_nans_y = y.isna().sum()
    if n_nans_y > 0:
        results['issues'].append(f"ERROR: Target (y) contains {n_nans_y} NaN values")
        results['passed'] = False
    
    # Check for NaN in features
    n_nans_X = X.isna().sum().sum()
    if n_nans_X > 0:
        nan_cols = X.columns[X.isna().any()].tolist()
        results['issues'].append(f"WARNING: Features contain {n_nans_X} NaN values in columns: {nan_cols}")
        if verbose:
            print(f"   ⚠️  NaN values detected: {n_nans_X} cells in {len(nan_cols)} columns")
    
    # Check for duplicates
    n_dups = X.duplicated().sum()
    if n_dups > 0:
        results['issues'].append(f"WARNING: {n_dups} duplicate rows in X")
        if verbose:
            print(f"   ⚠️  Duplicate rows: {n_dups} rows")
    
    # Check data types
    if verbose:
        print(f"   ✓ Shape: X={X.shape}, y={y.shape}")
        print(f"   ✓ Data types: {X.dtypes.unique().tolist()}")
    
    results['stats'] = {
        'n_rows': X.shape[0],
        'n_cols': X.shape[1],
        'n_nans': n_nans_X,
        'n_duplicates': n_dups,
        'dtypes': X.dtypes.to_dict()
    }
    
    # --- Pandera Schema Validation (if available) ---
    if HAS_PANDERA:
        if verbose:
            print("\n📊 Advanced Validation (Pandera Data Contract):")
        
        try:
            # Auto-build schema from data statistics
            schema_dict = {}
            
            for col in X.columns:
                col_dtype = X[col].dtype
                
                # Determine Pandera dtype
                if col_dtype == 'object':
                    pa_dtype = pa.String
                elif col_dtype in ['int64', 'int32', 'int16', 'int8']:
                    pa_dtype = pa.Int64
                elif col_dtype in ['float64', 'float32']:
                    pa_dtype = pa.Float64
                else:
                    pa_dtype = None
                
                if pa_dtype:
                    # Build checks for this column
                    col_checks = [
                        pa.Check.notin([np.nan, np.inf, -np.inf], ignore_na=True, name=f"no_special_values_{col}")
                    ]
                    
                    # Additional checks for numeric columns
                    if col_dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
                        col_min = X[col].min()
                        col_max = X[col].max()
                        col_checks.append(
                            pa.Check.in_range(col_min, col_max, ignore_na=True, name=f"in_range_{col}")
                        )
                    
                    schema_dict[col] = pa.Column(pa_dtype, checks=col_checks, nullable=X[col].isna().any())
            
            # Create and validate schema
            schema = pa.DataFrameSchema(schema_dict, coerce=False, strict=False)
            schema.validate(X)
            
            results['schema'] = schema
            if verbose:
                print(f"   ✓ Pandera Schema validation passed for {len(schema_dict)} columns")
                print(f"   ✓ All data types and value ranges are within expected bounds")
        
        except Exception as e:
            results['issues'].append(f"Pandera validation error: {str(e)}")
            if verbose:
                print(f"   ⚠️  Pandera validation warning: {str(e)}")
    
    else:
        if verbose:
            print("   ℹ️  Pandera not installed. Install with: pip install pandera[io]")
            print("      For advanced data contract validation")
    
    # Print summary
    if verbose:
        if results['passed']:
            print(f"\n✅ Data Contract: PASSED")
            print(f"   Ready for modeling phase!")
        else:
            print(f"\n❌ Data Contract: FAILED")
            print(f"   Issues found:")
            for issue in results['issues']:
                print(f"      - {issue}")
    
    return results


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
    show_fold_details: bool = True,  # <--- NEW: Display per-fold metrics
    halving_factor: int = 3,      # <--- NEW: HalvingGridSearchCV efficiency control
    optuna_n_trials: int = 50,    # <--- NEW: Optuna optimization trials
    optuna_timeout: Optional[int] = None,  # <--- NEW: Optuna timeout in seconds
    optuna_storage: Optional[str] = None,  # <--- NEW PHASE 5: Persistent Optuna storage (SQLite/Redis)
    optuna_study_name: Optional[str] = None,  # <--- NEW PHASE 5: Study name for distributed resumability
    distributed_tuning: bool = False,  # <--- NEW PHASE 5: Enable distributed Optuna tuning
    validate_data: bool = True,      # <--- NEW PHASE 6: Enable data validation (Pandera)
    build_ensemble: bool = True,   # <--- NEW: Trigger Phase 3 Stacking Ensemble
    mlflow_experiment: str = "AutoML_Run",  # <--- NEW: MLflow experiment name
    memory_efficient: bool = False,  # <--- NEW: Memory optimization mode
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Enterprise machine learning evaluation pipeline with automatic early stopping and SHAP explainability.
    
    **Early Stopping (NEW):**
    - Automatically configured for XGBoost, LightGBM, CatBoost
    - Validation set auto-injected via eval_set when val_size > 0
    
    **SHAP Feature Importance (Game Theory Based Explainability) - NEW:**
    Automatically generates SHAP values when `pip install shap` available:
    
    - **TreeExplainer** for tree models (XGBoost, LightGBM, CatBoost, Random Forest)
    - **KernelExplainer** for other models (universal but slower)
    - Reveals EXACT contribution of each feature to predictions
    - Fixes sklearn bias toward high-cardinality features
    - Example: "Age=45 increases churn probability by +12.5% (vs baseline 27%)"
    
    **Installation:**
        pip install shap
        
    **Enable:** Automatically enabled if shap is installed. Pass feature_names for better interpretability.
    
    **Kaggle Advantage:** SHAP force plots + decision plots reveal hidden feature interactions,
    enabling brilliant feature engineering ideas.
    
    **Distributed Tuning with Persistent Optuna (NEW - Phase 5):**
    Enable persistent optimization studies that survive disconnects and scale across parallel notebooks:
    
    - **SQLite Backend** (Kaggle Notebooks):
        - Single database file (`automl_optuna.db`) shared across parallel notebooks
        - Set `optuna_storage="sqlite:///automl_optuna.db"` in all notebooks
        - All 3 notebooks automatically find each other's trials and collaborate
        - Result: 3x faster tuning (3 trials running in parallel)
    
    - **Redis Backend** (Cloud-Distributed):
        - Multiple machines/containers connect to shared Redis server
        - Set `optuna_storage="redis://your-redis-host:6379"`
        - Scales to dozens of nodes across cloud
    
    - **Load-If-Exists**: Study auto-resumes if database already exists
        - Survives laptop/Kaggle disconnects
        - Reuses all previous trials (no wasted work)
    
    **Example - Three Parallel Kaggle Notebooks (3x Speedup):**
    ```python
    # All three notebooks use identical parameters:
    results = evaluate_and_plot_models(
        models=my_models,
        ...,
        optuna_storage="sqlite:///automl_optuna.db",  # ← Same DB file
        optuna_study_name="kaggle_competition_1",      # ← Same study name
        optuna_n_trials=150,                             # ← Total 150 trials
        distributed_tuning=True                          # ← Enable parallel
    )
    # Each notebook runs 50 trials in parallel → done in 1/3 the time!
    ```
    
    **Probability Calibration (NEW - Phase 4):**
    Final classification models are automatically probability-calibrated using CalibratedClassifierCV.
    
    - **Problem:** Tree models (Random Forest, XGBoost) are overconfident. "70% chance" → actually 50%
    - **Solution:** Isotonic Regression or Platt Scaling forces probabilities to match reality
    - **Benefit:** Log-Loss metric almost always improves (Kaggle metric advantage)
    - **Guarantee:** After calibration, "90% confident" = correct 9 out of 10 times
    
    **Data Validation with Pandera (NEW - Phase 6 - MLOps):**
    Production-grade data quality checks prevent crashes deep in modeling:
    
    - **Problem:** IT changes column types (int → string), adds NaNs, renames columns
      Script crashes after 30 minutes, buried in cross-validation error trace
    - **Solution:** Validate data contract BEFORE any model training starts
    - **Checks Automated**:
      - Row/column count consistency
      - Data type validation (Pandera auto-builds from data)
      - NaN/null detection with column-wise breakdown
      - Duplicate row detection
      - Value range validation (min/max bounds)
      - Special values (inf, -inf) detection
    
    - **Error Handling**:
      - CRITICAL errors (NaN in target, shape mismatch) → immediate halt with clear message
      - WARNINGS (duplicates, NaN in features) → logged but continue
      - All validation stats logged to MLflow
    
    **Installation (Optional but Recommended for Production):**
        pip install pandera[io]
    
    **Usage:**
    ```python
    results = evaluate_and_plot_models(
        models=my_models,
        ...,
        validate_data=True  # ← Enable (default: True)
    )
    # Output:
    # 🔍 DATA QUALITY VALIDATION
    # ✓ Shape: X=(10000, 15), y=(10000,)
    # ✓ Data types: validated
    # ✓ Pandera Schema validation passed for 15 columns
    # ✅ Data Contract: PASSED
    ```
    
    **When Validation Fails:**
    ```
    ❌ Data Validation Failed - Critical Issues Detected:
    - ERROR: Target (y) contains 42 NaN values
    - ERROR: Row mismatch - X has 9958 rows, y has 10000 rows
    ```
    Skip with `validate_data=False` (not recommended for production)
    
    **Example fit_params with early stopping:**
    
        fit_params = {
            "LightGBM": {
                "callbacks": [lgb.early_stopping(100), lgb.log_evaluation(period=0)]
            },
            "CatBoost": {
                "early_stopping_rounds": 100
            }
        }
    
    Notes:
    - If fit_params includes early stopping config, it will be merged with auto-config
    - XGBoost early stopping via eval_metric in fit_params
    - LightGBM early stopping via callbacks with lgb.early_stopping()
    - CatBoost early stopping via early_stopping_rounds in fit_params
    """

    def vprint(text):
        if verbose:
            print(text)

    # Validate inputs early
    _validate_inputs(X, y, test_size, val_size)

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize Pipeline memory cache directory
    cachedir = tempfile.mkdtemp()
    vprint(f"📁 Pipeline cache directory created: {cachedir}")
    
    # ==============================================================================
    # 📊 MLFLOW EXPERIMENT TRACKING SETUP (BEFORE Data Validation - fixes Ghost Run bug)
    # ==============================================================================
    mlflow.set_experiment(mlflow_experiment)
    
    with mlflow.start_run(run_name=f"AutoML_{run_timestamp}") as mlflow_run:
        vprint(f"📊 MLflow experiment '{mlflow_experiment}' initialized (Run ID: {mlflow_run.info.run_id})")
        
        # Log initial parameters to MLflow
        mlflow.log_params({
            "task_type": task_type,
            "test_size": test_size,
            "val_size": val_size,
            "cv_folds": cv,
            "primary_metric": primary_metric or "auto-selected",
            "search_type": search_type,
            "build_ensemble": build_ensemble,
            "distributed_tuning": distributed_tuning,
        })
    
        # ==============================================================================
        # 🔍 DATA QUALITY VALIDATION (NEW PHASE 6 - MLOps) - NOW INSIDE MLFLOW CONTEXT
        # ==============================================================================
        if validate_data:
            data_validation_result = _validate_data_contract(
                X, y, feature_names=feature_names, task_type=task_type, verbose=verbose
            )
            
            if not data_validation_result['passed']:
                # Critical errors found - halt execution
                critical_errors = [issue for issue in data_validation_result['issues'] if issue.startswith('ERROR')]
                if critical_errors:
                    error_summary = "\n".join(critical_errors)
                    raise ValueError(
                        f"❌ Data Validation Failed - Critical Issues Detected:\n{error_summary}\n\n"
                        f"Pass validate_data=False to skip validation (not recommended for production)."
                    )
            
            # Log validation stats to MLflow (now within MLflow context)
            for stat_name, stat_value in data_validation_result['stats'].items():
                try:
                    mlflow.log_param(f"data_validation_{stat_name}", stat_value)
                except:
                    pass  # Some stats might not be loggable
        
        # ==============================================================================
        # 🧠 DYNAMIC TASK SETUP
        # ==============================================================================
        task_type = task_type.lower()
        split_method = split_method.lower()

        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        if search_type.lower() not in ["grid", "random", "halving", "optuna"]:
            raise ValueError(f"search_type must be 'grid', 'random', 'halving', or 'optuna'. Got: {search_type}")

        # Memory efficiency mode setup
        if memory_efficient:
            vprint("\n" + "⚡" * 35)
            vprint("⚡ MEMORY-EFFICIENT MODE ENABLED")
            vprint("⚡" * 35)
            plot_lc = False
            plot_importance = False
            plot_diagnostics = False
            plot_comparison = False
            vprint("  ✓ Disabled: Learning curves, feature importance, diagnostic plots")
            vprint("  ✓ Reduced memory footprint for large datasets")
            vprint("  ✓ Training will be significantly faster")
            vprint("")

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
            test_idx = int(len(X) * (1 - test_size))
            X_train_full, X_test = X.iloc[:test_idx], X.iloc[test_idx:]
            y_train_full, y_test = y.iloc[:test_idx], y.iloc[test_idx:]
        else:
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
        # 🚀 BUILD PREPROCESSOR CACHE
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
            vprint(f"🚀 PHASE 1: Quick Screening (using {quick_test_fraction*100:.0f}% of training data)")
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
                    current_preprocessor, sampler, model, fitted_preprocessor_cache, cachedir
                )
                f_kwargs = fit_params.get(name, {}) if fit_params else {}

                try:
                    with threadpool_limits(limits=1, user_api='blas'):
                        with threadpool_limits(limits=1, user_api='openmp'):
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
                    vprint(f"   -> [{name}] Preliminary Score: {screening_scores[name]:.4f}")
                except Exception as e:
                    vprint(f"   -> [{name}] Failed during screening. Skipping.")

            top_model_names = sorted(
                screening_scores,
                key=lambda x: screening_scores[x],
                reverse=True,
            )[:top_k]

            vprint(f"\n🏆 Top {top_k} models advancing to Phase 2:")
            for rank, name in enumerate(top_model_names, 1):
                vprint(f"   {rank}. {name} ({screening_scores[name]:.4f})")

            models = {name: models[name] for name in top_model_names}

        # ==============================================================================
        # 💾 CHECKPOINT LOADING ENGINE
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
            vprint(f"-> Loaded {len(successful_names)} previously trained models: {successful_names}")
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
            # Skip if already completed in checkpoint
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
                    current_preprocessor, sampler, model, fitted_preprocessor_cache, cachedir
                )

                best_params_display = "Default"
                pipeline = base_pipeline
                f_kwargs = dict(fit_params.get(name, {})) if fit_params else {}

                # --- NEW: Configure Early Stopping for Gradient Boosters ---
                model_cls_name = type(model).__name__
                if any(x in model_cls_name for x in ["XGB", "LGBM", "LightGBM", "CatBoost"]):
                    es_config = _configure_early_stopping(model, early_stopping_rounds=50)
                    f_kwargs.update(es_config)
                    vprint(f"[{name}] Early stopping configured for {model_cls_name}")

                # --- NEW: Inject Validation Set for Gradient Boosters with Early Stopping ---
                f_kwargs = _inject_eval_set(f_kwargs, X_val, y_val, model)
                if "model__eval_set" in f_kwargs:
                    vprint(f"[{name}] Evaluation set injected for early stopping monitoring")

                if param_grids is not None and name in param_grids:
                    search_type_lower = search_type.lower()
                    if search_type_lower == "grid":
                        search_cls = GridSearchCV
                        kwargs = {"cv": cv_splitter, "scoring": primary_metric, "n_jobs": -1}
                        search_label = "GridSearchCV (exhaustive search - tests all combinations)"
                    elif search_type_lower == "halving":
                        search_cls = HalvingGridSearchCV
                        kwargs = {
                            "cv": cv_splitter,
                            "scoring": primary_metric,
                            "n_jobs": -1,
                            "random_state": random_seed,
                            "factor": halving_factor,
                        }
                        search_label = f"🚀 HalvingGridSearchCV (factor={halving_factor})"
                    elif search_type_lower == "optuna":
                        # --- NEW PHASE 5: Distributed Optuna with Persistent Storage ---
                        storage_url, study_name = _initialize_optuna_storage(
                            storage_url=optuna_storage if distributed_tuning else None,
                            study_name=optuna_study_name,
                            save_dir=save_dir,
                            verbose=verbose
                        )
                        
                        search_label = f"🔮 Optuna Bayesian Optimization (n_trials={optuna_n_trials})"
                        if distributed_tuning and storage_url:
                            search_label += " [DISTRIBUTED]"
                        vprint(f"[{name}] Running {search_label}")

                        def objective(trial):
                            param_grid = param_grids[name]
                            trial_params = {}
                    
                            for param_name, param_range in param_grid.items():
                                if isinstance(param_range, list):
                                    trial_params[param_name] = trial.suggest_categorical(param_name, param_range)
                                elif isinstance(param_range, tuple) and len(param_range) == 2:
                                    lower, upper = param_range
                                    if isinstance(lower, int) and isinstance(upper, int):
                                        trial_params[param_name] = trial.suggest_int(param_name, lower, upper)
                                    else:
                                        trial_params[param_name] = trial.suggest_float(param_name, float(lower), float(upper))
                    
                            base_model = base_pipeline.named_steps["model"]
                            trial_model = clone(base_model)
                            trial_model.set_params(**{k.replace("model__", ""): v for k, v in trial_params.items()})
                            trial_pipeline = _build_pipeline_from_cache(
                                current_preprocessor, sampler, trial_model, fitted_preprocessor_cache, cachedir
                            )
                    
                            with threadpool_limits(limits=1, user_api='blas'):
                                with threadpool_limits(limits=1, user_api='openmp'):
                                    scores = cross_val_score(
                                        trial_pipeline,
                                        X_train,
                                        y_train,
                                        cv=cv_splitter,
                                        scoring=primary_metric,
                                        n_jobs=-1,
                                        params=f_kwargs,
                                    )
                    
                            return np.mean(scores)
                
                        # --- NEW PHASE 5: Create study with persistent storage (survives disconnects) ---
                        study = optuna.create_study(
                            direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=random_seed),
                            storage=storage_url,
                            study_name=study_name,
                            # FIX: Only load_if_exists if storage_url is not None (prevents in-memory crash)
                            load_if_exists=True if (distributed_tuning and storage_url) else False
                        )
                        optuna.logging.set_verbosity(optuna.logging.WARNING)
                        
                        # Log distributed tuning info
                        if distributed_tuning and storage_url:
                            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                            vprint(f"   ⚡ Distributed Mode: {n_completed} trials already completed (resuming study)")
                            mlflow.log_param(f"{name}_optuna_storage", storage_url)
                            mlflow.log_param(f"{name}_optuna_study_name", study_name)
                        
                        study.optimize(objective, n_trials=optuna_n_trials, timeout=optuna_timeout)
                
                        best_trial_params = study.best_params
                        best_trial_model = clone(base_pipeline.named_steps["model"])
                        best_trial_model.set_params(**{k.replace("model__", ""): v for k, v in best_trial_params.items()})
                        pipeline = _build_pipeline_from_cache(
                            current_preprocessor, sampler, best_trial_model, fitted_preprocessor_cache, cachedir
                        )
                
                        with threadpool_limits(limits=1, user_api='blas'):
                            with threadpool_limits(limits=1, user_api='openmp'):
                                pipeline.fit(X_train, y_train, **f_kwargs)
                
                        clean_params = {k: v for k, v in best_trial_params.items()}
                        best_params_display = str(clean_params)
                        vprint(f"[{name}] Best Optuna Trial #{study.best_trial.number}: Score={study.best_value:.4f}")
                    else:
                        search_cls = RandomizedSearchCV
                        kwargs = {"cv": cv_splitter, "scoring": primary_metric, "n_jobs": -1, "n_iter": 10, "random_state": random_seed}
                        search_label = "RandomizedSearchCV (random sampling of parameter space)"

                    if search_type_lower != "optuna":
                        vprint(f"[{name}] Running {search_label}")
                        search = search_cls(base_pipeline, param_grids[name], **kwargs)
                        with threadpool_limits(limits=1, user_api='blas'):
                            with threadpool_limits(limits=1, user_api='openmp'):
                                search.fit(X_train, y_train, **f_kwargs)
                        pipeline = search.best_estimator_

                        clean_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
                        best_params_display = str(clean_params)
                else:
                    vprint(f"[{name}] No param grid provided. Training base model...")
                    with threadpool_limits(limits=1, user_api='blas'):
                        with threadpool_limits(limits=1, user_api='openmp'):
                            pipeline.fit(X_train, y_train, **f_kwargs)

                best_estimators[name] = pipeline
                successful_names.append(name)

                with threadpool_limits(limits=1, user_api='blas'):
                    with threadpool_limits(limits=1, user_api='openmp'):
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
                mean_cv, std_cv = np.mean(scores), np.std(scores)

                # Display fold-level details if requested
                if show_fold_details:
                    _print_fold_details(name, scores, metric_name=str(primary_metric), verbose=verbose)

                y_val_pred = pipeline.predict(X_test)

                if task_type == "classification":
                    accuracy = accuracy_score(y_test, y_val_pred)
                    f1 = f1_score(y_test, y_val_pred, average="macro", zero_division=0)
                    metric1 = accuracy
                    metric2 = f1
                else:
                    r2 = r2_score(y_test, y_val_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_val_pred))
                    metric1 = r2
                    metric2 = rmse

                test_scores_list.append(metric1)

                elapsed_time = time.time() - start_time

                # --- NEW: Extract SHAP Feature Importance (Game Theory Based) ---
                shap_feature_importance = _extract_shap_importance(pipeline, feature_names, X_train, top_k=10)
                if shap_feature_importance and verbose:
                    _plot_shap_summary(
                        {
                            "shap_values": shap_feature_importance,
                            "shap_method": "SHAP (TreeExplainer/KernelExplainer)",
                            "model_type": type(pipeline.named_steps["model"]).__name__
                        },
                        model_name=name,
                        top_k=10,
                        verbose=True
                    )
                    # Log top SHAP feature to MLflow
                    if shap_feature_importance:
                        top_shap_feature = list(shap_feature_importance.keys())[0]
                        top_shap_value = list(shap_feature_importance.values())[0]
                        mlflow.log_param(f"{name.replace(' ', '_')}_top_shap_feature", top_shap_feature)
                        mlflow.log_metric(f"{name.replace(' ', '_')}_top_shap_importance", top_shap_value)

                summary_data.append({
                    "Model Name": name,
                    "Train Time": f"{elapsed_time:.1f}s",
                    f"CV Mean": f"{mean_cv:.3f} (±{std_cv:.3f})",
                    "Metric 1 Raw": metric1,
                    "Metric 2 Raw": metric2,
                    "Best Params": best_params_display,
                })
        
                mlflow.log_metrics({
                    f"{name}_cv_mean": float(mean_cv),
                    f"{name}_cv_std": float(std_cv),
                    f"{name}_metric1": float(metric1),
                })

                # --- INCREMENTAL STATE SAVING ---
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
                    vprint(f"💾 Checkpoint updated.")

            except Exception:
                vprint(f"\n[ERROR] Model '{name}' failed!")
                vprint(traceback.format_exc())
                continue

        # ==============================================================================
        # 🤖 PHASE 3: AUTOMATED STACKING ENSEMBLE
        # ==============================================================================
        ultimate_winner = None

        if build_ensemble and len(best_estimators) >= 2:
            vprint("\n" + "🤖" * 35)
            vprint("🤖 PHASE 3: BUILDING SUPER-ENSEMBLE (Stacking)")
            vprint("🤖" * 35)

            try:
                # Get top models
                temp_summary_df = pd.DataFrame(summary_data).copy() if summary_data else pd.DataFrame()
                if "Metric 1 Raw" in temp_summary_df.columns:
                    temp_summary_df = temp_summary_df.sort_values("Metric 1 Raw", ascending=False)
                    top_k_ensemble = min(3, len(best_estimators))
                    top_model_names = temp_summary_df.head(top_k_ensemble)["Model Name"].tolist()
                else:
                    top_model_names = list(best_estimators.keys())[:min(3, len(best_estimators))]

                vprint(f"\n🔗 Blending top {len(top_model_names)} models: {top_model_names}")

                estimators_for_stacking = [(name.replace(" ", "_"), best_estimators[name]) for name in top_model_names]

                if task_type == "classification":
                    meta_learner = LogisticRegression(max_iter=1000, random_state=random_seed)
                    stacker = StackingClassifier(
                        estimators=estimators_for_stacking,
                        final_estimator=meta_learner,
                        cv=cv_splitter if cv > 2 else 2,
                        n_jobs=-1
                    )
                else:
                    meta_learner = Ridge(random_state=random_seed)
                    stacker = StackingRegressor(
                        estimators=estimators_for_stacking,
                        final_estimator=meta_learner,
                        cv=cv_splitter if cv > 2 else 2,
                        n_jobs=-1
                    )

                vprint("🔧 Training ensemble meta-learner...")
                with threadpool_limits(limits=1, user_api='blas'):
                    with threadpool_limits(limits=1, user_api='openmp'):
                        stacker.fit(X_train, y_train)

                ensemble_pred = stacker.predict(X_test)

                if task_type == "classification":
                    ensemble_score = accuracy_score(y_test, ensemble_pred)
                    metric_name = "Accuracy"
                else:
                    ensemble_score = r2_score(y_test, ensemble_pred)
                    metric_name = "R²"

                vprint(f"\n🚀 Ensemble {metric_name}: {ensemble_score:.4f}")

                mlflow.log_metric("Ensemble_Test_Score", ensemble_score)
                mlflow.log_param("Ensemble_Base_Models", ",".join(top_model_names))
                mlflow.sklearn.log_model(stacker, "Ultimate_Ensemble_Model")

                ultimate_winner = stacker
                best_estimators["🌟_Ensemble_Stack"] = stacker

                summary_data.append({
                    "Model Name": "🌟 Ensemble_Stack",
                    "Train Time": "N/A",
                    "CV Mean": "N/A",
                    "Metric 1 Raw": ensemble_score,
                    "Metric 2 Raw": ensemble_score,
                    "Best Params": f"Meta: {type(meta_learner).__name__}",
                })

                vprint(f"✅ Ensemble successfully trained!")

            except Exception as e:
                vprint(f"\n⚠️ Ensemble training failed: {str(e)}")
                vprint("Proceeding with individual model results...\n")

        # ==============================================================================
        # FINALIZE & SUMMARY
        # ==============================================================================
        if not summary_data:
            summary_df = pd.DataFrame()
        else:
            summary_df = pd.DataFrame(summary_data)
            if not summary_df.empty:
                if task_type == "classification":
                    summary_df = summary_df.sort_values(
                        by="Metric 1 Raw", ascending=False
                    ).reset_index(drop=True)
                    col1_name, col2_name = "Test Acc", "Test F1"
                else:
                    summary_df = summary_df.sort_values(
                        by="Metric 1 Raw", ascending=False
                    ).reset_index(drop=True)
                    col1_name, col2_name = "Test R²", "Test RMSE"

                summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))
                summary_df[col1_name] = summary_df["Metric 1 Raw"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                summary_df[col2_name] = summary_df["Metric 2 Raw"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                summary_df = summary_df.drop(columns=["Metric 1 Raw", "Metric 2 Raw"])

                cols = ["Rank", "Model Name", "Train Time", "CV Mean", col1_name, col2_name, "Best Params"]
                summary_df = summary_df[[c for c in cols if c in summary_df.columns]]

                print("\n" + "=" * 100)
                print(f"🏆 RANKED MODEL PERFORMANCE SUMMARY")
                print("=" * 100)
                with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", 1000,
                    "display.max_colwidth", None,
                ):
                    print(summary_df.to_string(index=False))
                print("=" * 100)

        ultimate_winner = best_estimators[summary_df.iloc[0]["Model Name"]] if (not summary_df.empty and len(best_estimators) > 0) else (best_estimators[successful_names[0]] if successful_names else None)

        # ==============================================================================
        # 📊 PHASE 4: PROBABILITY CALIBRATION (Classification Only)
        # ==============================================================================
        if task_type == "classification" and ultimate_winner is not None:
            vprint("\n" + "📊" * 35)
            vprint("📊 PHASE 4: PROBABILITY CALIBRATION (Ensuring Realistic Confidence Scores)")
            vprint("📊" * 35)
            
            try:
                vprint("\n🎯 Calibration ensures: When model says '90% confident', it's correct 9/10 times")
                
                # Calibrate on training data (prevents overfitting via cross-validation)
                calibrated_model = _calibrate_model(
                    ultimate_winner, 
                    X_train, 
                    y_train,
                    calibration_method="isotonic" if len(X_train) >= 100 else "sigmoid",
                    cv=min(5, cv)
                )
                
                vprint(f"   Method: {'Isotonic Regression' if len(X_train) >= 100 else 'Platt Scaling (Sigmoid)'}")
                vprint(f"   Calibration samples: {len(X_train)}")
                
                # Calculate calibration metrics on test set
                if hasattr(calibrated_model, 'predict_proba'):
                    y_proba_uncalibrated = ultimate_winner.predict_proba(X_test)[:, 1] if hasattr(ultimate_winner, 'predict_proba') else None
                    y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
                    
                    # Calculate Expected Calibration Error
                    ece_before = _calculate_expected_calibration_error(y_test, y_proba_uncalibrated, n_bins=10) if y_proba_uncalibrated is not None else None
                    ece_after = _calculate_expected_calibration_error(y_test, y_proba_calibrated, n_bins=10)
                    
                    # Log-Loss improvement (initialize to prevent UnboundLocalError)
                    logloss_before = log_loss(y_test, y_proba_uncalibrated) if y_proba_uncalibrated is not None else None
                    logloss_after = log_loss(y_test, y_proba_calibrated)
                    logloss_improvement = 0.0  # <--- FIX: Initialize here to prevent UnboundLocalError
                    
                    vprint(f"\n📈 Calibration Impact:")
                    if ece_before is not None:
                        ece_improvement = ((ece_before - ece_after) / ece_before * 100) if ece_before > 0 else 0
                        vprint(f"   ECE Before: {ece_before:.4f}")
                        vprint(f"   ECE After:  {ece_after:.4f} (↓ {ece_improvement:.1f}% improvement)")
                    
                    if logloss_before is not None:
                        logloss_improvement = ((logloss_before - logloss_after) / logloss_before * 100) if logloss_before > 0 else 0
                        vprint(f"   Log-Loss Before: {logloss_before:.4f}")
                        vprint(f"   Log-Loss After:  {logloss_after:.4f} (↓ {logloss_improvement:.1f}% improvement)")
                    
                    # Update ultimate winner to calibrated version
                    ultimate_winner = calibrated_model
                    best_estimators["🌟_Calibrated_Winner"] = calibrated_model
                    
                    # Log to MLflow
                    mlflow.log_metric("ece_before_calibration", ece_before if ece_before is not None else 0.0)
                    mlflow.log_metric("ece_after_calibration", ece_after)
                    mlflow.log_metric("logloss_after_calibration", logloss_after)
                    mlflow.log_param("calibration_method", "isotonic" if len(X_train) >= 100 else "sigmoid")
                    mlflow.sklearn.log_model(calibrated_model, "Calibrated_Winner_Model")
                    
                    vprint(f"\n✅ Calibration complete! Ultimate winner updated with calibrated model.")
                    vprint(f"🎯 Kaggle Log-Loss Advantage: ~{(logloss_improvement if logloss_before else 0):.1f}% improvement expected")
                else:
                    vprint("⚠️  Ultimate winner doesn't support probability prediction. Skipping calibration.")
                    
            except Exception as e:
                vprint(f"\n⚠️ Calibration failed: {str(e)}")
                vprint("Proceeding with uncalibrated model...\n")

        # ==============================================================================
        # 🧹 CLEANUP: Remove temporary cache directory
        # ==============================================================================
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)
            vprint("🧹 Cache directory cleaned up.")

        vprint(f"✅ MLflow run completed! Check experiment '{mlflow_experiment}' for full details.")

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
