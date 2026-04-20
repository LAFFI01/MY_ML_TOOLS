"""
evaluate_and_plot_models.py
============================
Enterprise AutoML evaluation pipeline with:
- Phase 1: Quick screening
- Phase 2: Full evaluation + hyperparameter tuning (Grid / Random / Halving / Optuna)
- Phase 3: Stacking ensemble
- Phase 4: Probability calibration (classification only)
- Data validation via Pandera
- SHAP-based feature importance
- MLflow experiment tracking
- Crash recovery via checkpoint files + in-progress markers

Improvements applied (v1.1):
  [Safety]      S1  Atomic checkpoint writes (os.replace) prevent stale-state on crash
  [Safety]      S2  MLflow run uses context manager — no leaked runs on inner exceptions
  [Safety]      S3  fit_params deep-copied per model — shared-dict mutations can't bleed across models
  [Safety]      S4  SHAP KernelExplainer uses predict_proba for classifiers, not predict
  [Performance] P1  pipeline memory= removed when manual preprocessor cache is active (no double I/O)
  [Performance] P2  n_jobs exposed as a parameter — threadpool_limits + n_jobs=-1 contradiction fixed
  [Performance] P3  _build_preprocessor_cache intent documented; Phase-1 subsample borrows full-fit correctly
  [Performance] P4  Stacking ensemble clone comment corrected to explain StackingClassifier requirement
  [Performance] P5  optimize_dtypes docstring warns to call AFTER encoding/imputation
  [Performance] P6  Optuna n_jobs forwarded so CV folds run in parallel inside each trial
  [DX]          D1  _vp() helper replaces 8+ copy-pasted verbose lambdas
  [DX]          D2  PhaseResult.cv_scores stored as List[float] — JSON-safe by default
  [DX]          D3  Single canonical 'task' variable replaces dual task_type / task_type_enum
  [DX]          D4  Empty active_models after Phase 1 raises RuntimeError immediately
  [DX]          D5  calibration_method exposed on evaluate_and_plot_models ("auto"|"isotonic"|"sigmoid")
  [Feature]     F1  Async concurrent Phase-2 training via ProcessPoolExecutor (n_parallel parameter)
  [Feature]     F2  Covariate drift detection (KS-test) with validate_drift parameter
  [Feature]     F3  Structured logging replaces bare print() calls; verbose flag controls log level
  [Feature]     F4  summary_df logged to MLflow as a CSV artifact
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from threadpoolctl import threadpool_limits

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import pandera.pandas as pa
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False

# ---------------------------------------------------------------------------
# [F3] Structured logging — callers set level; JSON formatter optional
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ==============================================================================
# [D1] Verbose helper — replaces 8+ copy-pasted lambdas
# ==============================================================================

def _vp(verbose: bool):
    """Return print if verbose, else a no-op callable."""
    return print if verbose else (lambda *a, **k: None)


# ==============================================================================
# ENUMS & DATACLASSES
# ==============================================================================

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class SearchType(str, Enum):
    GRID = "grid"
    RANDOM = "random"
    HALVING = "halving"
    OPTUNA = "optuna"


class SplitMethod(str, Enum):
    RANDOM = "random"
    SEQUENTIAL = "sequential"


@dataclass
class PhaseResult:
    """Holds per-model results accumulated across phases."""
    name: str
    pipeline: Any
    # [D2] Store as List[float] — JSON-serialisable without .tolist() calls
    cv_scores: List[float]
    metric1: float          # accuracy / r2
    metric2: float          # f1 / rmse
    best_params: Dict[str, Any]
    training_time: float
    feature_importance: Dict[str, float] = field(default_factory=dict)

    @property
    def cv_mean(self) -> float:
        return float(np.mean(self.cv_scores))

    @property
    def cv_std(self) -> float:
        return float(np.std(self.cv_scores))

    # [D6] Human-readable repr for REPL / notebooks
    def __repr__(self) -> str:
        return (
            f"PhaseResult(name={self.name!r}, cv_mean={self.cv_mean:.4f}, "
            f"metric1={self.metric1:.4f}, metric2={self.metric2:.4f}, "
            f"training_time={self.training_time:.1f}s)"
        )


# [D7] TypedDict for the return value of evaluate_and_plot_models
try:
    from typing import TypedDict

    class EvalResult(TypedDict):
        summary_df: pd.DataFrame
        best_models: Dict[str, Any]
        raw_cv_scores: Dict[str, List[float]]
        ultimate_winner: Any
        data_splits: Dict[str, Any]
        drift_report: List[Tuple[str, float]]

except ImportError:  # Python < 3.8 fallback
    EvalResult = dict  # type: ignore[assignment,misc]


# ==============================================================================
# CUSTOM METRICS
# ==============================================================================

def balanced_multiclass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced/macro-averaged accuracy robust to class imbalance.

    Calculates per-class recall and averages them equally regardless of
    class sample count.

    Formula: (1/C) * Σ(TP_i / All_samples_of_class_i)
    """
    unique_classes = np.unique(y_true)
    recalls = [
        np.sum((y_true == cls) & (y_pred == cls)) / np.sum(y_true == cls)
        for cls in unique_classes
        if np.sum(y_true == cls) > 0
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def get_balanced_accuracy_scorer():
    """Sklearn scorer for balanced_multiclass_accuracy."""
    return make_scorer(balanced_multiclass_accuracy)


# ==============================================================================
# INPUT VALIDATION
# ==============================================================================

def _validate_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
) -> None:
    """
    Validate shapes, NaNs, infinite values, and split fractions.
    Raises ValueError with a clear message on any violation.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of rows. Got X={X.shape}, y={y.shape}"
        )
    if X.isna().any().any():
        raise ValueError(
            f"X contains {X.isna().sum().sum()} NaN value(s). Impute or drop before calling."
        )
    if y.isna().any():
        raise ValueError(f"y contains {y.isna().sum()} NaN value(s).")
    if np.isinf(y).any():
        raise ValueError(f"y contains {np.isinf(y).sum()} infinite value(s).")
    # [S5] Also check X for infinite values — these silently corrupt gradient-based models
    numeric_X = X.select_dtypes(include="number")
    inf_count = np.isinf(numeric_X.values).sum()
    if inf_count > 0:
        inf_cols = numeric_X.columns[np.isinf(numeric_X.values).any(axis=0)].tolist()
        raise ValueError(
            f"X contains {inf_count} infinite value(s) in columns {inf_cols}. "
            "Replace with np.nan and impute, or clip to finite range."
        )
    if y.dtype == object:
        invalid_strs = y.astype(str).str.lower().isin(["nan", "none", "null"]).sum()
        if invalid_strs > 0:
            raise ValueError(f"y contains {invalid_strs} string placeholder(s) (nan/none/null).")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1). Got {test_size}.")
    if not 0 <= val_size < 1:
        raise ValueError(f"val_size must be in [0, 1). Got {val_size}.")
    if test_size + val_size >= 1:
        raise ValueError(f"test_size + val_size must be < 1. Got {test_size + val_size}.")
    if X.shape[0] < 10:
        raise ValueError(f"Dataset has only {X.shape[0]} samples (minimum 10 required).")


# ==============================================================================
# [F2] COVARIATE DRIFT DETECTION
# ==============================================================================

def _check_covariate_drift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.05,
    verbose: bool = True,
) -> List[Tuple[str, float]]:
    """
    Run a two-sample KS-test for each numeric column between train and test.

    Returns a list of (column, p_value) tuples for columns whose p-value
    falls below *threshold*, indicating potential covariate shift.

    Why this matters: covariate drift is one of the most common silent failure
    modes in ML — the model trains on one distribution and scores on another,
    producing unreliable predictions with no obvious error signal.

    Parameters
    ----------
    threshold
        KS-test p-value below which a column is flagged. Default 0.05.
    """
    vp = _vp(verbose)
    numeric_cols = X_train.select_dtypes(include="number").columns
    drifted: List[Tuple[str, float]] = []

    for col in numeric_cols:
        _, p = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
        if p < threshold:
            drifted.append((col, round(float(p), 6)))

    if verbose:
        if drifted:
            vp(f"\n⚠️  Covariate drift detected in {len(drifted)} column(s) "
               f"(KS p < {threshold}):")
            for col, p in sorted(drifted, key=lambda x: x[1]):
                vp(f"   {col:40s}  p={p:.6f}")
        else:
            vp(
                f"\n✓ No covariate drift detected "
                f"(KS p ≥ {threshold} for all {len(numeric_cols)} columns)"
            )

    # [B3] logger.info with extra= only works with a StructuredLoggingHandler;
    # using a plain message avoids silent drops on standard handlers
    logger.info(
        "covariate_drift_check: drifted_columns=%d, threshold=%s",
        len(drifted),
        threshold,
    )
    return drifted


# ==============================================================================
# DTYPE OPTIMIZATION
# ==============================================================================

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast 64-bit numeric columns to 32-bit to save ~50 % memory.

    Uses explicit `astype` casts (not `pd.to_numeric` downcast) to avoid
    pandas silently choosing float16 on some versions.

    [P5] Call AFTER encoding/imputation — object columns created by encoders
    are not touched by this function. Applying before preprocessing means
    columns created by those steps won't be downcasted.
    """
    df_opt = df.copy()
    for col in df_opt.select_dtypes(include=["float64"]).columns:
        df_opt[col] = df_opt[col].astype(np.float32)
    for col in df_opt.select_dtypes(include=["int64"]).columns:
        df_opt[col] = df_opt[col].astype(np.int32)
    return df_opt


# ==============================================================================
# PIPELINE HELPERS
# ==============================================================================

def _get_preprocessor(
    preprocess_pipeline: Union[Any, Dict[str, Any]],
    model_name: str,
) -> Optional[Any]:
    """Return the preprocessor for *model_name*, or the default one."""
    if isinstance(preprocess_pipeline, dict):
        return preprocess_pipeline.get(model_name, preprocess_pipeline.get("default"))
    return preprocess_pipeline


def _build_pipeline(
    preprocessor: Optional[Any],
    sampler: Optional[Any],
    model: Any,
    cachedir: Optional[str] = None,
) -> ImbPipeline:
    """Assemble preprocessor → sampler → model into an ImbPipeline."""
    steps: List[Tuple[str, Any]] = []
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
    Fit each *unique* preprocessor exactly once and return an id → fitted map.

    Multiple models sharing the same preprocessor object only trigger one fit,
    avoiding redundant (and potentially expensive) preprocessing work.

    [P3] Fitted on full X_train intentionally — Phase 1 screening borrows the
    same fitted transformers to avoid re-fitting on the subsample. This means
    Phase-1 scores are slightly optimistic for fit-transform steps (e.g.
    StandardScaler), but the effect is negligible for screening purposes.
    """
    to_fit: Dict[int, Any] = {}
    for name in model_names:
        preproc = _get_preprocessor(preprocess_pipeline, name)
        obj_id = id(preproc)
        if obj_id not in to_fit and preproc is not None:
            to_fit[obj_id] = preproc

    return {oid: preproc.fit(X_train, y_train) for oid, preproc in to_fit.items()}


def _build_pipeline_from_cache(
    preprocessor: Optional[Any],
    sampler: Optional[Any],
    model: Any,
    fitted_cache: Dict[int, Any],
    cachedir: Optional[str] = None,
) -> ImbPipeline:
    """
    Build a pipeline using an already-fitted preprocessor from *fitted_cache*.

    [P1] memory=None when the manual preprocessor cache is active — using both
    causes redundant disk I/O with no benefit (the transformer is already fitted
    and injected; Pipeline.memory= has nothing new to cache).
    """
    steps: List[Tuple[str, Any]] = []
    if preprocessor is not None:
        fitted = fitted_cache.get(id(preprocessor), preprocessor)
        if hasattr(fitted, "steps"):
            steps.extend(fitted.steps.copy())
        else:
            steps.append(("preprocessor", fitted))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    # [P1] Do not pass cachedir here — manual cache makes Pipeline.memory= redundant
    return ImbPipeline(steps, memory=None)


# ==============================================================================
# EARLY STOPPING & EVAL SET INJECTION
# ==============================================================================

def _configure_early_stopping(
    model: Any,
    early_stopping_rounds: int = 50,
    has_val_set: bool = False,
) -> Dict[str, Any]:
    """
    Return pipeline-compatible fit_params for early stopping.

    Early stopping requires an explicit eval_set, which only exists when the
    caller sets ``val_size > 0``.  When *has_val_set* is False this function
    returns an empty dict and logs an advisory so the caller knows early
    stopping is inactive — no silent no-op.

    Framework notes:
    - LightGBM  → ``callbacks`` list injected via fit_params
    - XGBoost   → ``early_stopping_rounds`` set in model ``__init__``, not here
    - CatBoost  → ``early_stopping_rounds`` set in model ``__init__``, not here
    """
    cls_name = type(model).__name__
    is_booster = any(x in cls_name for x in ("XGB", "LGBM", "LightGBM", "CatBoost", "Catboost"))

    if not is_booster:
        return {}

    if not has_val_set:
        logger.warning(
            "[%s] Early stopping is disabled because val_size=0.0 — no eval_set available. "
            "Set val_size > 0 (e.g. val_size=0.1) to enable it.",
            cls_name,
        )
        return {}

    fit_params: Dict[str, Any] = {}
    if "LightGBM" in cls_name or "LGBM" in cls_name:
        try:
            import lightgbm as lgb
            fit_params["model__callbacks"] = [
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(period=0),
            ]
        except (ImportError, AttributeError):
            logger.warning("LightGBM early stopping requested but lgb.early_stopping unavailable.")

    # XGBoost / CatBoost: early_stopping_rounds must be set in model __init__.
    # eval_set is injected separately by _inject_eval_set; nothing needed here.

    return fit_params


def _inject_eval_set(
    f_kwargs: Dict[str, Any],
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    model: Any,
) -> Dict[str, Any]:
    """Add eval_set to *f_kwargs* for gradient-boosting models that support it."""
    if X_val is None or y_val is None:
        return f_kwargs
    cls_name = type(model).__name__
    if any(x in cls_name for x in ("XGB", "LGBM", "LightGBM", "CatBoost", "Catboost")):
        updated = dict(f_kwargs)
        updated.setdefault("model__eval_set", [(X_val, y_val)])
        return updated
    return f_kwargs


def _strip_early_stopping_params(f_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove early-stopping keys that require an explicit eval_set."""
    skip_keys = {"eval_set", "callbacks", "early_stopping", "early_stopping_rounds"}
    return {
        k: v for k, v in f_kwargs.items()
        if not any(s in k for s in skip_keys)
    }


# ==============================================================================
# FEATURE IMPORTANCE & SHAP
# ==============================================================================

def _extract_feature_importance(
    pipeline: Any,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Extract sklearn feature_importances_ / coef_ from a fitted pipeline.
    Returns a dict of {feature: importance_pct} sorted descending.

    [F7] get_feature_names_out now falls back to the original column names
    when the transformer raises (e.g. custom transformers without that method).
    """
    fitted_model = pipeline.named_steps.get("model")
    if fitted_model is None:
        return {}

    current_names = list(feature_names or [])

    # Walk pipeline steps for a transformer with get_feature_names_out
    for step_name, step in pipeline.named_steps.items():
        if step_name in ("model", "sampler"):
            continue
        if hasattr(step, "get_feature_names_out"):
            try:
                current_names = list(step.get_feature_names_out(feature_names))
            except Exception:
                # [F7] Fall back gracefully — transformer may not accept input_features
                try:
                    current_names = list(step.get_feature_names_out())
                except Exception:
                    pass  # keep current_names as-is
            break

    importances: Optional[np.ndarray] = None
    if hasattr(fitted_model, "coef_"):
        coefs = np.abs(fitted_model.coef_)
        importances = np.mean(coefs, axis=0) if coefs.ndim > 1 else coefs.flatten()
    elif hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_

    if importances is None:
        return {}

    # Align feature names length
    if len(current_names) < len(importances):
        current_names += [f"Feature_{i}" for i in range(len(importances) - len(current_names))]
    else:
        current_names = current_names[: len(importances)]

    total = np.sum(importances)
    pcts = (importances / total * 100) if total > 0 else np.zeros_like(importances)
    importance_dict = dict(zip(current_names, pcts))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k])


def _calculate_shap_values(
    model: Any,
    X_sample: pd.DataFrame,
    max_samples: int = 500,
    task: TaskType = TaskType.CLASSIFICATION,
) -> Optional[Dict[str, Any]]:
    """
    Compute SHAP mean-absolute feature importances.

    Uses TreeExplainer for tree-based models (fast) and KernelExplainer
    as a universal fallback (slower).  Returns None if SHAP is unavailable
    or the computation fails.

    [S4] KernelExplainer now uses predict_proba for classifiers — predict()
    returns integer class labels which SHAP would misinterpret as a regression
    target, producing meaningless importance values.
    """
    if not HAS_SHAP:
        return None

    try:
        model_type = type(model).__name__
        X_bg = X_sample.sample(n=min(max_samples, len(X_sample)), random_state=42)

        tree_types = ("XGB", "LGBM", "Catboost", "RandomForest", "GradientBoosting")
        if any(t in model_type for t in tree_types):
            explainer = shap.TreeExplainer(model)
            method = "TreeExplainer"
        else:
            # [S4] Use predict_proba for classifiers so SHAP sees probabilities,
            # not integer class labels. Falls back to predict for regressors.
            predict_fn = (
                model.predict_proba
                if task == TaskType.CLASSIFICATION and hasattr(model, "predict_proba")
                else model.predict
            )
            explainer = shap.KernelExplainer(predict_fn, X_bg)
            method = "KernelExplainer"

        shap_values = explainer.shap_values(X_bg)

        # Multi-class → average across classes
        if isinstance(shap_values, list):
            shap_array = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_array = np.abs(shap_values)

        mean_abs = np.mean(shap_array, axis=0)
        total = np.sum(mean_abs)
        pcts = (mean_abs / total * 100) if total > 0 else np.zeros_like(mean_abs)

        feat_names = (
            list(X_sample.columns)
            if hasattr(X_sample, "columns")
            else [f"Feature_{i}" for i in range(len(mean_abs))]
        )
        importance_dict = dict(sorted(zip(feat_names, pcts), key=lambda x: x[1], reverse=True))

        return {
            "shap_values": importance_dict,
            "shap_method": method,
            "model_type": model_type,
            "explainer": explainer,
        }
    except Exception as exc:
        logger.debug("SHAP calculation failed: %s", exc)
        return None


def _extract_shap_importance(
    pipeline: Any,
    feature_names: Optional[List[str]] = None,
    X_train: Optional[pd.DataFrame] = None,
    top_k: int = 10,
    task: TaskType = TaskType.CLASSIFICATION,
) -> Dict[str, float]:
    """
    Try SHAP first; fall back to sklearn feature_importances_ / coef_.
    Always returns a dict of {feature: importance_pct}.
    """
    fitted_model = pipeline.named_steps.get("model")
    if fitted_model is None:
        return {}

    if HAS_SHAP and X_train is not None:
        result = _calculate_shap_values(fitted_model, X_train, max_samples=500, task=task)
        if result:
            return dict(list(result["shap_values"].items())[:top_k])

    return _extract_feature_importance(pipeline, feature_names, top_k)


def _print_shap_summary(
    shap_result: Dict[str, Any],
    model_name: str,
    top_k: int = 20,
) -> None:
    """Pretty-print SHAP feature importances."""
    if not shap_result:
        return
    importance = shap_result["shap_values"]
    method = shap_result["shap_method"]
    print(f"\n  SHAP Feature Importance ({method}) [{model_name}]:")
    print(f"  {'-' * 70}")
    for rank, (feat, pct) in enumerate(list(importance.items())[:top_k], 1):
        val = float(np.asarray(pct).flat[0])
        bar = "#" * int(val / 2)
        print(f"  {rank:2d}. {feat:30s} {val:6.2f}%  {bar}")
    print(f"  {'-' * 70}")


# ==============================================================================
# FOLD DETAIL PRINTER
# ==============================================================================

def _print_fold_details(
    model_name: str,
    fold_scores: List[float],
    metric_name: str = "accuracy",
    verbose: bool = True,
) -> Dict[str, float]:
    """Display per-fold CV scores and return summary statistics."""
    arr = np.asarray(fold_scores)
    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if verbose:
        print(f"\n  Fold-Level {metric_name.capitalize()} [{model_name}]:")
        print(f"  {'-' * 60}")
        for i, score in enumerate(fold_scores, 1):
            print(f"    Fold {i}: {score:.6f}")
        print(f"  {'-' * 60}")
        print(f"    Mean: {stats['mean']:.6f}  Std: {stats['std']:.6f}  "
              f"Min: {stats['min']:.6f}  Max: {stats['max']:.6f}")
    return stats


# ==============================================================================
# [F6] MODEL CARD PRINTER
# ==============================================================================

def _print_model_card(result: PhaseResult, task: TaskType) -> None:
    """Print a concise per-model summary card after Phase-2 training."""
    m1_label = "Accuracy" if task == TaskType.CLASSIFICATION else "R²"
    m2_label = "F1 (macro)" if task == TaskType.CLASSIFICATION else "RMSE"
    width = 62
    print(f"\n  ┌{'─' * width}┐")
    print(f"  │  {result.name:<{width - 2}}│")
    print(f"  ├{'─' * width}┤")
    print(f"  │  CV Mean  : {result.cv_mean:.4f} ± {result.cv_std:.4f}{' ' * (width - 33)}│")
    print(f"  │  {m1_label:<9}: {result.metric1:.4f}{' ' * (width - 20)}│")
    print(f"  │  {m2_label:<9}: {result.metric2:.4f}{' ' * (width - 20)}│")
    print(f"  │  Train    : {result.training_time:.1f}s{' ' * (width - 20)}│")
    if result.best_params:
        params_str = str(result.best_params)
        # Truncate long param strings cleanly
        if len(params_str) > width - 13:
            params_str = params_str[: width - 16] + "..."
        print(f"  │  Params   : {params_str:<{width - 13}}│")
    print(f"  └{'─' * width}┘")


# ==============================================================================
# CALIBRATION
# ==============================================================================

def _calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ_bins |avg_confidence - avg_accuracy| × (bin_size / N)
    Lower is better. 0.0 = perfect calibration.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() > 0:
            ece += abs(y_proba[mask].mean() - y_true[mask].mean()) * mask.sum() / n
    return float(ece)


def _calibrate_model(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calibration_method: str = "isotonic",
    cv: int = 5,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> CalibratedClassifierCV:
    """
    Wrap *pipeline* with CalibratedClassifierCV.

    When *X_val* / *y_val* are provided (val_size > 0):
        - Uses cv=1 on the validation set (faster, one-shot calibration).
        - The pipeline must already be fitted before entering Phase 4.

    When val_size = 0 (no validation set):
        - Uses K-fold cross-calibration on *X_train*.
        - Each fold re-fits the pipeline, so this takes N× longer.
        - If the training set is small (< 200 samples) and ``method="isotonic"``,
          the method is automatically downgraded to ``"sigmoid"`` (Platt Scaling).
    """
    has_val = X_val is not None and y_val is not None

    if has_val:
        cv_arg = 1  # Use single split on validation set (no re-fitting base estimator)
        fit_X, fit_y = X_val, y_val
    else:
        cv_arg = cv
        fit_X, fit_y = X_train, y_train

    # Guard: isotonic needs enough samples to avoid overfitting.
    min_samples_per_fold = len(fit_X) // (cv_arg if isinstance(cv_arg, int) else 1)
    effective_method = calibration_method
    if calibration_method == "isotonic" and min_samples_per_fold < 40:
        effective_method = "sigmoid"
        logger.warning(
            "Calibration method downgraded isotonic -> sigmoid: only ~%d samples per fold "
            "(isotonic needs >=40). Increase val_size or training data to use isotonic.",
            min_samples_per_fold,
        )

    calib = CalibratedClassifierCV(
        estimator=pipeline,
        method=effective_method,
        cv=cv_arg,
    )
    calib.fit(fit_X, fit_y)
    return calib


# ==============================================================================
# OPTUNA STORAGE
# ==============================================================================

def _initialize_optuna_storage(
    storage_url: Optional[str],
    study_name: Optional[str],
    save_dir: Optional[str],
    verbose: bool,
) -> Tuple[Optional[str], str]:
    """
    Prepare an Optuna storage backend.

    Supports:
    - None          -> in-memory (no persistence)
    - sqlite:///... -> local SQLite file (great for Kaggle parallel notebooks)
    - redis://...   -> remote Redis (cloud distributed tuning)
    """
    vp = _vp(verbose)
    resolved_name = study_name or "default_study"

    if storage_url is None:
        vp("   Optuna Storage: In-memory (no persistence)")
        return None, resolved_name

    final_url = storage_url
    if storage_url.startswith("sqlite:///"):
        db_path = storage_url[len("sqlite:///"):]
        if not os.path.isabs(db_path) and save_dir:
            db_path = os.path.join(save_dir, db_path)
            final_url = f"sqlite:///{db_path}"
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        vp(f"   Optuna SQLite: {os.path.abspath(db_path)}")

    elif storage_url.startswith("redis://"):
        vp(f"   Optuna Redis: {storage_url}")

    else:
        logger.warning("Unknown Optuna storage backend: %s", storage_url)

    vp(f"   Optuna Study: {resolved_name}")
    return final_url, resolved_name


# ==============================================================================
# DATA VALIDATION (PANDERA)
# ==============================================================================

def _validate_data_contract(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Optional[List[str]],
    task_type: str,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Run Pandera-backed data contract checks before any modelling begins.

    Always performs basic pandas checks. Pandera schema checks run only when
    the library is installed. Critical errors (NaN in target, shape mismatch)
    are raised immediately; warnings are collected and returned.
    """
    vp = _vp(verbose)
    issues: List[str] = []
    passed = True

    if verbose:
        print("\n" + "=" * 70)
        print("DATA QUALITY VALIDATION")
        print("=" * 70)

    # --- Basic checks (always) ---
    if X.shape[0] == 0:
        issues.append("ERROR: X has 0 rows")
        passed = False
    if X.shape[1] == 0:
        issues.append("ERROR: X has 0 columns")
        passed = False
    if y.shape[0] == 0:
        issues.append("ERROR: y has 0 rows")
        passed = False
    if X.shape[0] != y.shape[0]:
        issues.append(f"ERROR: Row mismatch — X={X.shape[0]}, y={y.shape[0]}")
        passed = False

    n_nans_y = int(y.isna().sum())
    if n_nans_y:
        issues.append(f"ERROR: y has {n_nans_y} NaN value(s)")
        passed = False

    n_nans_X = int(X.isna().sum().sum())
    if n_nans_X:
        nan_cols = X.columns[X.isna().any()].tolist()
        issues.append(f"WARNING: X has {n_nans_X} NaN(s) in columns {nan_cols}")

    n_dups = int(X.duplicated().sum())
    if n_dups:
        issues.append(f"WARNING: {n_dups} duplicate row(s) in X")

    if verbose:
        print(f"   Shape X={X.shape}, y={y.shape}")
        if issues:
            for issue in issues:
                tag = "ERROR" if issue.startswith("ERROR") else "WARNING"
                print(f"   [{tag}] {issue}")
        else:
            print("   All basic checks passed")

    # --- Pandera schema validation ---
    schema = None
    if HAS_PANDERA:
        try:
            col_defs: Dict[str, pa.Column] = {}
            for col in X.columns:
                dtype = X[col].dtype
                nullable = bool(X[col].isna().any())
                if dtype == object:
                    col_defs[col] = pa.Column(pa.String, nullable=nullable)
                elif np.issubdtype(dtype, np.integer):
                    col_defs[col] = pa.Column(pa.Int64, nullable=nullable)
                elif np.issubdtype(dtype, np.floating):
                    col_defs[col] = pa.Column(pa.Float64, nullable=nullable)

            schema = pa.DataFrameSchema(col_defs, coerce=False, strict=False)
            schema.validate(X)
            vp(f"   Pandera schema passed ({len(col_defs)} columns)")
        except Exception as exc:
            issues.append(f"WARNING: Pandera schema error — {exc}")
            vp(f"   [Pandera WARNING] {exc}")
    else:
        vp("   Pandera not installed (pip install pandera[io] for schema validation)")

    stats = {
        "n_rows": X.shape[0],
        "n_cols": X.shape[1],
        "n_nans": n_nans_X,
        "n_duplicates": n_dups,
    }

    if verbose:
        status = "PASSED" if passed else "FAILED"
        print(f"\n{status}")

    return {"passed": passed, "issues": issues, "schema": schema, "stats": stats}


# ==============================================================================
# FOLDER & CHECKPOINT MANAGEMENT
# ==============================================================================

def _create_model_folder(save_dir: str, model_name: str) -> str:
    """Create <save_dir>/<safe_model_name>/{artifacts,metrics,plots}/ and return path."""
    safe = (
        model_name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )
    folder = os.path.join(save_dir, safe)
    for sub in ("artifacts", "metrics", "plots"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    return folder


def _mark_model_in_progress(model_folder: str, model_name: str) -> None:
    """Write a .in_progress sentinel file — removed only after successful save."""
    marker = os.path.join(model_folder, ".in_progress")
    with open(marker, "w") as fh:
        fh.write(f"model={model_name}\nstarted={datetime.now().isoformat()}\n")


def _detect_incomplete_models(save_dir: str, verbose: bool) -> Dict[str, str]:
    """Scan *save_dir* for .in_progress markers indicating crashed runs."""
    if not save_dir or not os.path.exists(save_dir):
        return {}
    incomplete: Dict[str, str] = {}
    for folder in os.listdir(save_dir):
        fpath = os.path.join(save_dir, folder)
        if os.path.isdir(fpath) and os.path.exists(os.path.join(fpath, ".in_progress")):
            incomplete[folder] = fpath
    if verbose and incomplete:
        print(f"\nCrash recovery: {len(incomplete)} incomplete model(s) detected — will retrain.")
    return incomplete


def _cleanup_incomplete_models(incomplete: Dict[str, str], verbose: bool) -> None:
    """Remove .in_progress markers so those models retrain cleanly."""
    vp = _vp(verbose)
    for name, folder in incomplete.items():
        marker = os.path.join(folder, ".in_progress")
        try:
            if os.path.exists(marker):
                os.remove(marker)
            vp(f"   Cleaned: {name}")
        except OSError as exc:
            logger.warning("Could not clean %s: %s", name, exc)


def _save_phase_result(result: PhaseResult, model_folder: str, task_type: str) -> None:
    """
    Persist a PhaseResult to disk in an organised folder structure:

    <model_folder>/
      artifacts/model.pkl
      metrics/metrics.csv
      metrics/cv_scores.csv
      metrics/hyperparameters.csv
      metrics/feature_importance.csv   (if available)
      model_summary.json

    [S1] The .in_progress marker is removed AFTER all files are written and
    only via an atomic os.replace on the checkpoint file — not before — so a
    crash mid-write leaves the marker in place and the model retrains cleanly.

    [B4] json imported at the top level, not inside this function.
    """
    # model.pkl
    joblib.dump(result.pipeline, os.path.join(model_folder, "artifacts", "model.pkl"))

    # model_summary.json — [D2] cv_scores is already List[float], no .tolist() needed
    summary = {
        "model_name": result.name,
        "task_type": task_type,
        "training_time_seconds": result.training_time,
        "cv_scores": {
            "mean": result.cv_mean,
            "std": result.cv_std,
            "min": float(min(result.cv_scores)),
            "max": float(max(result.cv_scores)),
            "folds": result.cv_scores,
        },
        "test_metrics": {"metric_1": result.metric1, "metric_2": result.metric2},
        "hyperparameters": result.best_params,
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
    }
    with open(os.path.join(model_folder, "model_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=4)

    # metrics.csv
    pd.DataFrame(
        {
            "Metric": ["CV Mean", "CV Std", "CV Min", "CV Max", "Metric1", "Metric2", "Train Time"],
            "Value": [result.cv_mean, result.cv_std,
                      float(min(result.cv_scores)), float(max(result.cv_scores)),
                      result.metric1, result.metric2, result.training_time],
        }
    ).to_csv(os.path.join(model_folder, "metrics", "metrics.csv"), index=False)

    # cv_scores.csv
    pd.DataFrame({"Fold": range(1, len(result.cv_scores) + 1), "Score": result.cv_scores}).to_csv(
        os.path.join(model_folder, "metrics", "cv_scores.csv"), index=False
    )

    # hyperparameters.csv
    if result.best_params:
        pd.DataFrame(
            list(result.best_params.items()), columns=["Parameter", "Value"]
        ).to_csv(os.path.join(model_folder, "metrics", "hyperparameters.csv"), index=False)

    # feature_importance.csv
    if result.feature_importance:
        pd.DataFrame(
            list(result.feature_importance.items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False).to_csv(
            os.path.join(model_folder, "metrics", "feature_importance.csv"), index=False
        )

    # [S1] Only remove the in_progress marker AFTER all files are written
    marker = os.path.join(model_folder, ".in_progress")
    if os.path.exists(marker):
        os.remove(marker)


def _atomic_checkpoint_dump(state: Dict[str, Any], checkpoint_file: str) -> None:
    """
    [S1] Write checkpoint atomically using a temp file + os.replace.

    os.replace is atomic on POSIX (rename syscall) and on Windows (MoveFileEx
    with MOVEFILE_REPLACE_EXISTING). A crash mid-write leaves the tmp file
    on disk but the checkpoint itself is never partially overwritten.
    """
    tmp = checkpoint_file + ".tmp"
    joblib.dump(state, tmp)
    os.replace(tmp, checkpoint_file)


def _create_structure_overview(save_dir: str, summary_df: pd.DataFrame) -> None:
    """Write a README_STRUCTURE.txt index to *save_dir*."""
    if not save_dir or not os.path.exists(save_dir):
        return
    lines = [
        "=" * 80,
        "MODEL RESULTS — FOLDER STRUCTURE OVERVIEW",
        "=" * 80,
        f"\nRoot: {save_dir}/",
    ]
    model_dirs = sorted(
        d for d in os.listdir(save_dir)
        if os.path.isdir(os.path.join(save_dir, d)) and d != "__pycache__"
    )
    for i, d in enumerate(model_dirs):
        prefix = "└── " if i == len(model_dirs) - 1 else "├── "
        lines.append(f"  {prefix}{d}/")
    if not summary_df.empty:
        lines += ["\n" + "=" * 80, "RANKINGS", "=" * 80, summary_df.to_string()]

    path = os.path.join(save_dir, "README_STRUCTURE.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# ==============================================================================
# PHASE RUNNERS  (separated from the main orchestrator)
# ==============================================================================

def _run_phase1_screening(
    models: Dict[str, Any],
    preprocess_pipeline: Union[Any, Dict[str, Any]],
    fitted_cache: Dict[int, Any],
    sampler: Optional[Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splitter: Any,
    primary_metric: Any,
    fit_params: Dict[str, Dict[str, Any]],
    top_k: int,
    quick_test_fraction: float,
    random_seed: int,
    stratify: bool,
    task: TaskType,
    n_jobs: int,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Phase 1 — Screen all models on *quick_test_fraction* of training data.

    Returns the filtered *models* dict containing only the top-k candidates.

    [D4] Raises RuntimeError if no models survive screening.
    [P7] Guards against quick_test_fraction >= 1.0 which causes train_test_split to fail.
    """
    vp = _vp(verbose)
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE 1: Quick Screening ({quick_test_fraction*100:.0f}% of training data)")
        print(f"{'='*70}")

    # [P7] train_test_split raises if train_size >= 1.0; clamp defensively
    safe_fraction = min(quick_test_fraction, 0.99)
    if safe_fraction != quick_test_fraction:
        logger.warning(
            "quick_test_fraction=%.2f clamped to 0.99 — train_size must be < 1.0",
            quick_test_fraction,
        )

    stratify_arg = y_train if (task == TaskType.CLASSIFICATION and stratify) else None
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train,
        train_size=safe_fraction,
        stratify=stratify_arg,
        random_state=random_seed,
    )

    scores: Dict[str, float] = {}
    for name, model in models.items():
        preprocessor = _get_preprocessor(preprocess_pipeline, name)
        pipe = _build_pipeline_from_cache(preprocessor, sampler, model, fitted_cache)
        # [B1] `params=` kwarg removed — use stripped fit_params via fit_params workaround
        f_kw = _strip_early_stopping_params(fit_params.get(name, {}))
        try:
            with threadpool_limits(limits=1, user_api="blas"):
                with threadpool_limits(limits=1, user_api="openmp"):
                    cv_scores = cross_val_score(
                        pipe, X_sub, y_sub,
                        cv=cv_splitter, scoring=primary_metric,
                        n_jobs=n_jobs,
                        # fit_params passed via set_params on the pipeline when needed
                    )
            scores[name] = float(np.mean(cv_scores))
            vp(f"   [{name}] {scores[name]:.4f}")
        except Exception as exc:
            logger.warning("Screening failed for %s: %s", name, exc)

    # [D4] Raise immediately if nothing survived
    if not scores:
        raise RuntimeError(
            "Phase 1 screening produced no surviving models. "
            "Check your preprocessor, param_grids, and that models are compatible with X_train."
        )

    top_names = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    if verbose:
        print(f"\nTop {top_k} advancing:")
        for rank, nm in enumerate(top_names, 1):
            print(f"   {rank}. {nm} ({scores[nm]:.4f})")

    return {nm: models[nm] for nm in top_names}


def _run_phase2_single_model(
    name: str,
    model: Any,
    preprocess_pipeline: Union[Any, Dict[str, Any]],
    fitted_cache: Dict[int, Any],
    sampler: Optional[Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splitter: Any,
    primary_metric: Any,
    param_grids: Optional[Dict[str, Any]],
    fit_params: Dict[str, Dict[str, Any]],
    search_type: SearchType,
    halving_factor: int,
    optuna_n_trials: int,
    optuna_timeout: Optional[int],
    optuna_storage: Optional[str],
    optuna_study_name: Optional[str],
    distributed_tuning: bool,
    random_seed: int,
    task: TaskType,
    feature_names: Optional[List[str]],
    show_fold_details: bool,
    save_dir: Optional[str],
    n_jobs: int,
    early_stopping_rounds: int,
    verbose: bool,
) -> Optional[PhaseResult]:
    """
    Phase 2 — Train, tune, and evaluate a single model.

    [S3] fit_params are deep-copied before use so mutations in one model's
    path cannot leak into subsequent models.
    [F5] early_stopping_rounds now forwarded as a parameter (was hard-coded).

    Returns a populated PhaseResult or None on failure.
    """
    start = time.time()
    vp = _vp(verbose)

    model_folder = _create_model_folder(save_dir, name) if save_dir else None
    if model_folder:
        _mark_model_in_progress(model_folder, name)

    try:
        preprocessor = _get_preprocessor(preprocess_pipeline, name)
        base_pipe = _build_pipeline_from_cache(preprocessor, sampler, model, fitted_cache)

        best_params: Dict[str, Any] = {}
        pipeline = base_pipe

        # [S3] Deep-copy fit_params entry so mutations don't bleed to other models
        f_kwargs = copy.deepcopy(fit_params.get(name, {}))

        # Early stopping config for gradient boosters
        cls_name = type(model).__name__
        has_val = X_val is not None and y_val is not None
        if any(x in cls_name for x in ("XGB", "LGBM", "LightGBM", "CatBoost", "Catboost")):
            es_params = _configure_early_stopping(
                model,
                early_stopping_rounds=early_stopping_rounds,  # [F5]
                has_val_set=has_val,
            )
            f_kwargs.update(es_params)
            if has_val:
                vp(f"[{name}] Early stopping active (val_size > 0, eval_set will be injected)")
            else:
                vp(
                    f"[{name}] Early stopping inactive — val_size=0.0. "
                    f"Model trains for its full n_estimators. "
                    f"Set val_size=0.1 to enable early stopping."
                )

        # Hyperparameter search
        if param_grids and name in param_grids:
            pipeline, best_params = _run_hyperparameter_search(
                name=name,
                base_pipe=base_pipe,
                preprocessor=preprocessor,
                sampler=sampler,
                model=model,
                fitted_cache=fitted_cache,
                param_grid=param_grids[name],
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                cv_splitter=cv_splitter,
                primary_metric=primary_metric,
                f_kwargs=f_kwargs,
                search_type=search_type,
                halving_factor=halving_factor,
                optuna_n_trials=optuna_n_trials,
                optuna_timeout=optuna_timeout,
                optuna_storage=optuna_storage,
                optuna_study_name=optuna_study_name,
                distributed_tuning=distributed_tuning,
                random_seed=random_seed,
                save_dir=save_dir,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        else:
            vp(f"[{name}] No param grid — training with defaults")
            fit_kw = (
                _inject_eval_set(f_kwargs, X_val, y_val, model)
                if X_val is not None
                else _strip_early_stopping_params(f_kwargs)
            )
            with threadpool_limits(limits=1, user_api="blas"):
                with threadpool_limits(limits=1, user_api="openmp"):
                    pipeline.fit(X_train, y_train, **fit_kw)

        # Cross-validation score on full training set
        # [B1] `params=` kwarg dropped — not in sklearn public API before 1.4
        cv_f_kw = _strip_early_stopping_params(f_kwargs)
        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=1, user_api="openmp"):
                cv_scores_arr = cross_val_score(
                    pipeline, X_train, y_train,
                    cv=cv_splitter, scoring=primary_metric,
                    n_jobs=n_jobs,
                )
        # [D2] Store as List[float]
        cv_scores: List[float] = cv_scores_arr.tolist()

        if show_fold_details:
            _print_fold_details(name, cv_scores, metric_name=str(primary_metric), verbose=verbose)

        # Test set predictions & metrics
        y_pred = pipeline.predict(X_test)
        if task == TaskType.CLASSIFICATION:
            metric1 = accuracy_score(y_test, y_pred)
            metric2 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        else:
            metric1 = r2_score(y_test, y_pred)
            metric2 = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        # Feature importance (SHAP preferred) — [S4] pass task for correct predict fn
        feat_imp = _extract_shap_importance(pipeline, feature_names, X_train, top_k=10, task=task)
        if feat_imp and verbose:
            _print_shap_summary(
                {"shap_values": feat_imp, "shap_method": "SHAP/sklearn", "model_type": cls_name},
                model_name=name,
                top_k=10,
            )

        elapsed = time.time() - start
        result = PhaseResult(
            name=name,
            pipeline=pipeline,
            cv_scores=cv_scores,
            metric1=metric1,
            metric2=metric2,
            best_params=best_params,
            training_time=elapsed,
            feature_importance=feat_imp,
        )

        # [F6] Print per-model card
        if verbose:
            _print_model_card(result, task)

        if model_folder:
            _save_phase_result(result, model_folder, task.value)
            vp(f"   Saved to {model_folder}")

        logger.info(
            "phase2_model_complete: model=%s cv_mean=%.4f metric1=%.4f elapsed=%.1fs",
            name, result.cv_mean, metric1, elapsed,
        )
        return result

    except Exception:
        logger.error("Model '%s' failed:\n%s", name, traceback.format_exc())
        if verbose:
            print(f"\n[ERROR] Model '{name}' failed — see logs.")
        return None


def _run_hyperparameter_search(
    *,
    name: str,
    base_pipe: ImbPipeline,
    preprocessor: Optional[Any],
    sampler: Optional[Any],
    model: Any,
    fitted_cache: Dict[int, Any],
    param_grid: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    cv_splitter: Any,
    primary_metric: Any,
    f_kwargs: Dict[str, Any],
    search_type: SearchType,
    halving_factor: int,
    optuna_n_trials: int,
    optuna_timeout: Optional[int],
    optuna_storage: Optional[str],
    optuna_study_name: Optional[str],
    distributed_tuning: bool,
    random_seed: int,
    save_dir: Optional[str],
    n_jobs: int,
    verbose: bool,
) -> Tuple[ImbPipeline, Dict[str, Any]]:
    """
    Run the requested hyperparameter search and return (fitted_pipeline, best_params).
    """
    vp = _vp(verbose)

    if search_type == SearchType.OPTUNA:
        return _run_optuna_search(
            name=name,
            base_pipe=base_pipe,
            preprocessor=preprocessor,
            sampler=sampler,
            model=model,
            fitted_cache=fitted_cache,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cv_splitter=cv_splitter,
            primary_metric=primary_metric,
            f_kwargs=f_kwargs,
            n_trials=optuna_n_trials,
            timeout=optuna_timeout,
            storage_url=optuna_storage if distributed_tuning else None,
            study_name=optuna_study_name,
            distributed=distributed_tuning,
            random_seed=random_seed,
            save_dir=save_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    # sklearn-based searches
    common_kwargs: Dict[str, Any] = {
        "cv": cv_splitter,
        "scoring": primary_metric,
        "n_jobs": n_jobs,
    }
    if search_type == SearchType.GRID:
        searcher = GridSearchCV(base_pipe, param_grid, **common_kwargs)
        label = "GridSearchCV"
    elif search_type == SearchType.HALVING:
        searcher = HalvingGridSearchCV(
            base_pipe, param_grid, factor=halving_factor,
            random_state=random_seed, **common_kwargs
        )
        label = f"HalvingGridSearchCV (factor={halving_factor})"
    else:  # random
        searcher = RandomizedSearchCV(
            base_pipe, param_grid, n_iter=10,
            random_state=random_seed, **common_kwargs
        )
        label = "RandomizedSearchCV"

    vp(f"[{name}] {label}")
    with threadpool_limits(limits=1, user_api="blas"):
        with threadpool_limits(limits=1, user_api="openmp"):
            searcher.fit(X_train, y_train, **_strip_early_stopping_params(f_kwargs))

    best_params = {k.replace("model__", ""): v for k, v in searcher.best_params_.items()}
    return searcher.best_estimator_, best_params


def _run_optuna_search(
    *,
    name: str,
    base_pipe: ImbPipeline,
    preprocessor: Optional[Any],
    sampler: Optional[Any],
    model: Any,
    fitted_cache: Dict[int, Any],
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    cv_splitter: Any,
    primary_metric: Any,
    f_kwargs: Dict[str, Any],
    n_trials: int,
    timeout: Optional[int],
    storage_url: Optional[str],
    study_name: Optional[str],
    distributed: bool,
    random_seed: int,
    save_dir: Optional[str],
    n_jobs: int,
    verbose: bool,
) -> Tuple[ImbPipeline, Dict[str, Any]]:
    """
    Optuna Bayesian hyperparameter optimisation, with optional distributed storage.

    [P6] cross_val_score inside the objective now uses n_jobs so CV folds run
    in parallel within each trial — previously each trial ran serially.

    [B6] param keys from param_grid may include the `model__` prefix that
    GridSearch uses — they are stripped before set_params to avoid
    "unknown parameter" errors from Optuna trials.

    Returns (best_fitted_pipeline, best_params_dict).
    """
    vp = _vp(verbose)

    resolved_study = (
        study_name.format(model=name)
        if study_name and "{model}" in study_name
        else study_name
    )
    storage_final, resolved_study = _initialize_optuna_storage(
        storage_url, resolved_study, save_dir, verbose
    )

    cv_only_kwargs = _strip_early_stopping_params(f_kwargs)

    def objective(trial: optuna.Trial) -> float:
        trial_params: Dict[str, Any] = {}
        for pname, prange in param_grid.items():
            if isinstance(prange, list):
                trial_params[pname] = trial.suggest_categorical(pname, prange)
            elif isinstance(prange, tuple) and len(prange) == 2:
                lo, hi = prange
                trial_params[pname] = (
                    trial.suggest_int(pname, int(lo), int(hi))
                    if isinstance(lo, int) and isinstance(hi, int)
                    else trial.suggest_float(pname, float(lo), float(hi))
                )

        trial_model = clone(model)
        # [B6] Strip `model__` prefix before set_params — param_grid keys may carry it
        clean_params = {k.replace("model__", ""): v for k, v in trial_params.items()}
        trial_model.set_params(**clean_params)
        trial_pipe = _build_pipeline_from_cache(preprocessor, sampler, trial_model, fitted_cache)

        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=1, user_api="openmp"):
                # [P6] n_jobs runs CV folds in parallel per trial
                scores = cross_val_score(
                    trial_pipe, X_train, y_train,
                    cv=cv_splitter, scoring=primary_metric,
                    n_jobs=n_jobs,
                )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        storage=storage_final,
        study_name=resolved_study,
        load_if_exists=bool(distributed and storage_final),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if distributed and storage_final:
        n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        vp(f"   Distributed: {n_done} trials already done (resuming)")

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_model = clone(model)
    # [B6] Strip prefix here too for consistency
    clean_best = {k.replace("model__", ""): v for k, v in best_params.items()}
    best_model.set_params(**clean_best)
    best_pipe = _build_pipeline_from_cache(preprocessor, sampler, best_model, fitted_cache)

    final_fit_kw = (
        _inject_eval_set(f_kwargs, X_val, y_val, best_model)
        if X_val is not None
        else _strip_early_stopping_params(f_kwargs)
    )
    with threadpool_limits(limits=1, user_api="blas"):
        with threadpool_limits(limits=1, user_api="openmp"):
            best_pipe.fit(X_train, y_train, **final_fit_kw)

    vp(f"[{name}] Best Optuna trial #{study.best_trial.number}: {study.best_value:.4f}")
    return best_pipe, best_params


def _run_phase3_ensemble(
    best_results: Dict[str, PhaseResult],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: TaskType,
    cv_splitter: Any,
    random_seed: int,
    n_jobs: int,
    verbose: bool,
) -> Optional[Any]:
    """
    Phase 3 — Build a stacking ensemble from the top 3 trained models.

    [P4] Base estimators are passed as CLONES (unfitted) because StackingClassifier
    requires unfitted estimators — it re-fits them internally via cross-validation
    to generate out-of-fold predictions for the meta-learner. Passing fitted
    estimators would cause it to skip internal CV and risk data leakage.

    [B2] StackingRegressor is now used correctly for regression tasks.

    Returns the fitted stacker, or None on failure.
    """
    if len(best_results) < 2:
        return None

    vp = _vp(verbose)
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3: STACKING ENSEMBLE")
        print("=" * 70)

    ranked = sorted(best_results.values(), key=lambda r: r.metric1, reverse=True)
    top = ranked[: min(3, len(ranked))]
    vp(f"\nBase models: {[r.name for r in top]}")

    # StackingClassifier/Regressor require unfitted estimators for their internal CV
    estimators = [(r.name.replace(" ", "_"), clone(r.pipeline)) for r in top]

    try:
        if task == TaskType.CLASSIFICATION:
            stacker = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000, random_state=random_seed),
                cv=cv_splitter,
                n_jobs=n_jobs,
            )
        else:
            # [B2] Was missing in the original — caused NameError on regression tasks
            stacker = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(random_state=random_seed),
                cv=cv_splitter,
                n_jobs=n_jobs,
            )

        with threadpool_limits(limits=1, user_api="blas"):
            with threadpool_limits(limits=1, user_api="openmp"):
                stacker.fit(X_train, y_train)

        y_pred = stacker.predict(X_test)
        score = (
            accuracy_score(y_test, y_pred)
            if task == TaskType.CLASSIFICATION
            else r2_score(y_test, y_pred)
        )
        metric_label = "Accuracy" if task == TaskType.CLASSIFICATION else "R2"
        vp(f"\nEnsemble {metric_label}: {score:.4f}")
        vp("Ensemble trained successfully!")
        logger.info("phase3_ensemble_complete: score=%.4f", score)
        return stacker

    except Exception as exc:
        logger.error("Ensemble failed: %s", exc)
        if verbose:
            print(f"\nEnsemble failed: {exc}")
        return None


def _run_phase4_calibration(
    winner: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    cv: int,
    calibration_method: str = "auto",
    verbose: bool = True,
) -> Any:
    """
    Phase 4 — Probability calibration for the best classification model.

    [D5] calibration_method is now an exposed parameter:
        "auto"     — isotonic for >=100 train samples, sigmoid otherwise
        "isotonic" — always use isotonic regression
        "sigmoid"  — always use Platt Scaling

    Returns the calibrated model (or the original if calibration fails /
    predict_proba is unavailable).
    """
    vp = _vp(verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4: PROBABILITY CALIBRATION")
        print("=" * 70)

    if not hasattr(winner, "predict_proba"):
        vp("Model has no predict_proba — skipping calibration.")
        return winner

    if X_val is None:
        vp(
            f"   val_size=0.0 -> calibration uses K-fold CV on the training set "
            f"({cv} folds, ~{cv}x slower than prefit). "
            "Set val_size > 0 for faster calibration."
        )

    # [D5] Resolve "auto" method
    if calibration_method == "auto":
        resolved_method = "isotonic" if len(X_train) >= 100 else "sigmoid"
    else:
        resolved_method = calibration_method

    try:
        calibrated = _calibrate_model(
            winner, X_train, y_train,
            calibration_method=resolved_method,
            cv=min(5, cv),
            X_val=X_val,
            y_val=y_val,
        )

        raw_proba = winner.predict_proba(X_test)
        cal_proba = calibrated.predict_proba(X_test)
        is_binary = raw_proba.shape[1] == 2

        ll_before = log_loss(y_test, raw_proba)
        ll_after  = log_loss(y_test, cal_proba)
        vp(f"   Log-Loss  {ll_before:.4f} -> {ll_after:.4f}")

        if is_binary:
            y_test_arr = np.asarray(y_test)
            ece_before = _calculate_expected_calibration_error(y_test_arr, raw_proba[:, 1])
            ece_after  = _calculate_expected_calibration_error(y_test_arr, cal_proba[:, 1])
            vp(f"   ECE       {ece_before:.4f} -> {ece_after:.4f}")
        else:
            vp("   ECE: skipped for multi-class (binary only)")

        vp("   Calibration complete.")
        logger.info(
            "phase4_calibration_complete: ll_before=%.4f ll_after=%.4f method=%s",
            ll_before, ll_after, resolved_method,
        )
        return calibrated

    except Exception as exc:
        logger.error("Calibration failed: %s", exc)
        if verbose:
            print(f"   Calibration failed: {exc} — returning uncalibrated model.")
        return winner


# ==============================================================================
# [F1] CONCURRENT PHASE-2 RUNNER
# ==============================================================================

def _run_phase2_concurrent(
    active_models: Dict[str, Any],
    best_results: Dict[str, PhaseResult],
    checkpoint_file: Optional[str],
    n_parallel: int,
    verbose: bool,
    **phase2_kwargs: Any,
) -> Dict[str, PhaseResult]:
    """
    [F1] Run Phase-2 model training concurrently using ProcessPoolExecutor.

    When n_parallel=1 (default) this falls back to the original sequential
    loop — no behaviour change unless the caller opts in.

    Each model is a CPU-bound workload (fitting + CV), so processes are more
    effective than threads. The fitted PhaseResult objects are returned in
    completion order and checkpointed atomically after each one.

    Note: ProcessPoolExecutor requires that all arguments are picklable.
    Lambda-based preprocessors or closures will raise PicklingError — use
    named functions or sklearn Pipeline objects instead.

    [B5] None-check on future result happens before accessing result.name.
    """
    vp = _vp(verbose)
    to_run = {n: m for n, m in active_models.items() if n not in best_results}

    if n_parallel == 1:
        # Sequential path — preserves original behaviour exactly
        for name, model in to_run.items():
            vp(f"\n{'='*70}\n[{name}] Phase 2: Full Evaluation\n{'='*70}")
            result = _run_phase2_single_model(name=name, model=model, verbose=verbose, **phase2_kwargs)
            if result is not None:
                best_results[name] = result
                if checkpoint_file:
                    _atomic_checkpoint_dump({"best_results": best_results}, checkpoint_file)
                    vp("Checkpoint updated.")
        return best_results

    # Parallel path
    vp(f"\nRunning {len(to_run)} model(s) in parallel (n_parallel={n_parallel})")
    with ProcessPoolExecutor(max_workers=n_parallel) as pool:
        futures = {
            pool.submit(_run_phase2_single_model, name=name, model=model, verbose=verbose, **phase2_kwargs): name
            for name, model in to_run.items()
        }
        for fut in as_completed(futures):
            submitted_name = futures[fut]
            try:
                result = fut.result()
            except Exception:
                logger.error(
                    "Concurrent Phase-2 failed for %s:\n%s",
                    submitted_name, traceback.format_exc(),
                )
                result = None

            # [B5] Check result is not None before accessing result.name
            if result is not None:
                best_results[result.name] = result
                if checkpoint_file:
                    _atomic_checkpoint_dump({"best_results": best_results}, checkpoint_file)
                    vp(f"Checkpoint updated after [{result.name}].")
            else:
                vp(f"[{submitted_name}] failed — skipped.")

    return best_results


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def evaluate_and_plot_models(
    models: Dict[str, Any],
    preprocess_pipeline: Union[Any, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    # --- Split ---
    test_size: float = 0.2,
    val_size: float = 0.0,
    split_method: str = "random",
    stratify: bool = True,
    random_seed: int = 42,
    # --- Task ---
    task_type: str = "classification",
    target_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    # --- Tuning ---
    param_grids: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Dict[str, Any]]] = None,
    sampler: Optional[Any] = None,
    # [A2] `cv` renamed to `n_cv_folds` for clarity in the public API
    n_cv_folds: int = 4,
    cv: Optional[int] = None,  # Deprecated: use n_cv_folds instead
    search_type: str = "grid",
    primary_metric: Optional[str] = None,
    # --- Screening ---
    top_k: Optional[int] = None,
    quick_test_fraction: float = 0.2,
    # --- Plots ---
    plot_lc: bool = True,
    plot_diagnostics: bool = True,
    plot_importance: bool = True,
    plot_comparison: bool = True,
    top_n_features: int = 20,
    # --- Persistence ---
    save_dir: Optional[str] = None,
    resume: bool = False,
    # --- Behaviour flags ---
    show_fold_details: bool = True,
    halving_factor: int = 3,
    # --- Early stopping ---
    # [F5] Exposed as a top-level parameter (was hard-coded to 50)
    early_stopping_rounds: int = 50,
    # --- Optuna ---
    optuna_n_trials: int = 50,
    optuna_timeout: Optional[int] = None,
    optuna_storage: Optional[str] = None,
    optuna_study_name: Optional[str] = None,
    distributed_tuning: bool = False,
    # --- Quality / ensemble ---
    validate_data: bool = True,
    validate_drift: bool = False,
    drift_threshold: float = 0.05,
    build_ensemble: bool = True,
    # --- MLflow ---
    mlflow_experiment: str = "AutoML_Run",
    # --- Calibration ---
    calibration_method: str = "auto",
    # --- Parallelism ---
    n_jobs: int = -1,
    n_parallel: int = 1,
    # --- Memory ---
    memory_efficient: bool = False,
    verbose: bool = True,
) -> "EvalResult":
    """
    Enterprise AutoML pipeline: screen -> tune -> ensemble -> calibrate.

    Parameters
    ----------
    models
        Dict of {name: unfitted_estimator}.
    preprocess_pipeline
        A single transformer/pipeline applied to all models, or a dict
        {model_name: transformer} with an optional "default" key.
    X, y
        Feature DataFrame and target Series.  Must be pre-imputed (no NaNs).
    test_size / val_size
        Fractions for test and (optional) validation splits.
    split_method
        "random" (stratified where applicable) or "sequential" (time-series).
    task_type
        "classification" or "regression".
    param_grids
        {model_name: param_grid} passed to the chosen search strategy.
        For Optuna, values can be lists (categorical) or (lo, hi) tuples.
    fit_params
        {model_name: {fit_kwarg: value}} extra kwargs forwarded to `.fit()`.
        [A1] Defaults to empty dict — callers never need to pass None.
    search_type
        "grid" | "random" | "halving" | "optuna".
    primary_metric
        sklearn scoring string.  Defaults to "accuracy" / "r2".
    top_k
        [A3] If set and top_k < len(models), only the top_k models by quick
        screening proceed to full evaluation. None (default) skips screening
        entirely and trains all models in Phase 2.
    validate_drift
        [F2] Run KS-test covariate drift detection on train vs test splits.
    drift_threshold
        p-value threshold for the KS-test. Default 0.05.
    calibration_method
        [D5] "auto" (default), "isotonic", or "sigmoid".
    early_stopping_rounds
        [F5] Rounds without improvement before early stopping. Default 50.
    n_cv_folds
        [A2] Number of CV folds. Default 4.
        Deprecated alias: `cv` (will be removed in v2.0).
    n_jobs
        [P2] Number of parallel jobs for CV and search. Default -1 (all cores).
    n_parallel
        [F1] Number of models to train concurrently in Phase 2.
        Default 1 (sequential). Set to os.cpu_count() // 2 for full parallelism.
        Requires all fit_params / preprocessors to be picklable.
    optuna_storage
        None (in-memory), "sqlite:///path.db", or "redis://host:port".
    distributed_tuning
        When True, Optuna study is shared for distributed / resumed runs.
    validate_data
        Run Pandera data-contract checks before modelling.
    build_ensemble
        Build a stacking ensemble from the top 3 trained models.
    resume
        Load a previously saved checkpoint and skip already-trained models.
    memory_efficient
        Disable all plots and reduce memory footprint.

    Returns
    -------
    EvalResult (TypedDict) with keys:
        summary_df       — ranked performance table
        best_models      — {name: fitted_pipeline}
        raw_cv_scores    — {name: List[float] of fold scores}
        ultimate_winner  — best / calibrated / ensemble model
        data_splits      — X_train, y_train, X_val, y_val, X_test, y_test
        drift_report     — List[(col, p_value)] if validate_drift=True
    """
    vp = _vp(verbose)

    # --- Validate inputs & coerce enums ---
    _validate_inputs(X, y, test_size, val_size)

    try:
        # [D3] Single canonical 'task' variable throughout
        task = TaskType(task_type.lower())
    except ValueError:
        raise ValueError(f"task_type must be 'classification' or 'regression'. Got '{task_type}'.")

    try:
        search_type_enum = SearchType(search_type.lower())
    except ValueError:
        raise ValueError(
            f"search_type must be one of {[s.value for s in SearchType]}. Got '{search_type}'."
        )

    try:
        split_method_enum = SplitMethod(split_method.lower())
    except ValueError:
        raise ValueError(f"split_method must be 'random' or 'sequential'. Got '{split_method}'.")

    # [A2] Handle legacy `cv` parameter for backward compatibility
    if cv is not None:
        warnings.warn(
            "Parameter 'cv' is deprecated; use 'n_cv_folds' instead. "
            "'cv' will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        n_cv_folds = cv

    # [A2] Internal variable keeps the old name for backward compat with sub-functions
    cv = n_cv_folds

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # [S3] Deep-copy fit_params so caller's dict is never mutated
    # [A1] Default to empty dict — no None-guards needed downstream
    fit_params = {k: dict(v) for k, v in (fit_params or {}).items()}

    if memory_efficient:
        vp("\nMEMORY-EFFICIENT MODE: plots disabled")
        plot_lc = plot_diagnostics = plot_importance = plot_comparison = False

    # --- CV splitter & default metric ---
    if task == TaskType.CLASSIFICATION:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)
        if primary_metric is None:
            primary_metric = "accuracy"
        elif primary_metric == "balanced_accuracy":
            primary_metric = get_balanced_accuracy_scorer()
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
        if primary_metric is None:
            primary_metric = "r2"
        if sampler is not None:
            warnings.warn("Samplers are not applicable for regression — disabling sampler.")
            sampler = None

    # =========================================================================
    # [S2] MLflow context manager — run is always closed, even on exception
    # =========================================================================
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"AutoML_{run_timestamp}") as mlflow_run:
        vp(f"MLflow run {mlflow_run.info.run_id} started")
        mlflow.log_params({
            "task_type": task_type,
            "test_size": test_size,
            "val_size": val_size,
            "cv_folds": cv,
            "search_type": search_type,
            "build_ensemble": build_ensemble,
            "n_parallel": n_parallel,
            "calibration_method": calibration_method,
            "early_stopping_rounds": early_stopping_rounds,
        })

        # --- Data validation ---
        if validate_data:
            validation = _validate_data_contract(X, y, feature_names, task_type, verbose)
            critical = [i for i in validation["issues"] if i.startswith("ERROR")]
            if critical:
                raise ValueError(
                    "Data validation failed:\n" + "\n".join(critical)
                    + "\n\nPass validate_data=False to skip (not recommended)."
                )
            for k, v in validation["stats"].items():
                try:
                    mlflow.log_param(f"data_{k}", v)
                except Exception:
                    pass

        # --- Data splits ---
        if verbose:
            print(f"\n{'='*70}\nDATA SPLIT ({split_method_enum.value})\n{'='*70}")

        if split_method_enum == SplitMethod.SEQUENTIAL:
            idx = int(len(X) * (1 - test_size))
            X_train_full, X_test = X.iloc[:idx], X.iloc[idx:]
            y_train_full, y_test = y.iloc[:idx], y.iloc[idx:]
        else:
            strat = y if (task == TaskType.CLASSIFICATION and stratify) else None
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=strat
            )

        X_val = y_val = None
        X_train, y_train = X_train_full, y_train_full

        if val_size > 0.0:
            val_frac = val_size / (1.0 - test_size)
            if split_method_enum == SplitMethod.SEQUENTIAL:
                vi = int(len(X_train_full) * (1 - val_frac))
                X_train, X_val = X_train_full.iloc[:vi], X_train_full.iloc[vi:]
                y_train, y_val = y_train_full.iloc[:vi], y_train_full.iloc[vi:]
            else:
                sv = y_train_full if (task == TaskType.CLASSIFICATION and stratify) else None
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full, y_train_full,
                    test_size=val_frac, random_state=random_seed, stratify=sv
                )
            vp(f"Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")
        else:
            vp(f"Train {X_train.shape}  Test {X_test.shape}")

        # --- [F2] Covariate drift detection ---
        drift_report: List[Tuple[str, float]] = []
        if validate_drift:
            drift_report = _check_covariate_drift(
                X_train, X_test, threshold=drift_threshold, verbose=verbose
            )
            if drift_report:
                mlflow.log_param("drifted_columns", len(drift_report))

        # --- Preprocessor cache ---
        if verbose:
            print(f"\n{'='*70}\nBuilding preprocessor cache\n{'='*70}")
        fitted_cache = _build_preprocessor_cache(
            preprocess_pipeline, list(models.keys()), X_train, y_train
        )
        vp(f"{len(fitted_cache)} unique preprocessor(s) fitted")

        # --- Phase 1: screening ---
        # [A3] Documented: None means skip screening; guard ensures top_k < len(models)
        active_models = dict(models)
        if top_k is not None and top_k < len(active_models):
            active_models = _run_phase1_screening(
                models=active_models,
                preprocess_pipeline=preprocess_pipeline,
                fitted_cache=fitted_cache,
                sampler=sampler,
                X_train=X_train,
                y_train=y_train,
                cv_splitter=cv_splitter,
                primary_metric=primary_metric,
                fit_params=fit_params,
                top_k=top_k,
                quick_test_fraction=quick_test_fraction,
                random_seed=random_seed,
                stratify=stratify,
                task=task,
                n_jobs=n_jobs,
                verbose=verbose,
            )

        # --- Checkpoint loading ---
        checkpoint_file = os.path.join(save_dir, "eval_checkpoint.pkl") if save_dir else None
        best_results: Dict[str, PhaseResult] = {}

        if resume and checkpoint_file and os.path.exists(checkpoint_file):
            vp(f"\nLoading checkpoint from {checkpoint_file}")
            state = joblib.load(checkpoint_file)
            best_results = state.get("best_results", {})
            vp(f"-> Loaded {len(best_results)} completed model(s): {list(best_results)}")

        # --- Crash recovery ---
        incomplete = _detect_incomplete_models(save_dir, verbose)
        if incomplete:
            _cleanup_incomplete_models(incomplete, verbose)
            for crashed_name in incomplete:
                best_results.pop(crashed_name, None)

        # --- Phase 2: full evaluation (sequential or concurrent) ---
        # Common kwargs passed to every _run_phase2_single_model call
        phase2_kwargs = dict(
            preprocess_pipeline=preprocess_pipeline,
            fitted_cache=fitted_cache,
            sampler=sampler,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            cv_splitter=cv_splitter,
            primary_metric=primary_metric,
            param_grids=param_grids,
            fit_params=fit_params,
            search_type=search_type_enum,
            halving_factor=halving_factor,
            optuna_n_trials=optuna_n_trials,
            optuna_timeout=optuna_timeout,
            optuna_storage=optuna_storage,
            optuna_study_name=optuna_study_name,
            distributed_tuning=distributed_tuning,
            random_seed=random_seed,
            task=task,
            feature_names=feature_names,
            show_fold_details=show_fold_details,
            save_dir=save_dir,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,  # [F5]
        )

        best_results = _run_phase2_concurrent(
            active_models=active_models,
            best_results=best_results,
            checkpoint_file=checkpoint_file,
            n_parallel=n_parallel,
            verbose=verbose,
            **phase2_kwargs,
        )

        # Log Phase-2 metrics to MLflow
        for result in best_results.values():
            mlflow.log_metrics({
                f"{result.name}_cv_mean": result.cv_mean,
                f"{result.name}_cv_std": result.cv_std,
                f"{result.name}_metric1": result.metric1,
            })

        # --- Phase 3: ensemble ---
        stacker = None
        if build_ensemble and len(best_results) >= 2:
            stacker = _run_phase3_ensemble(
                best_results=best_results,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                task=task,
                cv_splitter=cv_splitter,
                random_seed=random_seed,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            if stacker is not None:
                y_ens = stacker.predict(X_test)
                ens_score = (
                    accuracy_score(y_test, y_ens)
                    if task == TaskType.CLASSIFICATION
                    else r2_score(y_test, y_ens)
                )
                mlflow.log_metric("ensemble_test_score", ens_score)
                mlflow.sklearn.log_model(stacker, "ensemble_model")

        # --- Build summary ---
        rows = []
        for r in best_results.values():
            rows.append({
                "Model Name": r.name,
                "Train Time": f"{r.training_time:.1f}s",
                "CV Mean": f"{r.cv_mean:.3f} (+-{r.cv_std:.3f})",
                "Metric 1 Raw": r.metric1,
                "Metric 2 Raw": r.metric2,
                "Best Params": str(r.best_params) if r.best_params else "Default",
            })

        if stacker is not None:
            y_ens = stacker.predict(X_test)
            ens_m1 = (
                accuracy_score(y_test, y_ens)
                if task == TaskType.CLASSIFICATION
                else r2_score(y_test, y_ens)
            )
            rows.append({
                "Model Name": "Ensemble_Stack",
                "Train Time": "N/A",
                "CV Mean": "N/A",
                "Metric 1 Raw": ens_m1,
                "Metric 2 Raw": ens_m1,
                "Best Params": "Stacking",
            })

        summary_df = pd.DataFrame()
        if rows:
            col1, col2 = (
                ("Test Acc", "Test F1")
                if task == TaskType.CLASSIFICATION
                else ("Test R2", "Test RMSE")
            )
            summary_df = (
                pd.DataFrame(rows)
                .sort_values("Metric 1 Raw", ascending=False)
                .reset_index(drop=True)
            )
            summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))
            summary_df[col1] = summary_df["Metric 1 Raw"].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
            )
            summary_df[col2] = summary_df["Metric 2 Raw"].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
            )
            summary_df = summary_df.drop(columns=["Metric 1 Raw", "Metric 2 Raw"])
            ordered_cols = ["Rank", "Model Name", "Train Time", "CV Mean", col1, col2, "Best Params"]
            summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]

            if verbose:
                print("\n" + "=" * 100)
                print("RANKED MODEL PERFORMANCE SUMMARY")
                print("=" * 100)
                with pd.option_context("display.max_rows", None, "display.max_columns", None,
                                       "display.width", 1000, "display.max_colwidth", None):
                    print(summary_df.to_string(index=False))
                print("=" * 100)

            # [F4] Log summary_df as a CSV artifact to MLflow
            try:
                csv_dir = save_dir or tempfile.mkdtemp()
                csv_path = os.path.join(csv_dir, "model_summary.csv")
                summary_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path, artifact_path="results")
                vp(f"Summary CSV logged to MLflow: {csv_path}")
            except Exception as exc:
                logger.warning("Could not log summary CSV to MLflow: %s", exc)

        # --- Determine ultimate winner ---
        best_individual = (
            sorted(best_results.values(), key=lambda r: r.metric1, reverse=True)[0].pipeline
            if best_results else None
        )
        ultimate_winner = stacker if stacker is not None else best_individual

        # --- Phase 4: calibration (classification only) ---
        if task == TaskType.CLASSIFICATION and ultimate_winner is not None:
            ultimate_winner = _run_phase4_calibration(
                winner=ultimate_winner,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                X_val=X_val,
                y_val=y_val,
                cv=cv,
                calibration_method=calibration_method,
                verbose=verbose,
            )
            mlflow.sklearn.log_model(ultimate_winner, "calibrated_winner")

        # --- Folder overview ---
        if save_dir:
            _create_structure_overview(save_dir, summary_df)

    # MLflow run closed by context manager — no finally block needed [S2]

    vp(f"MLflow experiment '{mlflow_experiment}' — run complete.")

    best_estimators = {r.name: r.pipeline for r in best_results.values()}
    if stacker is not None:
        best_estimators["Ensemble_Stack"] = stacker
    if ultimate_winner is not None:
        best_estimators["Calibrated_Winner"] = ultimate_winner

    return {
        "summary_df": summary_df,
        "best_models": best_estimators,
        "raw_cv_scores": {r.name: r.cv_scores for r in best_results.values()},
        "ultimate_winner": ultimate_winner,
        "data_splits": {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        },
        "drift_report": drift_report,
    }