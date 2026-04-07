import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Imblearn Pipeline alias to protect against Scikit-Learn overrides
from imblearn.pipeline import Pipeline as ImbPipeline

# Universal Metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Model Selection & Splitters
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _validate_inputs(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float) -> None:
    """
    Validate input data and parameters.

    Args:
        X: Feature dataframe
        y: Target series
        test_size: Test set fraction
        val_size: Validation set fraction

    Raises:
        ValueError: If inputs are invalid
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

    Args:
        preprocess_pipeline: Single pipeline or dict of pipelines
        model_name: Name of the model

    Returns:
        The preprocessor for this model, or None
    """
    if isinstance(preprocess_pipeline, dict):
        return preprocess_pipeline.get(model_name, preprocess_pipeline.get("default"))
    return preprocess_pipeline


def _build_pipeline(preprocessor: Optional[Any], sampler: Optional[Any], model: Any) -> ImbPipeline:
    """
    Build a pipeline with preprocessor, sampler, and model.

    Args:
        preprocessor: Optional preprocessing pipeline
        sampler: Optional sampling strategy (e.g., SMOTE)
        model: The ML model

    Returns:
        ImbPipeline with all components
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


def _extract_feature_importance(
    pipeline: Any, feature_names: Optional[List[str]] = None, top_k: int = 10
) -> Dict[str, float]:
    """
    Extract feature importance/coefficients from a trained pipeline.
    Returns a dict with feature names and their importance percentages.

    Args:
        pipeline: Trained ImbPipeline
        feature_names: Original feature names
        top_k: Number of top features to return

    Returns:
        Dict with feature names as keys and percentage importance as values
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
    # --- NEW: Master Data Inputs ---
    X: pd.DataFrame,
    y: pd.Series,
    # --- NEW: Splitting Parameters ---
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
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Enterprise machine learning evaluation pipeline.
    Handles internal data splitting, automated tuning, imbalanced sampling,
    and cross-validation for both classification and regression tasks.
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

    # Note: Cross-validation uses X_train. The internal X_val is returned in the final dict
    # for advanced users who need a clean holdout set for external post-processing.

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
            base_pipeline = _build_pipeline(current_preprocessor, sampler, model)
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
    # PHASE 2: FULL EVALUATION AND TUNING
    # ==============================================================================
    results_score = []
    test_scores_list = []
    successful_names = []
    summary_data = []
    best_estimators = {}

    for name, model in models.items():
        start_time = time.time()

        vprint("\n" + "=" * 70)
        vprint(f"[{name}] Starting Full Evaluation...")
        vprint("=" * 70)

        try:
            current_preprocessor = _get_preprocessor(preprocess_pipeline, name)
            base_pipeline = _build_pipeline(current_preprocessor, sampler, model)

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
                search_cls = GridSearchCV if search_type.lower() == "grid" else RandomizedSearchCV
                kwargs = {"cv": cv_splitter, "scoring": primary_metric, "n_jobs": -1}
                if search_type.lower() != "grid":
                    kwargs.update({"n_iter": 10, "random_state": random_seed})

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
                metric1 = accuracy_score(y_test, y_val_pred)
                metric2 = f1_score(y_test, y_val_pred, average="macro", zero_division=0)
                vprint(f"\nClassification Report for {name}:")
                vprint(
                    classification_report(
                        y_test, y_val_pred, target_names=target_names, zero_division=0
                    )
                )
            else:
                metric1 = r2_score(y_test, y_val_pred)
                metric2 = np.sqrt(mean_squared_error(y_test, y_val_pred))
                mae = mean_absolute_error(y_test, y_val_pred)
                vprint(f"\nRegression Report for {name}:")
                vprint(f"R² Score: {metric1:.4f} | RMSE: {metric2:.4f} | MAE: {mae:.4f}")

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
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title(f"Predicted vs. Actual: {name}")
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.6)

                plt.tight_layout()
                if verbose:
                    plt.show()
                plt.close(fig)

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

            if plot_importance:
                fitted_model = pipeline.named_steps["model"]
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

                if importances_array is not None:
                    current_names = list(feature_names) if feature_names else []

                    # Try to extract feature names from pipeline steps
                    # Handle both 'preprocessor' key and direct pipeline steps
                    preprocessor = pipeline.named_steps.get("preprocessor")
                    if preprocessor is None:
                        # If no preprocessor key, search for any step with get_feature_names_out
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

                    actual_n = len(importances_array)
                    if len(current_names) < actual_n:
                        missing = actual_n - len(current_names)
                        # Try to identify engineered features from step names
                        for i in range(missing):
                            current_names.append(f"Feature_{i+1}")
                    elif len(current_names) > actual_n:
                        current_names = current_names[:actual_n]

                    n_feats = min(len(importances_array), top_n_features)
                    top_indices = np.argsort(importances_array)[-n_feats:]
                    top_vals = importances_array[top_indices]
                    top_names = np.array(current_names)[top_indices]

                    fig = plt.figure(figsize=(8, 5))
                    color = "steelblue" if hasattr(fitted_model, "coef_") else "teal"
                    title_type = "Coefficients" if hasattr(fitted_model, "coef_") else "Importances"

                    plt.barh(top_names, top_vals, color=color)
                    plt.title(f"Top {n_feats} Feature {title_type}: {name}")
                    plt.xlabel("Importance / Impact")
                    plt.axvline(0, color="black", lw=1)
                    plt.tight_layout()
                    if verbose:
                        plt.show()
                    plt.close(fig)

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
        # --- NEW: Return the raw splits for advanced users ---
        "data_splits": {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        },
    }
