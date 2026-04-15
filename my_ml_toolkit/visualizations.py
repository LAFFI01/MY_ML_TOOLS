"""
Industry-Grade Model Visualization & Comparison Module

Comprehensive plotting library for ML model evaluation:
- Individual model performance dashboards
- Multi-model comparison plots
- Learning curves & convergence analysis
- Feature importance (SHAP + sklearn)
- Classification diagnostics (ROC, PR, confusion matrix, calibration)
- Regression diagnostics (residuals, prediction error, quantile plots)
- Model ranking & performance heatmaps
"""

import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
)

# Professional color palette
COLORS_PRIMARY = sns.color_palette("husl", 10)
COLORS_DIVERGING = sns.color_palette("RdYlGn", 256)
COLORS_CATEGORICAL = sns.color_palette("Set2", 8)
STYLE_DARK = "dark_background"
STYLE_LIGHT = "seaborn-v0_8-darkgrid"

# Set professional matplotlib defaults
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "grid.alpha": 0.3,
})


# ==============================================================================
# INDIVIDUAL MODEL PERFORMANCE DASHBOARDS
# ==============================================================================

def plot_model_performance_dashboard(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task_type: str = "classification",
    metrics_dict: Optional[Dict[str, float]] = None,
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Create comprehensive single-model performance dashboard.
    
    **Classification Dashboard includes:**
    - Confusion matrix (heatmap)
    - ROC-AUC curve
    - Precision-Recall curve
    - Calibration curve (if y_proba available)
    - Performance metrics table
    - Per-class performance breakdown
    
    **Regression Dashboard includes:**
    - Actual vs Predicted scatter plot
    - Residual plot
    - Residual distribution histogram
    - Quantile-Quantile plot
    - Metrics summary table
    
    Args:
        model_name: Name of the model
        y_true: True labels/values
        y_pred: Predicted labels/values
        y_proba: Predicted probabilities (classification only)
        task_type: 'classification' or 'regression'
        metrics_dict: Dictionary with pre-calculated metrics {metric_name: score}
        figsize: Figure size
        save_path: Path to save figure (e.g., 'plots/model_dashboard.png')
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"🎯 {model_name} - Performance Dashboard", fontsize=16, fontweight="bold", y=0.995)
    
    if task_type == "classification":
        _plot_classification_dashboard(
            axes, y_true, y_pred, y_proba, model_name, metrics_dict
        )
    else:
        _plot_regression_dashboard(
            axes, y_true, y_pred, model_name, metrics_dict
        )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Dashboard saved: {save_path}")
    plt.show()


def _plot_classification_dashboard(axes, y_true, y_pred, y_proba, model_name, metrics_dict):
    """Plot classification performance dashboard (6 subplots)."""
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0], 
                cbar_kws={"label": "Count"}, annot_kws={"fontsize": 12})
    axes[0, 0].set_title("Confusion Matrix", fontweight="bold")
    axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xlabel("Predicted Label")
    
    # 2. ROC-AUC Curve (if binary classification with probabilities)
    if y_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, linewidth=2.5, label=f"ROC-AUC = {roc_auc:.3f}", color="#FF6B6B")
        axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier", alpha=0.7)
        axes[0, 1].fill_between(fpr, tpr, alpha=0.2, color="#FF6B6B")
        axes[0, 1].set_title("ROC-AUC Curve", fontweight="bold")
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, "No probability data\nfor ROC curve", 
                       ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("ROC-AUC Curve", fontweight="bold")
    
    # 3. Precision-Recall Curve
    if y_proba is not None and len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(
            y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        )
        pr_auc = auc(recall, precision)
        axes[0, 2].plot(recall, precision, linewidth=2.5, label=f"PR-AUC = {pr_auc:.3f}", color="#4ECDC4")
        axes[0, 2].fill_between(recall, precision, alpha=0.2, color="#4ECDC4")
        axes[0, 2].set_title("Precision-Recall Curve", fontweight="bold")
        axes[0, 2].set_xlabel("Recall")
        axes[0, 2].set_ylabel("Precision")
        axes[0, 2].legend(loc="best")
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, "No probability data\nfor PR curve", 
                       ha="center", va="center", transform=axes[0, 2].transAxes)
        axes[0, 2].set_title("Precision-Recall Curve", fontweight="bold")
    
    # 4. Calibration Curve
    if y_proba is not None and len(np.unique(y_true)) == 2:
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba, n_bins=10
        )
        axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Perfect Calibration")
        axes[1, 0].plot(prob_pred, prob_true, "o-", linewidth=2.5, markersize=8, 
                       label="Model", color="#95E1D3")
        axes[1, 0].fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color="#95E1D3")
        axes[1, 0].set_title("Calibration Curve", fontweight="bold")
        axes[1, 0].set_xlabel("Mean Predicted Probability")
        axes[1, 0].set_ylabel("Fraction of Positives")
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend(loc="upper left")
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "No probability data\nfor calibration", 
                       ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Calibration Curve", fontweight="bold")
    
    # 5. Class Distribution Comparison
    unique_classes = np.unique(y_true)
    x = np.arange(len(unique_classes))
    width = 0.35
    
    true_counts = [np.sum(y_true == c) for c in unique_classes]
    pred_counts = [np.sum(y_pred == c) for c in unique_classes]
    
    axes[1, 1].bar(x - width/2, true_counts, width, label="True", color="#3498db", alpha=0.8)
    axes[1, 1].bar(x + width/2, pred_counts, width, label="Predicted", color="#e74c3c", alpha=0.8)
    axes[1, 1].set_title("Class Distribution", fontweight="bold")
    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(unique_classes)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    # 6. Metrics Summary Table
    axes[1, 2].axis("tight")
    axes[1, 2].axis("off")
    
    metrics_text = "📊 Performance Metrics\n" + "=" * 35 + "\n"
    if metrics_dict:
        for metric_name, value in metrics_dict.items():
            if isinstance(value, float):
                metrics_text += f"{metric_name}: {value:.4f}\n"
            else:
                metrics_text += f"{metric_name}: {value}\n"
    
    axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                   fontfamily="monospace", fontsize=10, verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))


def _plot_regression_dashboard(axes, y_true, y_pred, model_name, metrics_dict):
    """Plot regression performance dashboard (6 subplots)."""
    
    residuals = y_true - y_pred
    
    # 1. Actual vs Predicted Scatter
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=30, color="#3498db", edgecolors="navy")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    axes[0, 0].set_title("Actual vs Predicted", fontweight="bold")
    axes[0, 0].set_xlabel("Actual Values")
    axes[0, 0].set_ylabel("Predicted Values")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual Plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=30, color="#e74c3c", edgecolors="darkred")
    axes[0, 1].axhline(y=0, color="k", linestyle="--", linewidth=2)
    axes[0, 1].set_title("Residual Plot", fontweight="bold")
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residual Distribution
    axes[0, 2].hist(residuals, bins=30, color="#2ecc71", alpha=0.7, edgecolor="black")
    axes[0, 2].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[0, 2].set_title("Residual Distribution", fontweight="bold")
    axes[0, 2].set_xlabel("Residuals")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].grid(True, alpha=0.3, axis="y")
    
    # 4. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normality Check)", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Prediction Error Distribution
    errors = np.abs(residuals)
    axes[1, 1].hist(errors, bins=30, color="#9b59b6", alpha=0.7, edgecolor="black")
    axes[1, 1].axvline(x=np.mean(errors), color="r", linestyle="--", linewidth=2, 
                       label=f"Mean: {np.mean(errors):.3f}")
    axes[1, 1].set_title("Absolute Error Distribution", fontweight="bold")
    axes[1, 1].set_xlabel("Absolute Error")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    # 6. Metrics Summary Table
    axes[1, 2].axis("tight")
    axes[1, 2].axis("off")
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics_text = "📊 Performance Metrics\n" + "=" * 35 + f"\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\n"
    if metrics_dict:
        for metric_name, value in metrics_dict.items():
            if isinstance(value, float):
                metrics_text += f"{metric_name}: {value:.4f}\n"
    
    axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                   fontfamily="monospace", fontsize=10, verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))


# ==============================================================================
# MULTI-MODEL COMPARISON PLOTS
# ==============================================================================

def plot_model_comparison(
    results_df: pd.DataFrame,
    metric_column: str = "Test Accuracy",
    title: str = "Model Performance Comparison",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Compare multiple models with professional bar/violin plots.
    
    Args:
        results_df: DataFrame with model results (columns: model name, metrics)
        metric_column: Name of metric column to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"🏆 {title}", fontsize=14, fontweight="bold")
    
    # Sort by metric
    sorted_df = results_df.sort_values(metric_column, ascending=False)
    models = sorted_df.iloc[:, 0].astype(str).values  # First column as model names
    metrics = sorted_df[metric_column].values
    
    colors = COLORS_PRIMARY[:len(models)]
    
    # 1. Horizontal Bar Chart
    y_pos = np.arange(len(models))
    axes[0].barh(y_pos, metrics, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models)
    axes[0].set_xlabel(metric_column, fontweight="bold")
    axes[0].set_title("Ranked Model Performance", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="x")
    
    # Add value labels
    for i, v in enumerate(metrics):
        axes[0].text(v - 0.02, i, f"{v:.3f}", va="center", ha="right", fontweight="bold", color="white")
    
    # 2. Ranking Summary
    axes[1].axis("off")
    rank_text = "🥇 Model Rankings\n" + "=" * 40 + "\n\n"
    medals = ["🥇", "🥈", "🥉"] + ["#️⃣"] * (len(models) - 3)
    
    for rank, (medal, model, metric) in enumerate(zip(medals, models, metrics), 1):
        rank_text += f"{medal} {rank}. {model}\n    Score: {metric:.4f}\n"
    
    axes[1].text(0.05, 0.95, rank_text, transform=axes[1].transAxes,
                fontfamily="monospace", fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="black", linewidth=2))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Comparison plot saved: {save_path}")
    plt.show()


def plot_model_comparison_heatmap(
    results_df: pd.DataFrame,
    metric_columns: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Comprehensive heatmap of all models vs all metrics.
    
    Args:
        results_df: DataFrame with models (rows) and metrics (columns)
        metric_columns: Specific metric columns to include (if None, use all numeric)
        figsize: Figure size
        save_path: Path to save figure
    """
    # Select only numeric columns if not specified
    if metric_columns is None:
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
        metric_columns = numeric_cols
    
    # Create heatmap data
    heatmap_data = results_df.set_index(results_df.columns[0])[metric_columns].copy()
    heatmap_data = heatmap_data.astype(float)
    
    # Normalize each column to 0-1 for better visualization
    normalized_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(normalized_data, annot=heatmap_data, fmt=".3f", cmap="RdYlGn",
                center=0.5, cbar_kws={"label": "Normalized Score"}, ax=ax,
                linewidths=0.5, linecolor="gray")
    
    ax.set_title("🎯 Model Performance Heatmap (Normalized)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Metrics", fontweight="bold")
    ax.set_ylabel("Models", fontweight="bold")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Heatmap saved: {save_path}")
    plt.show()


def plot_model_comparison_radar(
    results_df: pd.DataFrame,
    metric_columns: Optional[List[str]] = None,
    top_n: int = 5,
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Radar/spider chart comparing top N models across multiple metrics.
    
    Args:
        results_df: DataFrame with models and metrics
        metric_columns: Specific metrics to use
        top_n: Number of top models to compare
        figsize: Figure size
        save_path: Path to save figure
    """
    if metric_columns is None:
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
        metric_columns = numeric_cols[:min(6, len(numeric_cols))]  # Max 6 metrics for readability
    
    heatmap_data = results_df.set_index(results_df.columns[0])[metric_columns].copy()
    heatmap_data = heatmap_data.astype(float)
    
    # Get top N models by mean score
    heatmap_data["_mean"] = heatmap_data.mean(axis=1)
    top_models = heatmap_data.nlargest(top_n, "_mean").drop("_mean", axis=1)
    
    # Normalize
    normalized = (top_models - top_models.min()) / (top_models.max() - top_models.min())
    
    angles = np.linspace(0, 2 * np.pi, len(metric_columns), endpoint=False).tolist()
    angles += angles[:1]  # Complete circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))
    
    colors = COLORS_PRIMARY[:len(normalized)]
    for idx, (model, color) in enumerate(zip(normalized.index, colors)):
        values = normalized.loc[model].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_columns, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_title("🌟 Top Models - Radar Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Radar chart saved: {save_path}")
    plt.show()


# ==============================================================================
# LEARNING CURVES & TRAINING DYNAMICS
# ==============================================================================

def plot_learning_curves(
    train_scores: List[float],
    val_scores: List[float],
    train_sizes: Optional[List[int]] = None,
    model_name: str = "Model",
    metric_name: str = "Accuracy",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot learning curves showing training vs validation scores.
    
    Args:
        train_scores: Training scores for each sample size
        val_scores: Validation scores for each sample size
        train_sizes: Sample sizes (if None, use indices)
        model_name: Name of the model
        metric_name: Name of the metric
        figsize: Figure size
        save_path: Path to save figure
    """
    if train_sizes is None:
        train_sizes = np.arange(1, len(train_scores) + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(train_sizes, train_scores, "o-", linewidth=2.5, markersize=8, 
           label="Training Score", color="#3498db")
    ax.plot(train_sizes, val_scores, "s-", linewidth=2.5, markersize=8, 
           label="Validation Score", color="#e74c3c")
    
    ax.fill_between(train_sizes, train_scores, val_scores, alpha=0.2, color="gray")
    
    ax.set_xlabel("Training Set Size", fontweight="bold", fontsize=12)
    ax.set_ylabel(metric_name, fontweight="bold", fontsize=12)
    ax.set_title(f"📈 Learning Curve - {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    
    # Add annotations for final scores
    ax.text(train_sizes[-1], train_scores[-1], f"{train_scores[-1]:.3f}", 
           fontsize=10, ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.text(train_sizes[-1], val_scores[-1], f"{val_scores[-1]:.3f}", 
           fontsize=10, ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Learning curve saved: {save_path}")
    plt.show()


# ==============================================================================
# FEATURE IMPORTANCE PLOTS
# ==============================================================================

def plot_feature_importance(
    importance_dict: Dict[str, float],
    model_name: str = "Model",
    top_k: int = 20,
    importance_type: str = "Percentage",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature importance/coefficients (sklearn or SHAP).
    
    Args:
        importance_dict: Dict mapping feature names to importance values
        model_name: Name of the model
        top_k: Number of top features to show
        importance_type: 'Percentage', 'SHAP', or 'Coefficients'
        figsize: Figure size
        save_path: Path to save figure
    """
    # Sort and select top K
    sorted_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k])
    features = list(sorted_dict.keys())
    values = list(sorted_dict.values())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel(importance_type, fontweight="bold", fontsize=12)
    ax.set_title(f"🔍 Top {top_k} Features - {model_name} ({importance_type})", 
                fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.2f}%", va="center", fontweight="bold", fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Feature importance plot saved: {save_path}")
    plt.show()


# ==============================================================================
# BATCH COMPARISON REPORT
# ==============================================================================

def generate_comparison_report(
    results_dict: Dict[str, Dict[str, Any]],
    save_dir: str = "./model_reports",
    task_type: str = "classification",
) -> None:
    """
    Generate comprehensive comparison report for all models.
    
    Creates PDF-like multi-page comparison with:
    - Summary statistics
    - Ranked comparisons
    - Individual model dashboards
    - Heatmaps and radar charts
    
    Args:
        results_dict: Dict with structure {model_name: {y_true, y_pred, y_proba, metrics}}
        save_dir: Directory to save all plots
        task_type: 'classification' or 'regression'
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"📊 Generating comparison report to: {save_dir}")
    
    # 1. Generate individual dashboards
    for model_name, results in results_dict.items():
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        plot_model_performance_dashboard(
            model_name=model_name,
            y_true=results["y_true"],
            y_pred=results["y_pred"],
            y_proba=results.get("y_proba"),
            task_type=task_type,
            metrics_dict=results.get("metrics"),
            save_path=f"{save_dir}/01_dashboard_{safe_name}.png"
        )
    
    # 2. Create comparison dataframe
    comparison_data = []
    for model_name, results in results_dict.items():
        metrics = results.get("metrics", {})
        comparison_data.append({"Model": model_name, **metrics})
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 3. Generate comparison plots
    if len(comparison_data) > 1:
        # Get first numeric column
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            plot_model_comparison(
                comparison_df,
                metric_column=numeric_cols[0],
                save_path=f"{save_dir}/02_comparison_bar.png"
            )
            
            if len(numeric_cols) > 1:
                plot_model_comparison_heatmap(
                    comparison_df,
                    metric_columns=numeric_cols[:6],
                    save_path=f"{save_dir}/03_comparison_heatmap.png"
                )
            
            if len(numeric_cols) >= 3:
                plot_model_comparison_radar(
                    comparison_df,
                    metric_columns=numeric_cols[:6],
                    top_n=min(5, len(comparison_data)),
                    save_path=f"{save_dir}/04_comparison_radar.png"
                )
    
    # Save comparison CSV
    comparison_df.to_csv(f"{save_dir}/comparison_summary.csv", index=False)
    print(f"✅ Report generated with {len(results_dict)} models in {save_dir}/")
    print(f"📄 Summary: {save_dir}/comparison_summary.csv")


if __name__ == "__main__":
    print("🎨 Visualization Module Ready!")
    print("   Use: from my_ml_toolkit.visualizations import plot_model_performance_dashboard")
