"""
🎨 INDUSTRY-GRADE VISUALIZATION GUIDE

Comprehensive examples showing how to use the visualization module
to create professional model comparison dashboards and reports.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import visualization module
from my_ml_toolkit.visualizations import (
    plot_model_performance_dashboard,
    plot_model_comparison,
    plot_model_comparison_heatmap,
    plot_model_comparison_radar,
    plot_learning_curves,
    plot_feature_importance,
    generate_comparison_report,
)


# ==============================================================================
# EXAMPLE 1: SINGLE MODEL PERFORMANCE DASHBOARD
# ==============================================================================

def example_single_model_dashboard():
    """
    Create a professional dashboard for a single model showing:
    - Confusion matrix
    - ROC-AUC curve
    - Precision-Recall curve
    - Calibration curve
    - Class distribution
    - Metrics summary
    """
    print("\n" + "=" * 80)
    print("📊 EXAMPLE 1: Single Model Performance Dashboard")
    print("=" * 80)
    
    # Load data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                               random_state=42, class_sep=1.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }
    
    # Plot dashboard
    plot_model_performance_dashboard(
        model_name="Random Forest Classifier",
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        task_type="classification",
        metrics_dict=metrics,
        save_path="plots/example_rf_dashboard.png"
    )


# ==============================================================================
# EXAMPLE 2: MULTI-MODEL COMPARISON
# ==============================================================================

def example_multi_model_comparison():
    """
    Compare multiple models with ranked bar charts and metrics.
    """
    print("\n" + "=" * 80)
    print("🏆 EXAMPLE 2: Multi-Model Comparison")
    print("=" * 80)
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    results = []
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results.append({
            "Model": model_name,
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        })
    
    results_df = pd.DataFrame(results)
    print("\n📊 Comparison Results:")
    print(results_df.to_string(index=False))
    
    # Plot comparison
    plot_model_comparison(
        results_df,
        metric_column="Test Accuracy",
        title="Breast Cancer Classification - Model Comparison",
        save_path="plots/example_model_comparison.png"
    )


# ==============================================================================
# EXAMPLE 3: COMPREHENSIVE HEATMAP COMPARISON
# ==============================================================================

def example_heatmap_comparison():
    """
    Create a heatmap showing all models vs all metrics.
    Useful for identifying which models perform best on which metrics.
    """
    print("\n" + "=" * 80)
    print("🔥 EXAMPLE 3: Comprehensive Heatmap Comparison")
    print("=" * 80)
    
    # Create sample results
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50),
    }
    
    results = []
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        })
    
    results_df = pd.DataFrame(results)
    
    # Plot heatmap
    plot_model_comparison_heatmap(
        results_df,
        figsize=(10, 4),
        save_path="plots/example_heatmap.png"
    )


# ==============================================================================
# EXAMPLE 4: RADAR CHART COMPARISON
# ==============================================================================

def example_radar_comparison():
    """
    Create a radar/spider chart for top N models across multiple metrics.
    Great for visual comparison of model capabilities.
    """
    print("\n" + "=" * 80)
    print("⭐ EXAMPLE 4: Radar Chart Comparison")
    print("=" * 80)
    
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    }
    
    results = []
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        })
    
    results_df = pd.DataFrame(results)
    
    # Plot radar
    plot_model_comparison_radar(
        results_df,
        top_n=3,
        figsize=(10, 10),
        save_path="plots/example_radar.png"
    )


# ==============================================================================
# EXAMPLE 5: LEARNING CURVES
# ==============================================================================

def example_learning_curves():
    """
    Plot learning curves showing how model performance changes with data size.
    Useful for identifying bias-variance tradeoff and data requirements.
    """
    print("\n" + "=" * 80)
    print("📈 EXAMPLE 5: Learning Curves")
    print("=" * 80)
    
    from sklearn.model_selection import learning_curve
    
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Calculate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train_scaled, y_train,
        cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    
    # Plot learning curves
    plot_learning_curves(
        train_scores=train_mean,
        val_scores=val_mean,
        train_sizes=train_sizes,
        model_name="Random Forest",
        metric_name="Accuracy",
        save_path="plots/example_learning_curve.png"
    )


# ==============================================================================
# EXAMPLE 6: FEATURE IMPORTANCE
# ==============================================================================

def example_feature_importance():
    """
    Plot top features from a model (sklearn or SHAP importance).
    """
    print("\n" + "=" * 80)
    print("🔍 EXAMPLE 6: Feature Importance")
    print("=" * 80)
    
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Extract importances
    importances = model.feature_importances_ * 100  # Convert to percentages
    importance_dict = dict(zip(feature_names, importances))
    
    # Plot
    plot_feature_importance(
        importance_dict,
        model_name="Random Forest - Breast Cancer",
        top_k=15,
        importance_type="Importance %",
        save_path="plots/example_feature_importance.png"
    )


# ==============================================================================
# EXAMPLE 7: FULL COMPARISON REPORT
# ==============================================================================

def example_full_comparison_report():
    """
    Generate a comprehensive comparison report with all visualization types.
    """
    print("\n" + "=" * 80)
    print("📊 EXAMPLE 7: Full Comparison Report")
    print("=" * 80)
    
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    }
    
    # Generate results dictionary
    results_dict = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        results_dict[model_name] = {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
            }
        }
    
    # Generate full report
    generate_comparison_report(
        results_dict,
        save_dir="model_reports",
        task_type="classification"
    )
    
    print("\n✅ Full report generated in 'model_reports/' directory!")
    print("   - Individual model dashboards")
    print("   - Comparison bar chart")
    print("   - Comparison heatmap")
    print("   - Radar chart")
    print("   - CSV summary")


# ==============================================================================
# USAGE WITH evaluate_and_plot_models
# ==============================================================================

def example_integration_with_evaluator():
    """
    Show how to use visualizations alongside the main evaluator function.
    """
    print("\n" + "=" * 80)
    print("🔗 EXAMPLE 8: Integration with evaluate_and_plot_models")
    print("=" * 80)
    
    from my_ml_toolkit.evaluator import evaluate_and_plot_models
    
    # Example code (pseudo-code):
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    }
    
    # Run evaluator (includes built-in plots)
    results = evaluate_and_plot_models(
        models=models,
        preprocess_pipeline=None,
        X=X_train,
        y=y_train,
        test_size=0.2,
        task_type="classification",
        cv=5,
        plot_diagnostics=True,
        plot_importance=True,
        plot_comparison=True,
        build_ensemble=True,
    )
    
    # After evaluator, generate additional comparison reports
    from my_ml_toolkit.visualizations import generate_comparison_report
    
    # Extract model results for custom visualization
    comparison_results = {
        model_name: {
            "y_true": y_test,
            "y_pred": results["predictions"][model_name],
            "y_proba": results.get("probabilities", {}).get(model_name),
            "metrics": results["scores"][model_name]
        }
        for model_name in models.keys()
        if model_name in results["scores"]
    }
    
    # Generate full report
    generate_comparison_report(comparison_results, save_dir="detailed_reports")
    """
    
    print("✅ See code comments for integration pattern!")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    os.makedirs("model_reports", exist_ok=True)
    
    print("\n" + "=" * 80)
    print("🎨 INDUSTRY-GRADE VISUALIZATION MODULE - EXAMPLES")
    print("=" * 80)
    print("\nChoose an example to run:")
    print("  1. Single Model Dashboard")
    print("  2. Multi-Model Comparison")
    print("  3. Heatmap Comparison")
    print("  4. Radar Chart Comparison")
    print("  5. Learning Curves")
    print("  6. Feature Importance")
    print("  7. Full Comparison Report")
    print("  8. Integration Guide")
    print("  0. Run All Examples")
    
    choice = input("\nEnter choice (0-8): ").strip()
    
    if choice == "1":
        example_single_model_dashboard()
    elif choice == "2":
        example_multi_model_comparison()
    elif choice == "3":
        example_heatmap_comparison()
    elif choice == "4":
        example_radar_comparison()
    elif choice == "5":
        example_learning_curves()
    elif choice == "6":
        example_feature_importance()
    elif choice == "7":
        example_full_comparison_report()
    elif choice == "8":
        example_integration_with_evaluator()
    elif choice == "0":
        example_single_model_dashboard()
        example_multi_model_comparison()
        example_heatmap_comparison()
        example_radar_comparison()
        example_learning_curves()
        example_feature_importance()
        example_full_comparison_report()
        print("\n✅ All examples completed!")
    else:
        print("❌ Invalid choice!")
