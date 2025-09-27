"""
Visualization utilities for model predictions and uncertainty analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple


def plot_predictions_with_uncertainty(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    uncertainty_results: Dict[str, np.ndarray],
    decision_time_days: int,
    batch_column: str = "batch_letter",
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Model Predictions with Uncertainty",
) -> plt.Figure:
    """
    Plot model predictions with uncertainty intervals.

    Args:
        X_test: Test feature matrix
        y_test: Test target values
        predictions: Model predictions
        uncertainty_results: Dictionary with uncertainty estimates
        decision_time_days: Decision time threshold
        batch_column: Column name for batch grouping
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for plotting
    plot_data = pd.DataFrame(
        {
            "h_days": X_test["h_days"],
            "actual": y_test,
            "predicted": predictions,
            "lower_ci": uncertainty_results["lower_ci"],
            "upper_ci": uncertainty_results["upper_ci"],
            "batch": X_test[batch_column] if batch_column in X_test.columns else "All",
        }
    )

    # Plot actual values
    if batch_column in X_test.columns:
        sns.scatterplot(
            data=plot_data,
            x="h_days",
            y="actual",
            hue="batch",
            alpha=0.6,
            s=50,
            ax=ax,
            # label="Actual ROI",
        )
        sns.lineplot(
            data=plot_data,
            x="h_days",
            y="predicted",
            hue="batch",
            linestyle="--",
            linewidth=2,
            ax=ax,
            legend=False,
        )
    else:
        ax.scatter(
            plot_data["h_days"],
            plot_data["actual"],
            alpha=0.6,
            s=50,
            label="Actual ROI",
            color="blue",
        )
        ax.plot(
            plot_data["h_days"],
            plot_data["predicted"],
            "--",
            linewidth=2,
            label="Predicted ROI",
            color="red",
        )

    # Add confidence intervals
    ax.fill_between(
        plot_data["h_days"],
        plot_data["lower_ci"],
        plot_data["upper_ci"],
        alpha=0.3,
        color="red",
        label="95% Confidence Interval",
    )

    # Add decision time line
    ax.axvline(
        decision_time_days,
        color="black",
        linestyle=":",
        linewidth=2,
        label="Decision Time",
    )

    ax.set_xlabel("h_days (Horizon Days)")
    ax.set_ylabel("ROI")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_uncertainty_analysis(
    X_test: pd.DataFrame,
    uncertainty_results: Dict[str, np.ndarray],
    decision_time_days: int,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plot uncertainty analysis with multiple subplots.

    Args:
        X_test: Test feature matrix
        uncertainty_results: Dictionary with uncertainty estimates
        decision_time_days: Decision time threshold
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    h_days = X_test["h_days"].values

    # Plot 1: Prediction standard deviation vs time
    axes[0].scatter(h_days, uncertainty_results["std"], alpha=0.6, color="green")
    axes[0].axvline(
        decision_time_days, color="black", linestyle=":", label="Decision Time"
    )
    axes[0].set_xlabel("h_days")
    axes[0].set_ylabel("Prediction Standard Deviation")
    axes[0].set_title("Uncertainty vs Time Horizon")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Confidence interval width vs time
    ci_width = uncertainty_results["upper_ci"] - uncertainty_results["lower_ci"]
    axes[1].scatter(h_days, ci_width, alpha=0.6, color="orange")
    axes[1].axvline(
        decision_time_days, color="black", linestyle=":", label="Decision Time"
    )
    axes[1].set_xlabel("h_days")
    axes[1].set_ylabel("95% CI Width")
    axes[1].set_title("Confidence Interval Width vs Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Distribution of prediction standard deviations
    axes[2].hist(
        uncertainty_results["std"],
        bins=20,
        alpha=0.7,
        color="purple",
        edgecolor="black",
    )
    axes[2].axvline(
        np.mean(uncertainty_results["std"]),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(uncertainty_results['std']):.4f}",
    )
    axes[2].set_xlabel("Prediction Standard Deviation")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Distribution of Uncertainty")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_bootstrap_distribution(
    uncertainty_results: Dict[str, np.ndarray],
    sample_indices: Optional[list] = None,
    n_samples: int = 5,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot bootstrap prediction distributions for selected samples.

    Args:
        uncertainty_results: Dictionary with uncertainty estimates
        sample_indices: Specific sample indices to plot. If None, random samples chosen.
        n_samples: Number of samples to plot if sample_indices is None
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    all_predictions = uncertainty_results["all_predictions"]

    if sample_indices is None:
        sample_indices = np.random.choice(
            all_predictions.shape[1],
            size=min(n_samples, all_predictions.shape[1]),
            replace=False,
        )

    fig, axes = plt.subplots(1, len(sample_indices), figsize=figsize)
    if len(sample_indices) == 1:
        axes = [axes]

    for i, sample_idx in enumerate(sample_indices):
        bootstrap_preds = all_predictions[:, sample_idx]

        axes[i].hist(
            bootstrap_preds, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[i].axvline(
            uncertainty_results["mean"][sample_idx],
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {uncertainty_results['mean'][sample_idx]:.3f}",
        )
        axes[i].axvline(
            uncertainty_results["lower_ci"][sample_idx],
            color="orange",
            linestyle="--",
            label=f"95% CI: [{uncertainty_results['lower_ci'][sample_idx]:.3f}, {uncertainty_results['upper_ci'][sample_idx]:.3f}]",
        )
        axes[i].axvline(
            uncertainty_results["upper_ci"][sample_idx], color="orange", linestyle="--"
        )

        axes[i].set_xlabel("Predicted ROI")
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"Bootstrap Distribution\nSample {sample_idx}")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_components(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    decision_time_days: int,
    batch_column: str = "batch_letter",
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot the components of the hybrid model (trend + residuals).

    Args:
        X: Feature matrix
        y: Target variable
        model: Fitted HybridROIModel
        decision_time_days: Decision time threshold
        batch_column: Column name for batch grouping
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Get model predictions and components
    full_predictions = model.predict(X)
    trend_predictions_boxcox = model.trend_model.predict(X)
    residual_predictions = model.residual_model.predict(X)

    # Transform trend back to original scale
    from scipy.special import inv_boxcox

    trend_predictions = inv_boxcox(trend_predictions_boxcox, model.boxcox_lambda) - 1.1

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Original data and full prediction
    if batch_column in X.columns:
        sns.lineplot(
            data=pd.DataFrame(
                {"h_days": X["h_days"], "ROI": y, "batch": X[batch_column]}
            ),
            x="h_days",
            y="ROI",
            hue="batch",
            alpha=0.6,
            ax=axes[0, 0],
        )
        sns.lineplot(
            data=pd.DataFrame(
                {
                    "h_days": X["h_days"],
                    "predicted": full_predictions,
                    "batch": X[batch_column],
                }
            ),
            x="h_days",
            y="predicted",
            hue="batch",
            linestyle="--",
            ax=axes[0, 0],
            legend=False,
        )
    else:
        axes[0, 0].plot(X["h_days"], y, alpha=0.6, label="Actual")
        axes[0, 0].plot(X["h_days"], full_predictions, "--", label="Predicted")

    axes[0, 0].axvline(
        decision_time_days, color="black", linestyle=":", label="Decision Time"
    )
    axes[0, 0].set_title("Full Model: Actual vs Predicted")
    axes[0, 0].set_ylabel("ROI")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Trend component
    if batch_column in X.columns:
        sns.lineplot(
            data=pd.DataFrame(
                {"h_days": X["h_days"], "ROI": y, "batch": X[batch_column]}
            ),
            x="h_days",
            y="ROI",
            hue="batch",
            alpha=0.3,
            ax=axes[0, 1],
        )
        sns.lineplot(
            data=pd.DataFrame(
                {
                    "h_days": X["h_days"],
                    "trend": trend_predictions,
                    "batch": X[batch_column],
                }
            ),
            x="h_days",
            y="trend",
            hue="batch",
            linestyle="-",
            linewidth=2,
            ax=axes[0, 1],
            legend=False,
        )
    else:
        axes[0, 1].plot(X["h_days"], y, alpha=0.3, label="Actual")
        axes[0, 1].plot(X["h_days"], trend_predictions, "-", linewidth=2, label="Trend")

    axes[0, 1].axvline(
        decision_time_days, color="black", linestyle=":", label="Decision Time"
    )
    axes[0, 1].set_title("Trend Component (Linear Model)")
    axes[0, 1].set_ylabel("ROI")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Residuals (predicted by LGBM model)
    axes[1, 0].scatter(X["h_days"], residual_predictions, alpha=0.6, s=20)
    axes[1, 0].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[1, 0].axvline(
        decision_time_days, color="black", linestyle=":", label="Decision Time"
    )
    axes[1, 0].set_title("Residual Predictions (LGBM)")
    axes[1, 0].set_xlabel("h_days")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Feature importance (if available)
    feature_importance = model.get_feature_importance()
    if feature_importance:
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())

        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1][:10]  # Top 10

        axes[1, 1].barh(range(len(sorted_idx)), [importances[i] for i in sorted_idx])
        axes[1, 1].set_yticks(range(len(sorted_idx)))
        axes[1, 1].set_yticklabels([features[i] for i in sorted_idx])
        axes[1, 1].set_title("Feature Importance (LGBM)")
        axes[1, 1].set_xlabel("Importance")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Feature importance\nnot available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Feature Importance")

    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
