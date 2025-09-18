import pandas as pd
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


def plot_roi_curves(
    repayment_curve: pd.DataFrame,
    ax: Axes | None = None,
    extra_title: str | None = None,
    show_legend: bool = True,
) -> Axes:
    """
    Plots ROI curves for each cohort in the repayment_curve DataFrame.

    Args:
        repayment_curve (pd.DataFrame): DataFrame containing 'batch', 'h_days', and 'ROI' columns.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """

    if ax is None:
        _, ax = plt.subplots()

    for batch, g in repayment_curve.groupby("batch"):
        ax.plot(g["h_days"], g["ROI"], label=f"Cohort {batch[:10]}...")  # type: ignore

    ax.set_title("ROI by Cohort" + (f" - {extra_title}" if extra_title else ""))
    ax.set_xlabel("H (days since cohort start)")
    ax.set_ylabel("ROI")
    ax.grid(which="both", axis="x")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    if show_legend:
        ax.legend(title="Cohort", bbox_to_anchor=(1.05, 1), loc="upper left")
    return ax
