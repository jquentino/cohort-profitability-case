from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
import pandas as pd


def initial_default_tolerance_simulation_plot(
    tolerance_results: pd.DataFrame,
    max_tolerance_rate: float,
    cohort_to_test="A",
    ax: Axes | None = None,
):
    """
    Plots ROI at horizon H versus initial default rates for a specified cohort.

    Args:
        tolerance_results (dict): Dictionary with keys 'default_rates' and 'roi_values', containing lists or arrays of rates and corresponding ROI values.
        cohort_to_test (str, optional): Cohort identifier to display in the plot title.

    Returns:
        None: Displays the plot.
    """

    # Find the maximum default rate with non-negative ROI

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        tolerance_results["default_rates"],
        tolerance_results["roi_values"],
        "bo-",
        linewidth=2,
        markersize=8,
    )
    ax.axhline(
        y=0, color="red", linestyle="--", alpha=0.7, label="Break-even (ROI = 0)"
    )
    ax.axvline(
        x=max_tolerance_rate,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Max tolerable rate: {max_tolerance_rate:.1%}",
    )
    ax.set_xlabel("Initial Default Rate")
    ax.set_ylabel("ROI at Horizon H")
    ax.set_title(f"Default Tolerance Analysis - Cohort {cohort_to_test}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    return ax
