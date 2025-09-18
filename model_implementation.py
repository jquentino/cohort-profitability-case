def run_ROI_over_time_eda(database_path: str) -> tuple[float, tuple[float, float]]:
    """
    Run ROI over time EDA.
    Args:
        database_path: Path to the database file. Considers all the loans, loan repayments, and transactions corresponding to a same batch.
    Returns:
        Tuple containing:
            - The expected ROI at the horizon.
            - Tuple containing the lower and upper bounds of the confidence interval.
    """
    ...
