"""
Feature Engineering Utilities

Shared utilities for feature engineering across loan-level and cohort-level features.

Author: Generated for CloudWalk Case Study
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Tuple


def prepare_loan_history_data(
    loans_and_cohort: pd.DataFrame, decision_time_days: int
) -> pd.DataFrame:
    """
    Prepare loan history data for feature engineering.

    Args:
        loans_and_cohort: DataFrame with loan and cohort information (contains loan status history)
        decision_time_days: Decision time t in days after cohort creation

    Returns:
        loan_status_at_decision: DataFrame with the status of each loan as of decision time
    """
    loans_history = loans_and_cohort.copy()

    # Defining the cohort start date for each loan
    loans_history["cohort_start"] = loans_history.groupby("batch_letter")[
        "allowlisted_date"
    ].transform("min")

    # Leaving just loans information until decision time
    loans_history_before_decision = loans_history[
        loans_history["updated_at_h_days"] <= decision_time_days
    ]

    # Get the loan status as of decision time (last update before decision cutoff)
    loan_status_at_decision = (
        loans_history_before_decision.sort_values("updated_at_h_days")
        .groupby("loan_id")
        .last()
        .reset_index()
    )

    return loan_status_at_decision


def get_unique_loans_from_history(loans_and_cohort: pd.DataFrame) -> pd.DataFrame:
    """Get unique loans from loan history data (earliest record per loan)."""
    loans_history = loans_and_cohort.copy()
    loans_history["updated_at"] = pd.to_datetime(loans_history["updated_at"])
    unique_loans = (
        loans_history.sort_values("updated_at").groupby("loan_id").first().reset_index()
    )
    return unique_loans


def calculate_hhi(values: np.ndarray) -> float:
    """Calculate Herfindahl-Hirschman Index for concentration measurement."""
    if len(values) == 0 or values.sum() == 0:
        return 0

    # Calculate market shares (proportions)
    total = values.sum()
    shares = values / total

    # HHI is sum of squared shares
    hhi = np.sum(shares**2)

    return hhi


def hhi_concentration(values: pd.Series) -> float:
    """Calculate HHI concentration for a pandas Series."""
    return calculate_hhi(np.array(values.values))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator


def safe_qcut(x, q=10):
    """Safely perform quantile cuts, handling cases with insufficient unique values."""
    try:
        return pd.qcut(x, q=q, labels=False, duplicates="drop") + 1
    except ValueError:
        # If can't create q quantiles, create as many as possible
        n_unique = len(x.drop_duplicates())
        if n_unique <= 1:
            return pd.Series([1] * len(x), index=x.index)
        else:
            q_adj = min(q, n_unique)
            return pd.qcut(x, q=q_adj, labels=False, duplicates="drop") + 1


def save_features_to_database(
    database_path: str,
    decision_time_days: int,
    time_horizon_days: int,
    loan_features_df: pd.DataFrame | None = None,
    cohort_features_df: pd.DataFrame | None = None,
) -> None:
    """Save feature tables to the database."""

    loan_table_name = f"loan_features_t{decision_time_days}_h{time_horizon_days}"
    cohort_table_name = f"cohort_features_t{decision_time_days}_h{time_horizon_days}"

    with sqlite3.connect(database_path) as conn:
        # Save loan-level features
        if loan_features_df is not None:
            loan_features_df.to_sql(
                loan_table_name, conn, if_exists="replace", index=False
            )
            print(
                f"Saved {len(loan_features_df)} loan features to table: {loan_table_name}"
            )

        # Save cohort-level features
        if cohort_features_df is not None:
            cohort_features_df.to_sql(
                cohort_table_name, conn, if_exists="replace", index=False
            )
            print(
                f"Saved {len(cohort_features_df)} cohort features to table: {cohort_table_name}"
            )


def load_features_from_database(
    database_path: str,
    decision_time_days: int,
    time_horizon_days: int,
    cohort_features: bool = True,
    loan_features: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature tables from the database."""

    loan_table_name = f"loan_features_t{decision_time_days}_h{time_horizon_days}"
    cohort_table_name = f"cohort_features_t{decision_time_days}_h{time_horizon_days}"

    loan_features_df = pd.DataFrame()
    cohort_features_df = pd.DataFrame()
    with sqlite3.connect(database_path) as conn:
        if loan_features:
            print(f"Loading loan features from table: {loan_table_name}")
            loan_features_df = pd.read_sql_query(
                f"SELECT * FROM {loan_table_name}", conn
            )
        if cohort_features:
            print(f"Loading cohort features from table: {cohort_table_name}")
            cohort_features_df = pd.read_sql_query(
                f"SELECT * FROM {cohort_table_name}", conn
            )

    return loan_features_df, cohort_features_df


def get_measure_points(decision_time_days: int, n_points: int = 3) -> list[int]:
    """Get n equally spaced measure points in a space of decision_time_days."""
    step = decision_time_days // n_points
    measure_points = np.arange(step, decision_time_days + 1, step).tolist()
    return measure_points
