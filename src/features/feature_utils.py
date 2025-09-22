"""
Feature Engineering Utilities

Shared utilities for feature engineering across loan-level and cohort-level features.

Author: Generated for CloudWalk Case Study
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Tuple


def filter_repayments_by_decision_time(
    repayments_and_loans: pd.DataFrame,
    loans_with_cutoff: pd.DataFrame,
    decision_time_days: int,
) -> pd.DataFrame:
    """Filter repayments to only include those up to decision time for each loan."""

    # Prepare repayments data
    repayments_filtered = repayments_and_loans.copy()
    repayments_filtered["date"] = pd.to_datetime(repayments_filtered["date"])

    # Merge with cutoff dates
    repayments_with_cutoff = repayments_filtered.merge(
        loans_with_cutoff[["loan_id", "decision_cutoff_date"]], on="loan_id", how="left"
    )

    # Filter to only repayments before decision time
    repayments_filtered = repayments_with_cutoff[
        repayments_with_cutoff["date"] <= repayments_with_cutoff["decision_cutoff_date"]
    ].copy()

    return repayments_filtered


def prepare_loan_history_data(
    loans_and_cohort: pd.DataFrame, decision_time_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare loan history data for feature engineering.

    Args:
        loans_and_cohort: DataFrame with loan and cohort information (contains loan status history)
        decision_time_days: Decision time t in days after cohort creation

    Returns:
        Tuple of (loans_history, loan_creation_records, loan_status_at_decision)
    """
    loans_history = loans_and_cohort.copy()
    loans_history["created_at"] = pd.to_datetime(loans_history["created_at"])
    loans_history["allowlisted_date"] = pd.to_datetime(
        loans_history["allowlisted_date"]
    )
    loans_history["updated_at"] = pd.to_datetime(loans_history["updated_at"])

    # Calculate decision time cutoff for each loan record
    loans_history["decision_cutoff_date"] = loans_history["allowlisted_date"].apply(
        lambda x: x + pd.Timedelta(days=decision_time_days)
    )

    # Get the loan creation record (earliest record per loan) for basic characteristics
    loan_creation_records = (
        loans_history.sort_values("updated_at").groupby("loan_id").first().reset_index()
    )

    # Get the loan status as of decision time (last update before decision cutoff)
    # Remove timezone info to avoid comparison issues
    loans_history["updated_at_naive"] = loans_history["updated_at"].dt.tz_localize(None)
    loans_history["decision_cutoff_date_naive"] = loans_history["decision_cutoff_date"]
    loans_before_decision = loans_history[
        loans_history["updated_at_naive"] <= loans_history["decision_cutoff_date_naive"]
    ]
    loan_status_at_decision = (
        loans_before_decision.sort_values("updated_at")
        .groupby("loan_id")
        .last()
        .reset_index()
    )

    return loans_history, loan_creation_records, loan_status_at_decision


def get_unique_loans_from_history(loans_and_cohort: pd.DataFrame) -> pd.DataFrame:
    """Get unique loans from loan history data (earliest record per loan)."""
    loans_history = loans_and_cohort.copy()
    loans_history["updated_at"] = pd.to_datetime(loans_history["updated_at"])
    unique_loans = (
        loans_history.sort_values("updated_at").groupby("loan_id").first().reset_index()
    )
    return unique_loans


def calculate_gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient for concentration measurement."""
    if len(values) == 0:
        return 0

    # Sort values
    sorted_values = np.sort(values)
    n = len(values)

    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (
        n + 1
    ) / n

    return gini


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


def gini_coefficient(values: pd.Series) -> float:
    """Calculate Gini coefficient for a pandas Series."""
    return calculate_gini_coefficient(np.array(values.values))


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
    loan_features_df: pd.DataFrame,
    cohort_features_df: pd.DataFrame,
    database_path: str,
    decision_time_days: int,
) -> None:
    """Save feature tables to the database."""

    loan_table_name = f"loan_features_t{decision_time_days}"
    cohort_table_name = f"cohort_features_t{decision_time_days}"

    with sqlite3.connect(database_path) as conn:
        # Save loan-level features
        loan_features_df.to_sql(loan_table_name, conn, if_exists="replace", index=False)
        print(
            f"Saved {len(loan_features_df)} loan features to table: {loan_table_name}"
        )

        # Save cohort-level features
        cohort_features_df.to_sql(
            cohort_table_name, conn, if_exists="replace", index=False
        )
        print(
            f"Saved {len(cohort_features_df)} cohort features to table: {cohort_table_name}"
        )


def load_features_from_database(
    database_path: str, decision_time_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature tables from the database."""

    loan_table_name = f"loan_features_t{decision_time_days}"
    cohort_table_name = f"cohort_features_t{decision_time_days}"

    with sqlite3.connect(database_path) as conn:
        loan_features_df = pd.read_sql_query(f"SELECT * FROM {loan_table_name}", conn)
        cohort_features_df = pd.read_sql_query(
            f"SELECT * FROM {cohort_table_name}", conn
        )

    return loan_features_df, cohort_features_df
