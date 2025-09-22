"""
Feature Engineering Module for Cohort Profitability Prediction

This module contains functions to create loan-level and cohort-level features
for predicting ROI at horizon H using only information available up to decision time t.

Author: Generated for CloudWalk Case Study
"""

import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
from typing import Tuple


def create_loan_level_features(
    loans_and_cohort: pd.DataFrame,
    repayments_and_loans: pd.DataFrame,
    decision_time_days: int = 90,
) -> pd.DataFrame:
    """
    Create loan-level features using only information available up to decision time t.

    Args:
        loans_and_cohort: DataFrame with loan and cohort information (contains loan status history)
        repayments_and_loans: DataFrame with repayment and loan information
        decision_time_days: Decision time t in days after cohort creation

    Returns:
        DataFrame with loan-level features (one record per unique loan)
    """
    print(
        f"Creating loan-level features with decision time = {decision_time_days} days"
    )

    # CRITICAL FIX: Handle loan status history properly
    # loans_and_cohort contains multiple records per loan (status history)
    # We need to get unique loans with their characteristics

    loans_history = loans_and_cohort.copy()
    loans_history["created_at"] = pd.to_datetime(loans_history["created_at"])
    loans_history["allowlisted_date"] = pd.to_datetime(
        loans_history["allowlisted_date"]
    )
    loans_history["updated_at"] = pd.to_datetime(loans_history["updated_at"])

    # Calculate decision time cutoff for each loan record
    loans_history["decision_cutoff_date"] = loans_history[
        "allowlisted_date"
    ] + pd.Timedelta(days=decision_time_days)

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

    # Start with creation records and merge decision time status
    features_df = loan_creation_records[
        [
            "loan_id",
            "user_id",
            "created_at",
            "loan_amount",
            "annual_interest",
            "allowlisted_date",
            "batch",
            "batch_letter",
            "decision_cutoff_date",
        ]
    ].copy()

    # Add status at decision time
    status_at_decision = loan_status_at_decision[
        ["loan_id", "status", "updated_at"]
    ].rename(
        columns={
            "status": "status_at_decision_time",
            "updated_at": "last_update_before_decision",
        }
    )
    features_df = features_df.merge(status_at_decision, on="loan_id", how="left")

    print(
        f"Processing {len(features_df)} unique loans (reduced from {len(loans_and_cohort)} historical records)"
    )

    # Calculate decision time cutoff for each loan
    # === LOAN CHARACTERISTICS ===
    features_df["loan_amount_raw"] = features_df["loan_amount"]
    features_df["loan_amount_log"] = np.log1p(
        features_df["loan_amount"]
    )  # log(1+x) to handle zeros
    features_df["annual_interest_rate"] = features_df["annual_interest"]

    # Loan size decile within cohort - handle cases with insufficient unique values
    def safe_qcut(x):
        try:
            return pd.qcut(x, q=10, labels=False, duplicates="drop") + 1
        except ValueError:
            # If can't create 10 deciles, create as many as possible
            n_unique = len(x.drop_duplicates())
            if n_unique <= 1:
                return pd.Series([1] * len(x), index=x.index)
            else:
                q = min(10, n_unique)
                return pd.qcut(x, q=q, labels=False, duplicates="drop") + 1

    features_df["loan_size_decile"] = features_df.groupby("batch_letter")[
        "loan_amount"
    ].transform(safe_qcut)

    # === TEMPORAL FEATURES ===
    features_df["days_since_loan_issuance"] = (
        features_df["decision_cutoff_date"] - features_df["created_at"]
    ).dt.days
    features_df["days_allowlist_to_loan"] = (
        features_df["created_at"] - features_df["allowlisted_date"]
    ).dt.days

    # === INTERACTION TERMS ===
    features_df["loan_amount_x_interest"] = (
        features_df["loan_amount"] * features_df["annual_interest"]
    )

    # === REPAYMENT-BASED FEATURES ===
    # Filter repayments up to decision time
    repayments_filtered = _filter_repayments_by_decision_time(
        repayments_and_loans, features_df, decision_time_days
    )

    # Add repayment behavior features
    repayment_features = _create_repayment_behavior_features(
        repayments_filtered, features_df, decision_time_days
    )

    # Merge repayment features
    features_df = features_df.merge(repayment_features, on="loan_id", how="left")

    # === BILLING INDICATORS ===
    billing_features = _create_billing_features(features_df, decision_time_days)
    features_df = features_df.merge(billing_features, on="loan_id", how="left")

    # Clean up intermediate columns
    feature_columns = [
        col
        for col in features_df.columns
        if not col.endswith("_temp")
        and col
        not in ["created_at", "allowlisted_date", "updated_at", "decision_cutoff_date"]
    ]

    return features_df[feature_columns].copy()


def create_cohort_level_features(
    loans_and_cohort: pd.DataFrame,
    repayments_and_loans: pd.DataFrame,
    decision_time_days: int = 90,
) -> pd.DataFrame:
    """
    Create cohort-level features using only information available up to decision time t.

    Args:
        loans_and_cohort: DataFrame with loan and cohort information (contains loan status history)
        repayments_and_loans: DataFrame with repayment and loan information
        decision_time_days: Decision time t in days after cohort creation

    Returns:
        DataFrame with cohort-level features
    """
    print(
        f"Creating cohort-level features with decision time = {decision_time_days} days"
    )

    # CRITICAL FIX: Use unique loans only, not historical records
    # Get unique loans (earliest record per loan for characteristics)
    loans_history = loans_and_cohort.copy()
    loans_history["updated_at"] = pd.to_datetime(loans_history["updated_at"])
    unique_loans = (
        loans_history.sort_values("updated_at").groupby("loan_id").first().reset_index()
    )

    # Group by cohort using unique loans
    cohort_groups = unique_loans.groupby("batch_letter")

    features_list = []

    for batch_letter, cohort_loans in cohort_groups:
        cohort_features = {"batch_letter": batch_letter}

        # === PORTFOLIO CONCENTRATION METRICS ===
        loan_amounts = cohort_loans["loan_amount"].values.astype(float)

        # Basic size metrics
        cohort_features["cohort_size"] = len(cohort_loans)
        cohort_features["total_loan_amount"] = float(loan_amounts.sum())
        cohort_features["value_weighted_avg_amount"] = float(
            np.average(loan_amounts, weights=loan_amounts)
        )

        # Concentration metrics
        cohort_features["gini_coefficient"] = _calculate_gini_coefficient(loan_amounts)
        cohort_features["hhi_loan_amounts"] = _calculate_hhi(loan_amounts)

        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95]
        for p in percentiles:
            cohort_features[f"loan_amount_p{p}"] = float(np.percentile(loan_amounts, p))

        # === RISK DISTRIBUTION METRICS ===
        cohort_features["loan_amount_std"] = float(loan_amounts.std())
        cohort_features["loan_amount_skewness"] = float(stats.skew(loan_amounts))
        cohort_features["loan_amount_cv"] = (
            float(loan_amounts.std() / loan_amounts.mean())
            if loan_amounts.mean() > 0
            else 0.0
        )

        # Interest rate statistics
        interest_rates = cohort_loans["annual_interest"].values.astype(float)
        cohort_features["avg_interest_rate"] = float(interest_rates.mean())
        cohort_features["interest_rate_std"] = float(interest_rates.std())

        features_list.append(cohort_features)

    cohort_features_df = pd.DataFrame(features_list)

    return cohort_features_df


def _filter_repayments_by_decision_time(
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


def _create_repayment_behavior_features(
    repayments_filtered: pd.DataFrame, loans_df: pd.DataFrame, decision_time_days: int
) -> pd.DataFrame:
    """Create repayment behavior features for each loan."""

    # Prepare repayments with total amounts
    repayments_filtered["repayment_total"] = repayments_filtered[
        "repayment_amount"
    ].fillna(0) + repayments_filtered["billings_amount"].fillna(0)

    # Group by loan to calculate features
    loan_repayment_stats = []

    for loan_id, loan_repayments in repayments_filtered.groupby("loan_id"):
        # Get loan info
        loan_info = loans_df[loans_df["loan_id"] == loan_id].iloc[0]
        loan_amount = loan_info["loan_amount"]
        loan_created = pd.to_datetime(loan_info["created_at"])

        # Sort repayments by date
        loan_repayments = loan_repayments.sort_values("date")

        features = {"loan_id": loan_id}

        # Days to first repayment
        if len(loan_repayments) > 0:
            first_repayment_date = loan_repayments.iloc[0]["date"]
            features["days_to_first_repayment"] = (
                first_repayment_date - loan_created
            ).days
        else:
            features["days_to_first_repayment"] = np.nan

        # Repayment velocity for different periods
        for period in [30, 60, 90]:
            if period <= decision_time_days:
                period_cutoff = loan_created + pd.Timedelta(days=period)
                period_repayments = loan_repayments[
                    loan_repayments["date"] <= period_cutoff
                ]

                if len(period_repayments) > 0:
                    total_repaid = period_repayments["repayment_total"].sum()
                    features[f"repayment_velocity_{period}d"] = total_repaid / period
                    features[f"loan_roi_{period}d"] = (
                        (total_repaid / loan_amount) - 1 if loan_amount > 0 else np.nan
                    )
                else:
                    features[f"repayment_velocity_{period}d"] = 0
                    features[f"loan_roi_{period}d"] = -1  # No repayments = -100% ROI
            else:
                features[f"repayment_velocity_{period}d"] = np.nan
                features[f"loan_roi_{period}d"] = np.nan

        # Repayment consistency (coefficient of variation)
        if len(loan_repayments) > 1:
            repayment_amounts = loan_repayments["repayment_total"].values.astype(float)
            repayment_amounts = repayment_amounts[
                repayment_amounts > 0
            ]  # Exclude zero payments
            if len(repayment_amounts) > 1:
                features["repayment_consistency_cv"] = float(
                    repayment_amounts.std() / repayment_amounts.mean()
                )
            else:
                features["repayment_consistency_cv"] = 0.0
        else:
            features["repayment_consistency_cv"] = np.nan

        # Average repayment amount relative to loan size
        if len(loan_repayments) > 0:
            avg_repayment = loan_repayments["repayment_total"].mean()
            features["avg_repayment_relative"] = (
                avg_repayment / loan_amount if loan_amount > 0 else 0
            )
        else:
            features["avg_repayment_relative"] = 0

        # Repayment acceleration/deceleration (comparing first half vs second half of decision period)
        if decision_time_days >= 60:  # Only calculate if we have enough time
            mid_point = loan_created + pd.Timedelta(days=decision_time_days // 2)

            early_repayments = loan_repayments[loan_repayments["date"] <= mid_point]
            late_repayments = loan_repayments[loan_repayments["date"] > mid_point]

            early_velocity = (
                early_repayments["repayment_total"].sum() / (decision_time_days // 2)
                if len(early_repayments) > 0
                else 0
            )
            late_velocity = (
                late_repayments["repayment_total"].sum()
                / (decision_time_days - decision_time_days // 2)
                if len(late_repayments) > 0
                else 0
            )

            if early_velocity > 0:
                features["repayment_acceleration"] = late_velocity / early_velocity
            else:
                features["repayment_acceleration"] = (
                    np.nan if late_velocity == 0 else np.inf
                )
        else:
            features["repayment_acceleration"] = np.nan

        loan_repayment_stats.append(features)

    # Handle loans with no repayments
    all_loan_ids = set(loans_df["loan_id"])
    loans_with_repayments = set([stat["loan_id"] for stat in loan_repayment_stats])
    loans_without_repayments = all_loan_ids - loans_with_repayments

    for loan_id in loans_without_repayments:
        features = {
            "loan_id": loan_id,
            "days_to_first_repayment": np.nan,
            "repayment_consistency_cv": np.nan,
            "avg_repayment_relative": 0,
            "repayment_acceleration": np.nan,
        }

        # Set velocity and ROI features to 0/-1
        for period in [30, 60, 90]:
            features[f"repayment_velocity_{period}d"] = (
                0 if period <= decision_time_days else np.nan
            )
            features[f"loan_roi_{period}d"] = (
                -1 if period <= decision_time_days else np.nan
            )

        loan_repayment_stats.append(features)

    return pd.DataFrame(loan_repayment_stats)


def _create_billing_features(
    loans_df: pd.DataFrame, decision_time_days: int
) -> pd.DataFrame:
    """Create billing-related features."""

    billing_features = []

    for _, loan in loans_df.iterrows():
        features = {"loan_id": loan["loan_id"]}

        # Use status at decision time instead of final status
        status_at_decision = loan.get(
            "status_at_decision_time", loan.get("status", "executed")
        )

        # Time in billing process (simplified assumption)
        # Note: This is a simplified calculation - in reality, we'd need loan status history
        if status_at_decision in ["debt_collection", "debt_repaid"]:
            # Assume loan entered billing sometime before decision time
            # This is a simplification - ideally we'd have status change timestamps
            features["time_in_billing_days"] = (
                np.nan
            )  # Cannot calculate without status history
            features["is_in_normal_repayment"] = False
        else:
            features["time_in_billing_days"] = 0
            features["is_in_normal_repayment"] = True

        billing_features.append(features)

    return pd.DataFrame(billing_features)


def _calculate_gini_coefficient(values: np.ndarray) -> float:
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


def _calculate_hhi(values: np.ndarray) -> float:
    """Calculate Herfindahl-Hirschman Index for concentration measurement."""
    if len(values) == 0 or values.sum() == 0:
        return 0

    # Calculate market shares (proportions)
    total = values.sum()
    shares = values / total

    # HHI is sum of squared shares
    hhi = np.sum(shares**2)

    return hhi


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
