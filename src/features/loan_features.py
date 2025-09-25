"""
Loan-Level Feature Engineering Functions

Individual functions for creating loan-level features for cohort profitability prediction.

Author: Generated for CloudWalk Case Study
"""

import pandas as pd
import numpy as np
from .feature_utils import safe_qcut, get_measure_points


def create_transformed_loan_characteristics(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create basic loan characteristic features."""
    result_df = features_df.copy()

    # log-transformed loan amounts
    result_df["loan_amount_log"] = np.log1p(
        result_df["loan_amount"]
    )  # log(1+x) to handle zeros

    return result_df


def create_loan_size_decile(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create loan size decile within cohort."""
    result_df = features_df.copy()

    # Loan size decile within cohort - handle cases with insufficient unique values
    result_df["loan_size_decile"] = result_df.groupby("batch_letter")[
        "loan_amount"
    ].transform(safe_qcut)

    return result_df


def create_temporal_features(
    features_df: pd.DataFrame, decision_time_days: int
) -> pd.DataFrame:
    """Create temporal features related to loan timing."""
    result_df = features_df.copy()

    # Days since loan issuance to decision time
    result_df["days_since_loan_issuance"] = (
        decision_time_days - result_df["created_at_h_days"]
    )

    # Days from allowlist to loan creation
    result_df["days_allowlist_to_loan"] = (
        result_df["created_at"] - result_df["allowlisted_date"]
    ).dt.days

    return result_df


def create_interaction_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction term features."""
    result_df = features_df.copy()

    # Loan amount × interest rate interaction
    result_df["loan_amount_x_interest"] = (
        result_df["loan_amount"] * result_df["annual_interest"]
    )

    return result_df


def create_repayment_behavior_features(
    repayments_filtered: pd.DataFrame,
    loans_last_update_df: pd.DataFrame,
    decision_time_days: int,
) -> pd.DataFrame:
    """Create repayment behavior features for each loan using vectorized operations."""

    # Merge loan info for vectorized operations
    # Ensure we have created_at as datetime
    loans_df_prep = loans_last_update_df.copy()

    # Sort by date for each loan (needed for various calculations)
    repayments_filtered = repayments_filtered.sort_values(["loan_id", "h_days"])

    # --- Calculate days to first repayment ---
    # For each loan, calculate days from created_at to first repayment
    first_repayment_dates = repayments_filtered.groupby("loan_id").first()
    first_repayment_dates["days_to_first_repayment"] = (
        first_repayment_dates["date"] - first_repayment_dates["created_at"]
    ).dt.days
    days_to_first_repayment_df = first_repayment_dates[["days_to_first_repayment"]]

    # --- Calculate total repaid until decision time ---
    total_repaid_until_decision = (
        repayments_filtered.groupby("loan_id")["repayment_total"]
        .agg(["count", "sum"])
        .rename(columns={"count": "num_repayments", "sum": "total_repaid_amount"})
        .reset_index()
    )

    # --- Calculate repayment velocity and ROI for different periods ---
    velocity_roi_results = []
    measure_points = get_measure_points(decision_time_days)

    for period in measure_points:
        # Filter repayments within the period
        period_data = repayments_filtered[
            repayments_filtered["h_days"] <= period
        ].copy()

        period_data["delta_days_from_creation_to_period_of_measurement"] = period_data[
            "created_at_h_days"
        ].apply(lambda x: (period - x) if pd.notna(x) else np.nan)

        # Calculate total repaid per loan within period
        period_repaid = (
            period_data.groupby("loan_id")
            .aggregate(
                {
                    "repayment_total": "sum",
                    "delta_days_from_creation_to_period_of_measurement": "first",
                    "loan_amount": "first",  # All values will be the same inside loan
                }
            )
            .reset_index()
        )
        period_repaid = period_repaid.rename(
            columns={"repayment_total": f"total_repaid_{period}d"}
        )

        # Calculate velocity and ROI
        period_repaid[f"repayment_velocity_{period}d"] = (
            period_repaid[f"total_repaid_{period}d"]
            / period_repaid["delta_days_from_creation_to_period_of_measurement"]
        )
        period_repaid[f"loan_roi_{period}d"] = np.where(
            period_repaid["loan_amount"] > 0,
            (period_repaid[f"total_repaid_{period}d"] / period_repaid["loan_amount"])
            - 1,
            np.nan,
        )

        # Store results
        velocity_roi_results.append(
            period_repaid[
                ["loan_id", f"repayment_velocity_{period}d", f"loan_roi_{period}d"]
            ]
        )

    # Combine all period results
    velocity_roi_df = velocity_roi_results[0]
    for df in velocity_roi_results[1:]:
        velocity_roi_df = velocity_roi_df.merge(df, on="loan_id", how="outer")

    # --- Calculate repayment consistency (CV) ---
    def calc_consistency(group):
        amounts = group["repayment_total"].values
        amounts = amounts[amounts > 0]  # Exclude zero payments
        if len(amounts) > 1:
            return np.std(amounts) / np.mean(amounts)
        elif len(amounts) == 1:
            return 0.0
        else:
            # return np.nan
            return 0  # I think it makes more sense to consider 0 consistency when there is no repayment

    repayment_cv = (
        repayments_filtered.groupby("loan_id").apply(calc_consistency).reset_index()
    )
    repayment_cv.columns = ["loan_id", "repayment_consistency_cv"]

    # --- Calculate average repayment relative to loan size ---
    avg_repayments = (
        repayments_filtered.groupby("loan_id")
        .agg(
            {
                "repayment_total": "mean",
                "loan_amount": "first",  # Get loan_amount for each loan
            }
        )
        .reset_index()
    )
    avg_repayments["avg_repayment_relative"] = np.where(
        avg_repayments["loan_amount"] > 0,
        avg_repayments["repayment_total"] / avg_repayments["loan_amount"],
        0,
    )
    avg_relative_df = avg_repayments[["loan_id", "avg_repayment_relative"]]

    # --- Combine all features ---
    # Merge all calculated features
    result_df = velocity_roi_df
    for df in [
        days_to_first_repayment_df.reset_index(),
        total_repaid_until_decision,
        repayment_cv,
        avg_relative_df,
    ]:
        result_df = result_df.merge(df, on="loan_id", how="outer")

    # --- Handle loans with no repayments ---
    # Find loans without repayments
    all_loan_ids = set(loans_df_prep["loan_id"])
    loans_with_repayments = set(result_df["loan_id"])
    loans_without_repayments = all_loan_ids - loans_with_repayments

    if loans_without_repayments:
        # Create dataframe for loans without repayments
        no_repay_data = {
            "loan_id": list(loans_without_repayments),
            "days_to_first_repayment": [np.nan] * len(loans_without_repayments),
            "repayment_consistency_cv": [np.nan] * len(loans_without_repayments),
            "avg_repayment_relative": [0] * len(loans_without_repayments),
            "total_repaid_amount": [0] * len(loans_without_repayments),
            "num_repayments": [0] * len(loans_without_repayments),
        }

        # Add velocity and ROI features
        for period in measure_points:
            if period <= decision_time_days:
                no_repay_data[f"repayment_velocity_{period}d"] = [0] * len(
                    loans_without_repayments
                )
                no_repay_data[f"loan_roi_{period}d"] = [-1] * len(
                    loans_without_repayments
                )

        no_repay_df = pd.DataFrame(no_repay_data)

        # Combine with existing results
        result_df = pd.concat([result_df, no_repay_df], ignore_index=True)

    return result_df
