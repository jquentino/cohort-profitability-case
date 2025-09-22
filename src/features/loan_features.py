"""
Loan-Level Feature Engineering Functions

Individual functions for creating loan-level features for cohort profitability prediction.

Author: Generated for CloudWalk Case Study
"""

import pandas as pd
import numpy as np
from .feature_utils import safe_qcut


def create_loan_characteristics(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create basic loan characteristic features."""
    result_df = features_df.copy()

    # Raw and log-transformed loan amounts
    result_df["loan_amount_raw"] = result_df["loan_amount"]
    result_df["loan_amount_log"] = np.log1p(
        result_df["loan_amount"]
    )  # log(1+x) to handle zeros

    # Interest rate
    result_df["annual_interest_rate"] = result_df["annual_interest"]

    return result_df


def create_loan_size_decile(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create loan size decile within cohort."""
    result_df = features_df.copy()

    # Loan size decile within cohort - handle cases with insufficient unique values
    result_df["loan_size_decile"] = result_df.groupby("batch_letter")[
        "loan_amount"
    ].transform(safe_qcut)

    return result_df


def create_temporal_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features related to loan timing."""
    result_df = features_df.copy()

    # Days since loan issuance to decision time
    result_df["days_since_loan_issuance"] = (
        result_df["decision_cutoff_date"] - result_df["created_at"]
    ).dt.days

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
        features.update(
            calculate_days_to_first_repayment(loan_repayments, loan_created)
        )

        # Repayment velocity for different periods
        features.update(
            calculate_repayment_velocity(
                loan_repayments, loan_created, loan_amount, decision_time_days
            )
        )

        # Repayment consistency metrics
        features.update(calculate_repayment_consistency(loan_repayments))

        # Average repayment relative to loan size
        features.update(calculate_avg_repayment_relative(loan_repayments, loan_amount))

        # Repayment acceleration/deceleration
        features.update(
            calculate_repayment_acceleration(
                loan_repayments, loan_created, decision_time_days
            )
        )

        loan_repayment_stats.append(features)

    # Handle loans with no repayments
    loan_repayment_stats.extend(
        create_no_repayment_features(loans_df, loan_repayment_stats, decision_time_days)
    )

    return pd.DataFrame(loan_repayment_stats)


def calculate_days_to_first_repayment(
    loan_repayments: pd.DataFrame, loan_created: pd.Timestamp
) -> dict:
    """Calculate days to first repayment."""
    if len(loan_repayments) > 0:
        first_repayment_date = loan_repayments.iloc[0]["date"]
        return {"days_to_first_repayment": (first_repayment_date - loan_created).days}
    else:
        return {"days_to_first_repayment": np.nan}


def calculate_repayment_velocity(
    loan_repayments: pd.DataFrame,
    loan_created: pd.Timestamp,
    loan_amount: float,
    decision_time_days: int,
) -> dict:
    """Calculate repayment velocity and ROI for different periods."""
    features = {}

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

    return features


def calculate_repayment_consistency(loan_repayments: pd.DataFrame) -> dict:
    """Calculate repayment consistency (coefficient of variation)."""
    if len(loan_repayments) > 1:
        repayment_amounts = loan_repayments["repayment_total"].values.astype(float)
        repayment_amounts = repayment_amounts[
            repayment_amounts > 0
        ]  # Exclude zero payments
        if len(repayment_amounts) > 1:
            return {
                "repayment_consistency_cv": float(
                    repayment_amounts.std() / repayment_amounts.mean()
                )
            }
        else:
            return {"repayment_consistency_cv": 0.0}
    else:
        return {"repayment_consistency_cv": np.nan}


def calculate_avg_repayment_relative(
    loan_repayments: pd.DataFrame, loan_amount: float
) -> dict:
    """Calculate average repayment amount relative to loan size."""
    if len(loan_repayments) > 0:
        avg_repayment = loan_repayments["repayment_total"].mean()
        return {
            "avg_repayment_relative": avg_repayment / loan_amount
            if loan_amount > 0
            else 0
        }
    else:
        return {"avg_repayment_relative": 0}


def calculate_repayment_acceleration(
    loan_repayments: pd.DataFrame, loan_created: pd.Timestamp, decision_time_days: int
) -> dict:
    """Calculate repayment acceleration/deceleration trends."""
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
            return {"repayment_acceleration": late_velocity / early_velocity}
        else:
            return {"repayment_acceleration": np.nan if late_velocity == 0 else np.inf}
    else:
        return {"repayment_acceleration": np.nan}


def create_no_repayment_features(
    loans_df: pd.DataFrame, loan_repayment_stats: list, decision_time_days: int
) -> list:
    """Create feature records for loans with no repayments."""
    all_loan_ids = set(loans_df["loan_id"])
    loans_with_repayments = set([stat["loan_id"] for stat in loan_repayment_stats])
    loans_without_repayments = all_loan_ids - loans_with_repayments

    no_repayment_features = []
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

        no_repayment_features.append(features)

    return no_repayment_features


def create_billing_features(
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
