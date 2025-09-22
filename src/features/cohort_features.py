"""
Cohort-Level Feature Engineering Functions

Individual functions for creating cohort-level features for cohort profitability prediction.

Author: Generated for CloudWalk Case Study
"""

import pandas as pd
from .feature_utils import gini_coefficient, hhi_concentration, safe_divide


def create_loan_composition_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort loan composition features."""
    cohort_features = []

    for batch_letter, cohort_df in features_df.groupby("batch_letter"):
        features = {"batch_letter": batch_letter}

        # Basic count and totals
        features["cohort_size"] = len(cohort_df)
        features["total_loan_amount"] = cohort_df["loan_amount"].sum()
        features["avg_loan_amount"] = cohort_df["loan_amount"].mean()
        features["median_loan_amount"] = cohort_df["loan_amount"].median()

        # Interest rate statistics
        features["avg_interest_rate"] = cohort_df["annual_interest"].mean()
        features["median_interest_rate"] = cohort_df["annual_interest"].median()
        features["std_interest_rate"] = cohort_df["annual_interest"].std()

        cohort_features.append(features)

    return pd.DataFrame(cohort_features)


def create_loan_distribution_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort loan distribution and concentration features."""
    cohort_features = []

    for batch_letter, cohort_df in features_df.groupby("batch_letter"):
        features = {"batch_letter": batch_letter}

        # Loan amount concentration
        features["loan_amount_gini"] = gini_coefficient(cohort_df["loan_amount"])
        features["loan_amount_hhi"] = hhi_concentration(cohort_df["loan_amount"])

        # Loan size distribution percentiles
        features["loan_amount_p25"] = cohort_df["loan_amount"].quantile(0.25)
        features["loan_amount_p75"] = cohort_df["loan_amount"].quantile(0.75)
        features["loan_amount_p90"] = cohort_df["loan_amount"].quantile(0.90)

        # Coefficient of variation for loan amounts
        features["loan_amount_cv"] = safe_divide(
            cohort_df["loan_amount"].std(), cohort_df["loan_amount"].mean()
        )

        cohort_features.append(features)

    return pd.DataFrame(cohort_features)


def create_temporal_cohort_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features at cohort level."""
    cohort_features = []

    for batch_letter, cohort_df in features_df.groupby("batch_letter"):
        features = {"batch_letter": batch_letter}

        # Average timing metrics
        features["avg_days_since_loan_issuance"] = cohort_df[
            "days_since_loan_issuance"
        ].mean()
        features["avg_days_allowlist_to_loan"] = cohort_df[
            "days_allowlist_to_loan"
        ].mean()

        # Timing distribution
        features["std_days_since_issuance"] = cohort_df[
            "days_since_loan_issuance"
        ].std()
        features["std_days_allowlist_to_loan"] = cohort_df[
            "days_allowlist_to_loan"
        ].std()

        cohort_features.append(features)

    return pd.DataFrame(cohort_features)


def create_repayment_cohort_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort-level repayment behavior features."""
    cohort_features = []

    for batch_letter, cohort_df in features_df.groupby("batch_letter"):
        features = {"batch_letter": batch_letter}

        # Repayment velocity aggregates
        for period in [30, 60, 90]:
            velocity_col = f"repayment_velocity_{period}d"
            roi_col = f"loan_roi_{period}d"

            if velocity_col in cohort_df.columns:
                features[f"avg_{velocity_col}"] = cohort_df[velocity_col].mean()
                features[f"median_{velocity_col}"] = cohort_df[velocity_col].median()

            if roi_col in cohort_df.columns:
                features[f"avg_{roi_col}"] = cohort_df[roi_col].mean()
                features[f"median_{roi_col}"] = cohort_df[roi_col].median()

        # First repayment timing
        if "days_to_first_repayment" in cohort_df.columns:
            features["avg_days_to_first_repayment"] = cohort_df[
                "days_to_first_repayment"
            ].mean()
            features["median_days_to_first_repayment"] = cohort_df[
                "days_to_first_repayment"
            ].median()

        # Repayment behavior consistency
        if "repayment_consistency_cv" in cohort_df.columns:
            features["avg_repayment_consistency"] = cohort_df[
                "repayment_consistency_cv"
            ].mean()

        # Proportion of loans with good/bad repayment behavior
        if "loan_roi_90d" in cohort_df.columns:
            good_performers = (cohort_df["loan_roi_90d"] >= 0).sum()
            features["pct_positive_roi_90d"] = good_performers / len(cohort_df)

        cohort_features.append(features)

    return pd.DataFrame(cohort_features)


def create_risk_distribution_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort risk distribution features."""
    cohort_features = []

    for batch_letter, cohort_df in features_df.groupby("batch_letter"):
        features = {"batch_letter": batch_letter}

        # Status distribution at decision time
        if "status_at_decision_time" in cohort_df.columns:
            status_counts = cohort_df["status_at_decision_time"].value_counts()
            total_loans = len(cohort_df)

            # Calculate proportions for key statuses
            features["pct_executed"] = status_counts.get("executed", 0) / total_loans
            features["pct_debt_collection"] = (
                status_counts.get("debt_collection", 0) / total_loans
            )
            features["pct_debt_repaid"] = (
                status_counts.get("debt_repaid", 0) / total_loans
            )
            features["pct_active"] = status_counts.get("active", 0) / total_loans

        # Billing features if available
        if "is_in_normal_repayment" in cohort_df.columns:
            features["pct_normal_repayment"] = cohort_df[
                "is_in_normal_repayment"
            ].mean()

        cohort_features.append(features)

    return pd.DataFrame(cohort_features)


def create_interaction_cohort_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort-level interaction features."""
    cohort_features = []

    for batch_letter, cohort_df in features_df.groupby("batch_letter"):
        features = {"batch_letter": batch_letter}

        # Average interaction terms
        if "loan_amount_x_interest" in cohort_df.columns:
            features["avg_loan_amount_x_interest"] = cohort_df[
                "loan_amount_x_interest"
            ].mean()

        # Risk-adjusted metrics (if we have ROI data)
        if "loan_roi_90d" in cohort_df.columns:
            # Average ROI weighted by loan amount
            total_amount = cohort_df["loan_amount"].sum()
            if total_amount > 0:
                features["amount_weighted_avg_roi_90d"] = (
                    cohort_df["loan_roi_90d"] * cohort_df["loan_amount"]
                ).sum() / total_amount

        cohort_features.append(features)

    return pd.DataFrame(cohort_features)
