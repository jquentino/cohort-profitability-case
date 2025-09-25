"""
Feature Engineering Module

High-level interface for creating loan-level and cohort-level features
for cohort profitability prediction.

This module provides the main functions that coordinate feature engineering
across all feature types, maintaining the same interface as the original
monolithic features.py file.

Author: Generated for CloudWalk Case Study
"""

import pandas as pd

from .feature_utils import (
    prepare_loan_history_data,
    get_unique_loans_from_history,
    save_features_to_database,
    load_features_from_database,
)
from .loan_features import (
    create_transformed_loan_characteristics,
    create_loan_size_decile,
    create_temporal_features,
    create_interaction_features,
    create_repayment_behavior_features,
)
from .cohort_features import (
    create_loan_composition_features,
    create_loan_distribution_features,
    create_temporal_cohort_features,
    create_repayment_cohort_features,
    create_repayment_performance_features,
    create_risk_distribution_features,
    create_interaction_cohort_features,
)


def create_loan_level_features(
    loans_and_cohort: pd.DataFrame,
    repayments_and_loans: pd.DataFrame,
    decision_time_days: int = 90,
    time_horizon_days: int = 180,
) -> pd.DataFrame:
    """
    Create comprehensive loan-level features for cohort profitability prediction.

    This function coordinates the creation of all loan-level features including:
    - Basic loan characteristics (amount, interest rate)
    - Temporal features (timing, loan age)
    - Repayment behavior features (velocity, consistency, ROI)
    - Interaction features
    - Billing/status features

    Args:
        loans_and_cohort: DataFrame with loan and cohort information (contains loan status history)
        repayments_and_loans: DataFrame with repayment information joined with loan data
        decision_time_days: Decision time t in days after cohort creation (default: 90)
        time_horizon_days: Time horizon H in days for target variable (default: 180)

    Returns:
        DataFrame with loan-level features, one row per unique loan
    """
    print(
        f"Creating loan-level features with decision time t={decision_time_days} days..."
    )

    # Prepare loan history data
    loan_status_at_decision = prepare_loan_history_data(
        loans_and_cohort, decision_time_days
    )

    # Start with loan creation records (basic characteristics)
    features_df = loan_status_at_decision.copy()

    # Add status at decision time
    features_df = features_df.rename(columns={"status": "status_at_decision_time"})

    print(f"Base features dataset: {len(features_df)} unique loans")

    # Create basic loan characteristics
    features_df = create_transformed_loan_characteristics(features_df)

    # Create loan size decile within cohort
    features_df = create_loan_size_decile(features_df)

    # Create temporal features
    features_df = create_temporal_features(features_df, decision_time_days)

    # Create interaction features
    features_df = create_interaction_features(features_df)

    # Create repayment behavior features
    print("Creating repayment behavior features...")
    repayments_filtered = repayments_and_loans[
        repayments_and_loans["h_days"] <= decision_time_days
    ]

    repayment_features = create_repayment_behavior_features(
        repayments_filtered, features_df, decision_time_days
    )

    # Merge repayment features
    features_df = features_df.merge(repayment_features, on="loan_id", how="left")

    # Creating the prediction target
    repayment_target = (
        repayments_and_loans[repayments_and_loans["h_days"] <= time_horizon_days]
        .groupby("loan_id")["repayment_total"]
        .sum()
        .reset_index()
        .rename(columns={"repayment_total": "repayment_at_H"})
    )

    features_df = features_df.merge(repayment_target, on="loan_id", how="left")

    print(
        f"Final loan features dataset: {len(features_df)} loans with {len(features_df.columns)} features"
    )

    return features_df


def create_cohort_level_features(
    loans_and_cohort: pd.DataFrame,
    repayments_and_loans: pd.DataFrame,
    decision_time_days: int = 90,
) -> pd.DataFrame:
    """
    Create comprehensive cohort-level features for cohort profitability prediction.

    This function coordinates the creation of all cohort-level features including:
    - Loan composition and distribution metrics
    - Risk concentration measures
    - Temporal patterns
    - Repayment behavior aggregates
    - Risk distribution features

    Args:
        loans_and_cohort: DataFrame with loan and cohort information (contains loan status history)
        repayments_and_loans: DataFrame with repayment information joined with loan data
        decision_time_days: Decision time t in days after cohort creation (default: 90)

    Returns:
        DataFrame with cohort-level features, one row per cohort
    """
    print("Creating cohort-level features...")

    # First create the loan-level features to use as input for cohort features
    loan_features_df = create_loan_level_features(
        loans_and_cohort=loans_and_cohort,
        repayments_and_loans=repayments_and_loans,
        decision_time_days=decision_time_days,
    )

    return create_cohort_features_from_loan_features(
        loan_features_df, decision_time_days
    )


def create_cohort_features_from_loan_features(
    loan_features_df: pd.DataFrame,
    decision_time_days: int = 90,
) -> pd.DataFrame:
    """
    Create cohort-level features from already processed loan-level features.

    Args:
        loan_features_df: DataFrame with loan-level features

    Returns:
        DataFrame with cohort-level features, one row per cohort
    """
    # Start with basic composition features
    cohort_features = create_loan_composition_features(loan_features_df)

    # Add distribution and concentration features
    distribution_features = create_loan_distribution_features(loan_features_df)
    cohort_features = cohort_features.merge(distribution_features, on="batch_letter")

    # Add temporal cohort features
    temporal_features = create_temporal_cohort_features(loan_features_df)
    cohort_features = cohort_features.merge(temporal_features, on="batch_letter")

    # Add repayment behavior features
    repayment_features = create_repayment_cohort_features(
        loan_features_df, decision_time_days
    )
    cohort_features = cohort_features.merge(repayment_features, on="batch_letter")

    # Add repayment performance features
    performance_features = create_repayment_performance_features(loan_features_df)
    cohort_features = cohort_features.merge(performance_features, on="batch_letter")

    # Add risk distribution features
    risk_features = create_risk_distribution_features(loan_features_df)
    cohort_features = cohort_features.merge(risk_features, on="batch_letter")

    # Add interaction features
    interaction_features = create_interaction_cohort_features(
        loan_features_df, decision_time_days=decision_time_days
    )
    cohort_features = cohort_features.merge(interaction_features, on="batch_letter")

    print(
        f"Final cohort features dataset: {len(cohort_features)} cohorts with {len(cohort_features.columns)} features"
    )

    return cohort_features


# Export the utility functions as well for backward compatibility
__all__ = [
    "create_loan_level_features",
    "create_cohort_level_features",
    "create_cohort_features_from_loan_features",
    "save_features_to_database",
    "load_features_from_database",
    "get_unique_loans_from_history",
]
