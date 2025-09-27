"""
Cohort-Level Feature Engineering Functions

Individual functions for creating cohort-level features for cohort profitability prediction.

Author: Generated for CloudWalk Case Study
"""

import pandas as pd
from .feature_utils import hhi_concentration, safe_divide


def create_loan_composition_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort loan composition features using vectorized operations."""

    # Group by batch letter and calculate all metrics at once
    cohort_features = (
        features_df.groupby("batch_letter")
        .agg(
            # Basic count and totals
            cohort_size=("loan_id", "count"),
            total_loan_amount=("loan_amount", "sum"),
            avg_loan_amount=("loan_amount", "mean"),
            median_loan_amount=("loan_amount", "median"),
            loan_amount_skewness=("loan_amount", "skew"),
            # Interest rate statistics
            avg_interest_rate=("annual_interest", "mean"),
            median_interest_rate=("annual_interest", "median"),
            std_interest_rate=("annual_interest", "std"),
            total_repaid_amount=("total_repaid_amount", "sum"),
        )
        .reset_index()
    )

    return cohort_features


def create_loan_distribution_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort loan distribution and concentration features using vectorized operations."""

    # Create a function to apply hhi to each group
    def calc_group_metrics(group):
        return pd.Series(
            {
                "loan_amount_hhi": hhi_concentration(group["loan_amount"]),
                "loan_amount_p25": group["loan_amount"].quantile(0.25),
                "loan_amount_p75": group["loan_amount"].quantile(0.75),
                "loan_amount_p90": group["loan_amount"].quantile(0.90),
                "loan_amount_cv": safe_divide(
                    group["loan_amount"].std(), group["loan_amount"].mean()
                ),
            }
        )

    # Apply the function to each group and reset the index
    cohort_features = (
        features_df.groupby("batch_letter").apply(calc_group_metrics).reset_index()
    )

    return cohort_features


def create_temporal_cohort_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features at cohort level using vectorized operations."""

    # Group by batch letter and calculate all metrics at once
    cohort_features = (
        features_df.groupby("batch_letter")
        .agg(
            # Average timing metrics
            avg_days_since_loan_issuance=("days_since_loan_issuance", "mean"),
            avg_days_allowlist_to_loan=("days_allowlist_to_loan", "mean"),
            # Timing distribution
            std_days_since_issuance=("days_since_loan_issuance", "std"),
            std_days_allowlist_to_loan=("days_allowlist_to_loan", "std"),
        )
        .reset_index()
    )

    return cohort_features


def create_repayment_cohort_features(
    features_df: pd.DataFrame, decision_time_days: int
) -> pd.DataFrame:
    """Create cohort-level repayment behavior features using vectorized operations."""

    # Create an empty list to store the calculated metrics for each period
    agg_dict = {}

    # Add other potential columns
    if "days_to_first_repayment" in features_df.columns:
        agg_dict["avg_days_to_first_repayment"] = pd.NamedAgg(
            column="days_to_first_repayment", aggfunc="mean"
        )
        agg_dict["median_days_to_first_repayment"] = pd.NamedAgg(
            column="days_to_first_repayment", aggfunc="median"
        )

    if "repayment_consistency_cv" in features_df.columns:
        agg_dict["avg_repayment_consistency"] = pd.NamedAgg(
            column="repayment_consistency_cv", aggfunc="mean"
        )

    # Group by batch_letter and calculate all metrics at once
    cohort_features = features_df.groupby("batch_letter").agg(**agg_dict).reset_index()

    return cohort_features


def create_risk_distribution_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort risk distribution features using vectorized operations."""

    result_dfs = []

    # Base dataframe with batch letters
    batch_letters = features_df[["batch_letter"]].drop_duplicates()
    result_dfs.append(batch_letters)

    # Status distribution at decision time
    if "status_at_decision_time" in features_df.columns:
        # Get counts of each status by batch
        status_counts = pd.crosstab(
            features_df["batch_letter"],
            features_df["status_at_decision_time"],
            normalize="index",
        ).reset_index()

        # Rename and select columns for specific statuses
        status_renamed = status_counts.rename(
            columns={
                "executed": "pct_executed",
                "debt_collection": "pct_debt_collection",
                "debt_repaid": "pct_debt_repaid",
                "repaid": "pct_repaid",
            }
        )

        # Only keep columns that exist after renaming
        cols_to_keep = ["batch_letter"] + [
            col
            for col in [
                "pct_executed",
                "pct_debt_collection",
                "pct_debt_repaid",
                "pct_repaid",
            ]
            if col in status_renamed.columns
        ]

        status_features = status_renamed[cols_to_keep]
        result_dfs.append(status_features)

    # Merge all result dataframes
    if len(result_dfs) > 1:
        cohort_features = result_dfs[0]
        for df in result_dfs[1:]:
            cohort_features = cohort_features.merge(df, on="batch_letter", how="left")
    else:
        cohort_features = result_dfs[0]

    return cohort_features


def create_repayment_performance_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort-level repayment performance features using vectorized operations."""
    result_dfs = []

    # Base dataframe with batch letters
    batch_letters = features_df[["batch_letter"]].drop_duplicates()
    result_dfs.append(batch_letters)

    # Calculate loan amount sums and counts by batch
    loan_totals = (
        features_df.groupby("batch_letter")
        .agg(
            total_cohort_amount=("loan_amount", "sum"),
            total_cohort_loans=("loan_id", "count"),
        )
        .reset_index()
    )
    result_dfs.append(loan_totals)

    # Status-based features
    if "status_at_decision_time" in features_df.columns:
        # Count loans in each status by batch
        status_counts = (
            features_df.groupby(["batch_letter", "status_at_decision_time"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        # Merge with loan totals
        status_features = status_counts.merge(
            loan_totals, on="batch_letter", how="left"
        )

        # Calculate percentage metrics
        # 2. Number of totally repaid loans / total loans in cohort
        repaid_cols = ["repaid", "debt_repaid"]
        repaid_cols_exist = [
            col for col in repaid_cols if col in status_features.columns
        ]

        if repaid_cols_exist:
            status_features["totally_repaid_count"] = status_features[
                repaid_cols_exist
            ].sum(axis=1)

            # Apply safe_divide row by row
            status_features["pct_loans_totally_repaid"] = status_features.apply(
                lambda row: safe_divide(
                    row["totally_repaid_count"], row["total_cohort_loans"]
                ),
                axis=1,
            )

        # 3. Number of loans in billing status / total loans in cohort
        if "debt_collection" in status_features.columns:
            status_features["pct_loans_in_billing"] = status_features.apply(
                lambda row: safe_divide(
                    row["debt_collection"], row["total_cohort_loans"]
                ),
                axis=1,
            )

        # 4. Number of loans in normal repayment status / total loans in cohort
        if "executed" in status_features.columns:
            status_features["pct_loans_normal_repayment"] = status_features.apply(
                lambda row: safe_divide(row["executed"], row["total_cohort_loans"]),
                axis=1,
            )

        # Select only the calculated metrics columns
        cols_to_keep = ["batch_letter"] + [
            col
            for col in [
                "pct_loans_totally_repaid",
                "pct_loans_in_billing",
                "pct_loans_normal_repayment",
            ]
            if col in status_features.columns
        ]

        result_dfs.append(status_features[cols_to_keep])

    # Merge all result dataframes
    if len(result_dfs) > 1:
        cohort_features = result_dfs[0]
        for df in result_dfs[1:]:
            cohort_features = cohort_features.merge(df, on="batch_letter", how="left")
    else:
        cohort_features = result_dfs[0]

    # Drop the temporary columns we used for calculations
    cols_to_drop = ["total_cohort_amount", "total_cohort_loans"]
    cohort_features = cohort_features.drop(
        [col for col in cols_to_drop if col in cohort_features.columns], axis=1
    )

    return cohort_features


def create_interaction_cohort_features(
    features_df: pd.DataFrame, decision_time_days: int
) -> pd.DataFrame:
    """Create cohort-level interaction features using vectorized operations."""
    result_dfs = []

    # Base dataframe with batch letters
    batch_letters = features_df[["batch_letter"]].drop_duplicates()
    result_dfs.append(batch_letters)

    # Average interaction terms
    if "loan_amount_x_interest" in features_df.columns:
        avg_interaction = (
            features_df.groupby("batch_letter")["loan_amount_x_interest"]
            .mean()
            .reset_index()
        )
        avg_interaction.columns = ["batch_letter", "avg_loan_amount_x_interest"]
        result_dfs.append(avg_interaction)

    # Merge all result dataframes
    if len(result_dfs) > 1:
        cohort_features = result_dfs[0]
        for df in result_dfs[1:]:
            cohort_features = cohort_features.merge(df, on="batch_letter", how="left")
    else:
        cohort_features = result_dfs[0]

    return cohort_features
