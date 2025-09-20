import pandas as pd
import numpy as np


def stratified_loan_selection(
    loans_df, target_default_rate, random_state=42, verbose=True
) -> pd.DataFrame:
    """
    Select loans for default simulation using stratified sampling to avoid bias from skewed distribution.

    Args:
        loans_df: DataFrame with loan data
        target_default_rate: Target default rate (as proportion of principal amount)
        n_strata: Number of loan size strata
        random_state: Random seed for reproducibility

    Returns:
        pd.DataFrame: Loans selected for default
    """
    np.random.seed(random_state)

    # Create loan size strata based on quantiles
    n_strata = 3
    strata_labels = ["Small", "Medium", "Large"]
    loans_df = loans_df.copy()

    # In this loop we try to create the strata, if it fails due to not enough unique values,
    # we reduce the number of strata
    while True:
        try:
            loans_df["size_stratum"] = pd.qcut(
                loans_df["loan_amount"],
                q=n_strata,
                labels=strata_labels,
                duplicates="drop",
            )
        except ValueError as e:
            # print(
            #     f"Error occurred while spliting cohort into {n_strata} strata, reducing number of strata"
            # )
            n_strata -= 1
            strata_labels.pop(-1)
            continue
        else:
            break

    # Calculate target default amount
    total_principal = loans_df["loan_amount"].sum()
    target_default_amount = total_principal * target_default_rate

    # Sample proportionally from each stratum
    selected_loans = []
    remaining_default_amount = target_default_amount

    for stratum in strata_labels:
        stratum_loans = loans_df[loans_df["size_stratum"] == stratum].copy()
        stratum_principal = stratum_loans["loan_amount"].sum()

        # Calculate proportional default amount for this stratum
        stratum_target_amount = (
            stratum_principal / total_principal
        ) * target_default_amount

        # Sort by loan amount and select loans until target is reached
        stratum_loans = stratum_loans.sort_values("loan_amount")
        stratum_selected = []
        stratum_cumulative = 0

        # Randomly shuffle within stratum to avoid always selecting smallest loans
        stratum_loans = stratum_loans.sample(
            frac=1, random_state=random_state + hash(stratum) % 1000
        )

        for _, loan in stratum_loans.iterrows():
            if stratum_cumulative + loan["loan_amount"] <= stratum_target_amount:
                stratum_selected.append(loan["loan_id"])
                stratum_cumulative += loan["loan_amount"]

            if stratum_cumulative >= stratum_target_amount * 0.95:  # Allow 5% tolerance
                break

        selected_loans.extend(stratum_selected)
        remaining_default_amount -= stratum_cumulative
        if verbose:
            print(f"Selected {len(stratum_selected)} loans from {stratum} stratum")

    return loans_df[loans_df["loan_id"].isin(selected_loans)]
