import pandas as pd


def compute_expected_loss(
    loans_and_cohort: pd.DataFrame,
    repayments_and_loans: pd.DataFrame,
    batch_letter: str,
    time_cut: int,
) -> tuple[float, float, float, float]:
    """
    Compute the expected loss for a given cohort at a specific time horizon.

    Args:
        loans_and_cohort: DataFrame containing loan and cohort information.
        repayments_and_loans: DataFrame containing repayment and loan information.
        batch_letter: The cohort batch letter to analyze.
        time_cut: The time horizon in days to consider.

    Returns:
        A tuple containing the probability of default, loss given default, and the expected loss.
    """

    loans_status_at_t = (
        loans_and_cohort[
            (loans_and_cohort["batch_letter"] == batch_letter)
            & (loans_and_cohort["updated_at_h_days"] <= time_cut)
        ]
        .sort_values("updated_at_h_days")
        .drop_duplicates(subset=["loan_id"], keep="last")
    )

    unpaid_loans_at_t = loans_status_at_t[
        loans_status_at_t["status"] == "debt_collection"
    ]
    if unpaid_loans_at_t.empty:
        return 0.0, 0.0, 0.0, 0.0

    probability_of_default = len(unpaid_loans_at_t) / len(loans_status_at_t)

    unpaid_loans_principal = unpaid_loans_at_t["loan_amount"].sum()

    repayments_and_loans_at_t = repayments_and_loans[
        (repayments_and_loans["batch_letter"] == batch_letter)
        & (repayments_and_loans["h_days"] <= time_cut)
    ]

    repayments_of_default_loans_at_t = repayments_and_loans_at_t[
        repayments_and_loans_at_t.loan_id.isin(unpaid_loans_at_t.loan_id)
    ][["repayment_total"]].sum()

    loss_given_default = (
        unpaid_loans_principal - repayments_of_default_loans_at_t["repayment_total"]
    ) / unpaid_loans_principal

    amount_in_risk = (
        loans_status_at_t["loan_amount"].sum()
        - repayments_and_loans_at_t["repayment_total"].sum()
    )
    if amount_in_risk < 0:
        amount_in_risk = 0.0

    expected_loss = probability_of_default * loss_given_default * amount_in_risk

    return probability_of_default, loss_given_default, amount_in_risk, expected_loss
