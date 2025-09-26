import sqlite3
import pandas as pd


def load_data(db_path: str = "../database.db", remove_loans_with_errors: bool = False):
    """
    Load and prepare data from the SQLite database.

    Args:
        db_path (str): Path to the SQLite database file

    Returns:
        tuple: A tuple containing (allowlist, loans, repayments, repayments_and_loans)
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Load tables into pandas DataFrames
    allowlist = pd.read_sql_query("SELECT * FROM allowlist", conn)
    loans = pd.read_sql_query("SELECT * FROM loans", conn)
    repayments = pd.read_sql_query("SELECT * FROM loans_repayments", conn)

    allowlist["allowlisted_date"] = pd.to_datetime(
        allowlist["allowlisted_date"], format="ISO8601"
    )
    # Fixing datetime formats
    loans["created_at"] = pd.to_datetime(loans["created_at"], format="ISO8601")
    loans["updated_at"] = pd.to_datetime(
        loans["updated_at"], format="ISO8601", utc=False
    ).dt.tz_localize(None)
    repayments["date"] = pd.to_datetime(repayments["date"], format="ISO8601")

    # Map each batch to a letter (A, B, C, ...) to be more readable in plots
    batch_to_letter = {
        batch: chr(65 + i)
        for i, batch in enumerate(sorted(allowlist["batch"].unique()))
    }
    allowlist["batch_letter"] = allowlist["batch"].map(batch_to_letter)

    if remove_loans_with_errors:
        loans_ids_to_exclude = loans[
            loans.status.isin(
                [
                    "technical_loss",
                    "manual_cancellation",
                    "manual_cancelled",
                    "cancelled",
                ]
            )
        ].loan_id.unique()
        loans = loans[~loans.loan_id.isin(loans_ids_to_exclude)].copy()

    # Join tables to create a comprehensive dataset
    loans_and_cohort = loans.merge(allowlist, on="user_id", how="left")
    loans_and_cohort["cohort_start"] = loans_and_cohort.groupby("batch_letter")[
        "allowlisted_date"
    ].transform("min")

    loans_and_cohort["created_at_h_days"] = (
        loans_and_cohort["created_at"] - loans_and_cohort["cohort_start"]
    ).dt.days

    loans_and_cohort["updated_at_h_days"] = (
        loans_and_cohort["updated_at"] - loans_and_cohort["cohort_start"]
    ).dt.days

    # The loans_and_cohort shows the historical status of each loan. To join with repayments,
    # we just need information about the loan_id, batch and allowlisted_date, so we can drop other
    # columns and duplicated rows.

    repayments_and_loans = repayments.merge(
        loans_and_cohort[
            [
                "loan_id",
                "batch_letter",
                "allowlisted_date",
                "loan_amount",
                "cohort_start",
                "created_at",
                "created_at_h_days",
            ]
        ].drop_duplicates(subset=["loan_id"]),
        on="loan_id",
        how="left",
    )

    # Prepare ROI columns
    repayments_and_loans = _prepare_roi_columns(repayments_and_loans)

    # Close the database connection
    conn.close()

    return allowlist, loans, repayments, loans_and_cohort, repayments_and_loans


def _prepare_roi_columns(repayments_and_loans: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare additional columns for ROI calculations.

    Adds the following columns to the DataFrame:
        - cohort_start: The minimum allowlisted_date per batch (cohort creation date)
        - h_days: Relative days since the cohort_start for each repayment
        - repayment_total: Sum of repayment_amount and billings_amount for each row

    Args:
        repayments_and_loans (pd.DataFrame): The merged repayments and loans DataFrame

    Returns:
        pd.DataFrame: DataFrame with additional columns for ROI calculations
    """
    # Relative time since the beginning of the cohort
    repayments_and_loans["h_days"] = (
        repayments_and_loans["date"] - repayments_and_loans["cohort_start"]
    ).dt.days

    # Cash flow received over time
    repayments_and_loans["repayment_total"] = repayments_and_loans[
        "repayment_amount"
    ].fillna(0) + repayments_and_loans["billings_amount"].fillna(0)

    return repayments_and_loans


def compute_roi_curves(
    repayments_and_loans: pd.DataFrame, loans_and_cohort: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute ROI curves for each cohort over time.

    Args:
        repayments_and_loans (pd.DataFrame): DataFrame containing repayments and loans data with necessary columns.

    Returns:
        pd.DataFrame: DataFrame containing ROI curves with columns 'batch', 'h_days', and 'ROI'.
    """
    repayment_curve = _compute_repayments_curve(repayments_and_loans)
    max_days = repayment_curve["h_days"].max()
    cohort_principal = _compute_cohort_principal(loans_and_cohort, max_days)

    # Combine the repayment curve and cohort principal into a single DataFrame
    roi_curves = repayment_curve.merge(
        cohort_principal,
        left_on=["batch_letter", "h_days"],
        right_on=["batch_letter", "created_at_h_days"],
        how="left",
    )
    roi_curves["ROI"] = (
        roi_curves["repayment_total"] / roi_curves["cohort_principal"] - 1
    )

    return roi_curves


def _compute_repayments_curve(repayments_and_loans: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the cumulative repayments curve for each cohort over time.

    Args:
        repayments_and_loans (pd.DataFrame): DataFrame containing repayments and loans data with necessary columns.

    Returns:
        pd.DataFrame: DataFrame containing cumulative repayments with columns 'batch', 'h_days', and 'repayment_total'.
    """
    # Aggregate cash flows by batch and h_days
    # Vectorized approach - no loops needed
    repayment_curve = (
        repayments_and_loans.groupby(["batch_letter", "h_days"])["repayment_total"]
        .sum()
        .groupby(level=0)
        .cumsum()
        .reset_index()
        .astype({"h_days": int})
    )

    # Create a complete time series using vectorized operations
    max_h_days = repayment_curve["h_days"].max()
    batch_letters = repayment_curve["batch_letter"].unique().tolist()

    # Create MultiIndex with all combinations of batch_letter and h_days
    complete_index = pd.MultiIndex.from_product(
        [batch_letters, range(0, max_h_days + 1)], names=["batch_letter", "h_days"]
    ).to_frame(index=False)

    # Merge with actual data and forward fill
    repayment_curve = (
        complete_index.merge(repayment_curve, on=["batch_letter", "h_days"], how="left")
        .sort_values(["batch_letter", "h_days"])
        .groupby("batch_letter")["repayment_total"]
        .ffill()
        .fillna(0)
        .reset_index()
        .merge(complete_index, left_index=True, right_index=True)[
            ["batch_letter", "h_days", "repayment_total"]
        ]
    )

    return repayment_curve


def _compute_cohort_principal(
    loans_and_cohort: pd.DataFrame, max_h_days: int | None = None
) -> pd.DataFrame:
    """
    Compute the cohort principal for each batch.

    Args:
        loans_and_cohort (pd.DataFrame): The merged repayments and loans DataFrame
        max_days (int): Maximum number of days to consider for the cohort principal
    Returns:
        pd.Series: Series with cohort principal for each batch
    """
    cohort_principal = (
        loans_and_cohort[
            ["loan_id", "batch_letter", "loan_amount", "created_at_h_days"]
        ]
        .sort_values(by=["created_at_h_days"])
        .drop_duplicates(subset=["loan_id"])
        .groupby(["batch_letter", "created_at_h_days"])["loan_amount"]
        .sum()
        .groupby(level=0)
        .cumsum()
        .reset_index()
        .rename(columns={"loan_amount": "cohort_principal"})
    )
    if max_h_days is None:
        max_h_days = cohort_principal["created_at_h_days"].max()

    batch_letters = cohort_principal["batch_letter"].unique().tolist()

    complete_index = pd.MultiIndex.from_product(
        [batch_letters, range(0, max_h_days + 1)],
        names=["batch_letter", "created_at_h_days"],
    ).to_frame(index=False)

    cohort_principal = (
        complete_index.merge(
            cohort_principal, on=["batch_letter", "created_at_h_days"], how="left"
        )
        .sort_values(["batch_letter", "created_at_h_days"])
        .groupby("batch_letter")["cohort_principal"]
        .ffill()
        .fillna(0)
        .reset_index()
        .merge(complete_index, left_index=True, right_index=True)[
            ["batch_letter", "created_at_h_days", "cohort_principal"]
        ]
    )

    return cohort_principal
