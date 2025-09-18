import sqlite3
import pandas as pd


def load_data(db_path: str = "../database.db"):
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
    loans["updated_at"] = pd.to_datetime(loans["updated_at"], format="ISO8601")
    repayments["date"] = pd.to_datetime(repayments["date"], format="ISO8601")

    # Join tables to create a comprehensive dataset
    loans_and_cohort = loans.merge(allowlist, on="user_id", how="left")
    repayments_and_loans = repayments.merge(loans_and_cohort, on="loan_id", how="left")

    # Close the database connection
    conn.close()

    return allowlist, loans, repayments, loans_and_cohort, repayments_and_loans


def prepare_roi_columns(repayments_and_loans: pd.DataFrame) -> pd.DataFrame:
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
    # Define the base date as the cohort creation date (t=0)
    repayments_and_loans["cohort_start"] = repayments_and_loans.groupby("batch")[
        "allowlisted_date"
    ].transform("min")

    # Relative time since the beginning of the cohort
    repayments_and_loans["h_days"] = (
        pd.to_datetime(repayments_and_loans["date"])
        - pd.to_datetime(repayments_and_loans["cohort_start"])
    ).dt.days

    # Cash flow received over time
    repayments_and_loans["repayment_total"] = repayments_and_loans[
        "repayment_amount"
    ].fillna(0) + repayments_and_loans["billings_amount"].fillna(0)

    return repayments_and_loans
