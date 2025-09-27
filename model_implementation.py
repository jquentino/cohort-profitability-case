from src.models import train_hybrid_model, make_predictions_with_uncertainty
from src.features.feature_utils import load_features_from_database
from src.config import DECISION_TIME_DAYS, TIME_HORIZON_DAYS
import numpy as np


def run_ROI_over_time_eda(
    database_path: str,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Run ROI over time EDA.
    Args:
        database_path: Path to the database file. Considers all the loans, loan repayments, and transactions corresponding to a same batch.
    Returns:
        Tuple containing:
            - The expected ROI at the horizon for each batch
            - Tuple containing the lower and upper bounds of the confidence interval.
    """

    # Load cohort features from database
    _, cohort_features = load_features_from_database(
        database_path,
        DECISION_TIME_DAYS,
        TIME_HORIZON_DAYS,
        cohort_features=True,
        loan_features=False,
    )

    # Prepare training data (only data up to decision time)
    train_mask = cohort_features["h_days"] <= DECISION_TIME_DAYS

    X = cohort_features.drop(["ROI"], axis=1)
    y = cohort_features["ROI"]

    X_train = X[train_mask]
    y_train = y[train_mask]

    # Create horizon prediction data (at TIME_HORIZON_DAYS)
    horizon_mask = cohort_features["h_days"] == TIME_HORIZON_DAYS
    X_horizon = X[horizon_mask]

    if len(X_horizon) == 0:
        # If no exact horizon data, create prediction data for the horizon
        # Use the latest available data structure and set h_days to horizon
        latest_data = X[X["h_days"] == X["h_days"].max()].copy()
        X_horizon = latest_data.copy()
        X_horizon["h_days"] = TIME_HORIZON_DAYS

    # Configure LightGBM parameters for production use
    lgbm_params = {
        "random_state": 42,
        "n_estimators": 150,
        "learning_rate": 0.1,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,  # Suppress training output
    }

    # Train the hybrid model
    model = train_hybrid_model(X_train, y_train, lgbm_params)

    # Make predictions with uncertainty quantification at the horizon
    _, uncertainty_results, _ = make_predictions_with_uncertainty(
        model,
        X_train,
        y_train,
        X_horizon,
        n_bootstrap=100,  # Use 100 bootstrap samples for robust estimates
        confidence_level=0.95,
        random_state=42,
    )

    # Extract results for the horizon
    expected_roi = uncertainty_results["mean"]  # Mean prediction
    lower_bound = uncertainty_results["lower_ci"]  # Lower 95% CI
    upper_bound = uncertainty_results["upper_ci"]  # Upper 95% CI

    return expected_roi, (lower_bound, upper_bound)
