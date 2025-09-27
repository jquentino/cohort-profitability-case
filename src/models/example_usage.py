"""
Example usage of the HybridROIModel for cohort profitability prediction.

This script demonstrates how to use the hybrid model with uncertainty quantification.
"""

import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Add src to path if running as script
sys.path.append("../")

from src.models.hybrid_roi_model import (
    train_hybrid_model,
    make_predictions_with_uncertainty,
)
from src.models.visualization import (
    plot_predictions_with_uncertainty,
    plot_uncertainty_analysis,
    plot_model_components,
)
from src.features.feature_utils import load_features_from_database
from src.config import DATABASE_PATH, DECISION_TIME_DAYS, TIME_HORIZON_DAYS


def run_hybrid_model_example():
    """
    Complete example of training and evaluating the hybrid ROI model.
    """
    print("Loading data...")

    # Load cohort features
    _, cohort_features = load_features_from_database(
        DATABASE_PATH,
        DECISION_TIME_DAYS,
        TIME_HORIZON_DAYS,
        cohort_features=True,
        loan_features=False,
    )

    print(f"Loaded {len(cohort_features)} samples")

    # Prepare train/test splits
    train_mask = cohort_features["h_days"] <= 0.8 * DECISION_TIME_DAYS
    test_mask = (cohort_features["h_days"] > DECISION_TIME_DAYS) & (
        cohort_features["h_days"] <= TIME_HORIZON_DAYS
    )

    X = cohort_features.drop(["ROI"], axis=1)
    y = cohort_features["ROI"]

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train the hybrid model
    print("\nTraining hybrid model...")

    lgbm_params = {
        "random_state": 42,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    }

    model = train_hybrid_model(X_train, y_train, lgbm_params)

    print("Model training completed!")

    # Make standard predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)

    # Calculate basic metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")

    # Make predictions with uncertainty
    print("\nCalculating uncertainty estimates (this may take a while)...")

    predictions_with_uncertainty, uncertainty_results, summary_stats = (
        make_predictions_with_uncertainty(
            model,
            X_train,
            y_train,
            X_test,
            n_bootstrap=50,  # Reduced for faster execution
            confidence_level=0.95,
            random_state=42,
        )
    )

    # Evaluate with uncertainty
    evaluation_metrics = model.evaluate(X_test, y_test, uncertainty_results)

    print("\nUncertainty Analysis Results:")
    print(f"Mean prediction std: {summary_stats['mean_prediction_std']:.4f}")
    print(f"95% CI coverage: {evaluation_metrics['ci_coverage']:.2%}")
    print(f"Mean CI width: {summary_stats['mean_ci_width']:.4f}")

    # Get feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance:
        print("\nTop 5 Most Important Features:")
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Plot predictions with uncertainty
    plot_predictions_with_uncertainty(
        X_test,
        y_test,
        predictions,
        uncertainty_results,
        DECISION_TIME_DAYS,
        title="Hybrid Model: Predictions with 95% Confidence Intervals",
    )

    # Plot uncertainty analysis
    plot_uncertainty_analysis(X_test, uncertainty_results, DECISION_TIME_DAYS)

    # Plot model components (using full dataset for better visualization)
    plot_model_components(X, y, model, DECISION_TIME_DAYS)

    print("Example completed! Check the generated plots.")

    return model, predictions, uncertainty_results, evaluation_metrics


def quick_prediction_example():
    """
    Quick example showing how to make predictions with a trained model.
    """
    print("Quick prediction example...")

    # This would typically be done with a saved/loaded model
    # For demonstration, we'll create a simple example

    # Create dummy data matching your feature structure
    dummy_data = pd.DataFrame(
        {
            "h_days": [30, 60, 90, 120, 150],
            "batch_letter": ["A", "A", "A", "A", "A"],
            "feature1": [1.0, 1.1, 1.2, 1.3, 1.4],
            "feature2": [0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )

    print("To use the model for predictions:")
    print("1. Load or train your HybridROIModel")
    print("2. Call model.predict(X) for point predictions")
    print("3. Use model.predict_with_uncertainty() for uncertainty estimates")

    print(f"\nExample prediction input shape: {dummy_data.shape}")
    print("Features:", list(dummy_data.columns))


if __name__ == "__main__":
    # Run the complete example
    try:
        model, predictions, uncertainty_results, metrics = run_hybrid_model_example()
        print("\n" + "=" * 50)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
    except Exception as e:
        print(f"Error running example: {e}")
        print("Running quick example instead...")
        quick_prediction_example()
