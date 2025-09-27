"""
Hybrid Model for ROI Prediction with Uncertainty Quantification

This module implements a hybrid modeling approach for predicting ROI over time.
The model combines:
1. Linear regression for trend modeling (with Box-Cox transformation)
2. LightGBM for residual modeling
3. Bootstrap sampling for uncertainty estimation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor


class HybridROIModel:
    """
    Hybrid model combining linear trend modeling with tree-based residual modeling.

    This model addresses the limitation of tree-based models in extrapolating
    beyond training data by decomposing the prediction into:
    1. Trend component (linear model on Box-Cox transformed target)
    2. Residual component (LightGBM on residuals)
    """

    def __init__(self, lgbm_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the hybrid model.

        Args:
            lgbm_params: Parameters for LightGBM regressor. If None, uses defaults.
        """
        self.lgbm_params = lgbm_params or {"random_state": 42}
        self.trend_model = None
        self.residual_model = None
        self.preprocessor = None
        self.boxcox_lambda = None
        self.is_fitted = False

    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline.

        Args:
            X: Feature matrix

        Returns:
            ColumnTransformer for preprocessing
        """
        numerical_columns_to_scale = [
            col for col in X.columns if col not in ["h_days", "batch_letter"]
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("h_days", "passthrough", ["h_days"]),
                ("batch_letter", "drop", ["batch_letter"]),
                ("scaled", StandardScaler(), numerical_columns_to_scale),
            ],
        )

        return preprocessor

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridROIModel":
        """
        Fit the hybrid model.

        Args:
            X: Feature matrix
            y: Target variable (ROI)

        Returns:
            self
        """
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X)

        # Apply Box-Cox transformation to target
        y_boxcox, self.boxcox_lambda = boxcox(y + 1.1)

        # Fit trend model (Linear Regression on Box-Cox transformed target)
        self.trend_model = make_pipeline(self.preprocessor, LinearRegression())
        self.trend_model.fit(X, y_boxcox)

        # Calculate residuals
        y_trend_pred = self.trend_model.predict(X)
        y_residuals = y_boxcox - y_trend_pred

        # Fit residual model (LightGBM on residuals)
        self.residual_model = make_pipeline(
            self.preprocessor, LGBMRegressor(**self.lgbm_params)
        )
        self.residual_model.fit(X, y_residuals)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the hybrid model.

        Args:
            X: Feature matrix

        Returns:
            Predictions in original scale
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Predict trend and residuals in Box-Cox space
        trend_pred = self.trend_model.predict(X)
        residual_pred = self.residual_model.predict(X)

        # Combine predictions in Box-Cox space
        y_pred_boxcox = trend_pred + residual_pred

        # Transform back to original scale
        y_pred = inv_boxcox(y_pred_boxcox, self.boxcox_lambda) - 1.1

        return y_pred

    def predict_with_uncertainty(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimation using bootstrap sampling.

        Args:
            X_train: Training feature matrix
            y_train: Training target variable
            X_test: Test feature matrix for predictions
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals (default 95%)
            random_state: Random state for reproducibility

        Returns:
            Dictionary containing:
            - 'mean': Mean prediction
            - 'std': Standard deviation of predictions
            - 'lower_ci': Lower confidence interval
            - 'upper_ci': Upper confidence interval
            - 'all_predictions': All bootstrap predictions (n_bootstrap x n_samples)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        np.random.seed(random_state)
        bootstrap_predictions = []

        # Box-Cox transform training target
        y_train_boxcox = boxcox(y_train + 1.1, lmbda=self.boxcox_lambda)

        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(len(X_train)), random_state=random_state + i)
            X_boot = X_train.iloc[indices]
            y_boot = y_train_boxcox[indices]

            # Fit trend model on bootstrap sample
            trend_boot = make_pipeline(self.preprocessor, LinearRegression())
            trend_boot.fit(X_boot, y_boot)

            # Calculate residuals for bootstrap sample
            y_residuals_boot = y_boot - trend_boot.predict(X_boot)

            # Fit residual model on bootstrap sample
            residual_boot = make_pipeline(
                self.preprocessor, LGBMRegressor(**self.lgbm_params)
            )
            residual_boot.fit(X_boot, y_residuals_boot)

            # Make predictions on test set
            trend_pred = trend_boot.predict(X_test)
            residual_pred = residual_boot.predict(X_test)
            y_pred_boxcox = trend_pred + residual_pred

            # Transform back to original scale
            y_pred = inv_boxcox(y_pred_boxcox, self.boxcox_lambda) - 1.1

            bootstrap_predictions.append(y_pred)

        bootstrap_predictions = np.array(bootstrap_predictions)

        # Calculate uncertainty metrics
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        results = {
            "mean": np.mean(bootstrap_predictions, axis=0),
            "std": np.std(bootstrap_predictions, axis=0),
            "lower_ci": np.percentile(bootstrap_predictions, lower_percentile, axis=0),
            "upper_ci": np.percentile(bootstrap_predictions, upper_percentile, axis=0),
            "all_predictions": bootstrap_predictions,
        }

        return results

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        bootstrap_results: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test feature matrix
            y_test: Test target variable
            bootstrap_results: Optional bootstrap results for coverage analysis

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)

        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }

        if bootstrap_results is not None:
            # Calculate confidence interval coverage
            coverage = np.mean(
                (y_test >= bootstrap_results["lower_ci"])
                & (y_test <= bootstrap_results["upper_ci"])
            )
            metrics["ci_coverage"] = coverage
            metrics["mean_prediction_std"] = np.mean(bootstrap_results["std"])

        return metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the residual model (LightGBM).

        Returns:
            Dictionary with feature names and importance scores, or None if not available
        """
        if not self.is_fitted or not hasattr(
            self.residual_model.named_steps["lgbmregressor"], "feature_importances_"
        ):
            return None

        # Get feature names after preprocessing
        feature_names = []
        for transformer_name, transformer, features in self.preprocessor.transformers_:
            if transformer_name == "h_days":
                feature_names.extend(["h_days"])
            elif transformer_name == "scaled":
                feature_names.extend(features)

        importance_scores = self.residual_model.named_steps[
            "lgbmregressor"
        ].feature_importances_

        return dict(zip(feature_names, importance_scores))


def train_hybrid_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lgbm_params: Optional[Dict[str, Any]] = None,
) -> HybridROIModel:
    """
    Convenience function to train a hybrid ROI model.

    Args:
        X_train: Training feature matrix
        y_train: Training target variable
        lgbm_params: Parameters for LightGBM regressor

    Returns:
        Fitted HybridROIModel
    """
    model = HybridROIModel(lgbm_params=lgbm_params)
    model.fit(X_train, y_train)
    return model


def make_predictions_with_uncertainty(
    model: HybridROIModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """
    Convenience function to make predictions with uncertainty estimation.

    Args:
        model: Fitted HybridROIModel
        X_train: Training feature matrix
        y_train: Training target variable
        X_test: Test feature matrix
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random state for reproducibility

    Returns:
        Tuple of (predictions, uncertainty_results, summary_stats)
    """
    # Make standard predictions
    predictions = model.predict(X_test)

    # Get uncertainty estimates
    uncertainty_results = model.predict_with_uncertainty(
        X_train, y_train, X_test, n_bootstrap, confidence_level, random_state
    )

    # Calculate summary statistics
    summary_stats = {
        "mean_prediction_std": np.mean(uncertainty_results["std"]),
        "max_prediction_std": np.max(uncertainty_results["std"]),
        "min_prediction_std": np.min(uncertainty_results["std"]),
        "mean_ci_width": np.mean(
            uncertainty_results["upper_ci"] - uncertainty_results["lower_ci"]
        ),
    }

    return predictions, uncertainty_results, summary_stats
