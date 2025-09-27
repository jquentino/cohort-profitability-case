"""
Models package for cohort profitability prediction.

This package provides:
- HybridROIModel: A hybrid model combining linear trend and tree-based residual modeling
- Bootstrap uncertainty quantification
- Visualization utilities for model analysis
"""

from .hybrid_roi_model import (
    HybridROIModel,
    train_hybrid_model,
    make_predictions_with_uncertainty,
)

__all__ = ["HybridROIModel", "train_hybrid_model", "make_predictions_with_uncertainty"]
