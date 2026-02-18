"""
Core module for shared functionality across all trading models.

This module provides:
- feature_engineering: Technical indicators and feature creation
- data_pipeline: Data loading, splitting, and preprocessing (no data leakage)
- base_model: Abstract base class for all prediction models
- metrics: Standardized metrics calculation
"""

from core.feature_engineering import create_features, FeatureSet
from core.data_pipeline import DataPipeline, DataSplit
from core.metrics import calculate_metrics, calculate_direction_accuracy
from core.base_model import BaseTradeModel

__all__ = [
    'create_features',
    'FeatureSet',
    'DataPipeline',
    'DataSplit',
    'calculate_metrics',
    'calculate_direction_accuracy',
    'BaseTradeModel',
]
