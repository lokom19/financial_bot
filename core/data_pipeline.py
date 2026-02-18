"""
Data pipeline module for proper train/test splitting.

This module ensures NO DATA LEAKAGE by:
1. Splitting data BEFORE any transformations
2. Fitting scaler ONLY on training data
3. Proper time series handling (no shuffle by default)
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


@dataclass
class DataSplit:
    """Container for train/test split data."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    timestamps_train: pd.Series
    timestamps_test: pd.Series
    current_prices_train: pd.Series
    current_prices_test: pd.Series

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)


@dataclass
class ScaledDataSplit:
    """Container for scaled train/test data."""
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    timestamps_train: pd.Series
    timestamps_test: pd.Series
    current_prices_train: pd.Series
    current_prices_test: pd.Series
    scaler: RobustScaler
    feature_columns: List[str]


class DataPipeline:
    """
    Pipeline for data preparation that ensures no data leakage.

    The critical principle: split happens BEFORE any transformations
    that could leak information from test to train set.

    Usage:
        pipeline = DataPipeline(test_size=0.2)
        split = pipeline.prepare_data(df_features)
        scaled_split = pipeline.scale_data(split)
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        target_col: str = 'next_close',
        timestamp_col: str = 'timestamp'
    ):
        """
        Initialize the data pipeline.

        Args:
            test_size: Fraction of data to use for testing (0.0 to 1.0)
            random_state: Random seed for reproducibility
            target_col: Name of the target column
            timestamp_col: Name of the timestamp column
        """
        self.test_size = test_size
        self.random_state = random_state
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.scaler: Optional[RobustScaler] = None
        self.feature_columns: Optional[List[str]] = None

    def prepare_data(
        self,
        df_features: pd.DataFrame,
        shuffle: bool = False
    ) -> DataSplit:
        """
        Prepare data for model training with proper train/test split.

        CRITICAL: Split happens FIRST, before any scaling or transformation.

        Args:
            df_features: DataFrame with features and target column
            shuffle: Whether to shuffle data (False for time series)

        Returns:
            DataSplit object containing train and test data
        """
        # Remove rows where target is NaN (typically the last row)
        valid_mask = ~df_features[self.target_col].isna()
        df_valid = df_features[valid_mask].copy()

        if len(df_valid) == 0:
            raise ValueError("No valid data after removing NaN targets")

        # Extract timestamps and current prices
        timestamps = df_valid[self.timestamp_col] if self.timestamp_col in df_valid.columns else pd.Series(range(len(df_valid)))
        y = df_valid[self.target_col]
        current_prices = df_valid['close']

        # Determine feature columns (everything except metadata and target)
        exclude_cols = [self.timestamp_col, self.target_col, 'volume', 'figi']
        if 'volume_log' in df_valid.columns:
            # If we have volume_log, exclude raw volume
            exclude_cols.append('volume')

        self.feature_columns = [
            col for col in df_valid.columns
            if col not in exclude_cols and not col.startswith('_')
        ]

        X = df_valid[self.feature_columns]

        # SPLIT FIRST - this is critical for preventing data leakage
        (X_train, X_test,
         y_train, y_test,
         ts_train, ts_test,
         prices_train, prices_test) = train_test_split(
            X, y, timestamps, current_prices,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=shuffle
        )

        return DataSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            timestamps_train=ts_train,
            timestamps_test=ts_test,
            current_prices_train=prices_train,
            current_prices_test=prices_test
        )

    def scale_data(self, split: DataSplit) -> ScaledDataSplit:
        """
        Scale the data using RobustScaler.

        CRITICAL: Scaler is fitted ONLY on training data.

        Args:
            split: DataSplit object from prepare_data()

        Returns:
            ScaledDataSplit object with scaled features
        """
        # Initialize scaler
        self.scaler = RobustScaler()

        # FIT ONLY ON TRAINING DATA
        X_train_scaled = self.scaler.fit_transform(split.X_train)

        # TRANSFORM (not fit_transform) on test data
        X_test_scaled = self.scaler.transform(split.X_test)

        return ScaledDataSplit(
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=split.y_train.values,
            y_test=split.y_test.values,
            timestamps_train=split.timestamps_train,
            timestamps_test=split.timestamps_test,
            current_prices_train=split.current_prices_train,
            current_prices_test=split.current_prices_test,
            scaler=self.scaler,
            feature_columns=self.feature_columns
        )

    def prepare_and_scale(
        self,
        df_features: pd.DataFrame,
        shuffle: bool = False
    ) -> ScaledDataSplit:
        """
        Convenience method to prepare and scale data in one call.

        Args:
            df_features: DataFrame with features and target
            shuffle: Whether to shuffle data

        Returns:
            ScaledDataSplit object ready for model training
        """
        split = self.prepare_data(df_features, shuffle=shuffle)
        return self.scale_data(split)

    def transform_new_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted scaler.

        Use this for making predictions on new data.

        Args:
            X: New feature data

        Returns:
            Scaled features as numpy array
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call scale_data() first.")
        if self.feature_columns is None:
            raise RuntimeError("Feature columns not set. Call prepare_data() first.")

        # Ensure same columns in same order
        X_aligned = X[self.feature_columns]
        return self.scaler.transform(X_aligned)


class TimeSeriesSplit:
    """
    Time series cross-validation with expanding window.

    Unlike sklearn's TimeSeriesSplit, this provides proper
    train/validation/test splits for financial data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.1,
        gap: int = 0
    ):
        """
        Initialize time series splitter.

        Args:
            n_splits: Number of splits for cross-validation
            test_size: Fraction of data for each test fold
            gap: Number of samples to skip between train and test
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate train/test indices for time series cross-validation.

        Uses expanding window: each fold has more training data.

        Args:
            X: Feature data

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        test_samples = int(n_samples * self.test_size)

        for i in range(self.n_splits):
            # Test set is at the end, moves backward
            test_end = n_samples - (i * test_samples)
            test_start = test_end - test_samples

            if test_start <= 0:
                break

            train_end = test_start - self.gap
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices
