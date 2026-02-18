"""
Base model class for all trading prediction models.

This module provides an abstract base class that ensures:
1. Consistent interface across all models
2. Proper data handling without leakage
3. Standardized metrics calculation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from core.feature_engineering import FeatureSet, create_features
from core.data_pipeline import DataPipeline, ScaledDataSplit
from core.metrics import (
    calculate_metrics,
    calculate_direction_accuracy,
    calculate_trading_signal,
    format_metrics_text
)


class BaseTradeModel(ABC):
    """
    Abstract base class for all trading prediction models.

    Subclasses must implement:
    - _create_model(): Create the underlying ML model
    - _fit_model(): Train the model
    - _predict(): Make predictions

    The base class handles:
    - Feature engineering
    - Data splitting (without leakage)
    - Scaling (fitted only on training data)
    - Metrics calculation
    - Trading signal generation
    """

    # Override in subclasses to specify required features
    REQUIRED_FEATURES: Set[FeatureSet] = {
        FeatureSet.BASIC,
        FeatureSet.VOLUME,
        FeatureSet.VOLATILITY,
        FeatureSet.MOMENTUM
    }

    # Model name for identification
    MODEL_NAME: str = "base"

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the base model.

        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state

        # Will be set during training
        self.scaler: Optional[RobustScaler] = None
        self.feature_columns: Optional[List[str]] = None
        self.last_price: Optional[float] = None
        self.is_fitted: bool = False

        # Metrics storage
        self.train_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}
        self.feature_importances: Optional[pd.DataFrame] = None

        # Data pipeline
        self.pipeline = DataPipeline(
            test_size=test_size,
            random_state=random_state
        )

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create and return the underlying model instance.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """
        Fit the model on training data.

        Args:
            X_train: Scaled training features
            y_train: Training targets
            X_val: Scaled validation features
            y_val: Validation targets
        """
        pass

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: Scaled features

        Returns:
            Predicted values
        """
        pass

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for the model.

        Args:
            df: Raw OHLCV data

        Returns:
            DataFrame with technical indicators
        """
        return create_features(df, feature_sets=self.REQUIRED_FEATURES)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Full training pipeline with proper data handling.

        This method:
        1. Creates features
        2. Splits data (BEFORE scaling)
        3. Scales data (fitted ONLY on training set)
        4. Trains the model
        5. Calculates metrics

        Args:
            df: Raw OHLCV data with timestamp column

        Returns:
            Dictionary with all metrics
        """
        # Step 1: Create features
        df_features = self.prepare_features(df)

        # Step 2 & 3: Split and scale (no data leakage)
        scaled_data = self.pipeline.prepare_and_scale(df_features, shuffle=False)

        # Store scaler and feature columns for later use
        self.scaler = scaled_data.scaler
        self.feature_columns = scaled_data.feature_columns

        # Store last known price for predictions
        self.last_price = float(scaled_data.current_prices_test.iloc[-1])

        # Step 4: Train the model
        self._fit_model(
            scaled_data.X_train_scaled,
            scaled_data.y_train,
            scaled_data.X_test_scaled,
            scaled_data.y_test
        )

        self.is_fitted = True

        # Step 5: Calculate metrics
        train_preds = self._predict(scaled_data.X_train_scaled)
        test_preds = self._predict(scaled_data.X_test_scaled)

        self.train_metrics = calculate_metrics(
            scaled_data.y_train, train_preds, prefix="train_"
        )
        self.test_metrics = calculate_metrics(
            scaled_data.y_test, test_preds, prefix="test_"
        )

        # Direction accuracy
        train_direction = calculate_direction_accuracy(
            scaled_data.y_train, train_preds
        )
        test_direction = calculate_direction_accuracy(
            scaled_data.y_test, test_preds,
            scaled_data.current_prices_test.values
        )

        self.train_metrics['train_direction_accuracy'] = train_direction
        self.test_metrics['test_direction_accuracy'] = test_direction

        # Combine and return all metrics
        all_metrics = {**self.train_metrics, **self.test_metrics}
        return all_metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: Raw OHLCV data

        Returns:
            Predicted prices
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Create features
        df_features = self.prepare_features(df)

        # Get feature columns in correct order
        X = df_features[self.feature_columns]

        # Scale using fitted scaler
        X_scaled = self.scaler.transform(X)

        return self._predict(X_scaled)

    def predict_next(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict the next price and generate trading signal.

        Args:
            df: Raw OHLCV data (should include the most recent data point)

        Returns:
            Dictionary with prediction details
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Create features
        df_features = self.prepare_features(df)

        # Use the last available row
        last_row = df_features[self.feature_columns].iloc[[-1]]

        # Scale and predict
        X_scaled = self.scaler.transform(last_row)
        predicted_price = float(self._predict(X_scaled)[0])

        # Get current price
        current_price = float(df['close'].iloc[-1])

        # Generate trading signal
        signal, expected_change = calculate_trading_signal(
            current_price, predicted_price
        )

        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'expected_change': expected_change,
            'signal': signal,
            'model_name': self.MODEL_NAME
        }

    def get_results_text(self, df: pd.DataFrame) -> str:
        """
        Get formatted results text compatible with existing parsing.

        Args:
            df: Raw OHLCV data used for training

        Returns:
            Formatted text with metrics and prediction
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        # Get prediction for the most recent data
        prediction = self.predict_next(df)

        return format_metrics_text(
            metrics=self.test_metrics,
            current_price=prediction['current_price'],
            predicted_price=prediction['predicted_price'],
            direction_accuracy=self.test_metrics.get('test_direction_accuracy', 0),
            signal=prediction['signal'],
            expected_change=prediction['expected_change']
        )

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importances if available.

        Returns:
            DataFrame with features and their importances, or None
        """
        return self.feature_importances


class SklearnTradeModel(BaseTradeModel):
    """
    Base class for sklearn-compatible models.

    Simplifies implementation for models that follow
    the standard sklearn fit/predict interface.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        super().__init__(test_size, random_state)
        self.model = None

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """Fit sklearn-compatible model."""
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Extract feature importances if available
        self._extract_feature_importances()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using sklearn-compatible model."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def _extract_feature_importances(self) -> None:
        """Extract feature importances from the model."""
        if self.model is None or self.feature_columns is None:
            return

        # Try different attribute names used by sklearn models
        importances = None
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)

        if importances is not None:
            self.feature_importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
