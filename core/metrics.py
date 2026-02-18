"""
Centralized metrics calculation module.

This module consolidates all metrics calculations that were
previously duplicated across model files.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate standard regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Optional prefix for metric names (e.g., "train_", "test_")

    Returns:
        Dictionary with metrics: mse, rmse, mae, r2, mape
    """
    # Filter out any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            f"{prefix}mse": np.nan,
            f"{prefix}rmse": np.nan,
            f"{prefix}mae": np.nan,
            f"{prefix}r2": np.nan,
            f"{prefix}mape": np.nan,
        }

    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)

    # MAPE with protection against division by zero
    mask_nonzero = y_true_clean != 0
    if mask_nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true_clean[mask_nonzero] - y_pred_clean[mask_nonzero]) / y_true_clean[mask_nonzero])) * 100
    else:
        mape = np.nan

    return {
        f"{prefix}mse": mse,
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
        f"{prefix}mape": mape,
    }


def calculate_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_prices: Optional[np.ndarray] = None
) -> float:
    """
    Calculate direction accuracy (percentage of correctly predicted price movements).

    Args:
        y_true: True future prices
        y_pred: Predicted future prices
        current_prices: Current prices (if None, uses previous true value)

    Returns:
        Direction accuracy as percentage (0-100)
    """
    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) < 2:
        return 0.0

    if current_prices is not None:
        current_clean = current_prices[mask]
        # Direction relative to current price
        true_direction = y_true_clean > current_clean
        pred_direction = y_pred_clean > current_clean
    else:
        # Direction relative to previous value
        true_direction = np.diff(np.concatenate([[y_true_clean[0]], y_true_clean])) > 0
        pred_direction = np.diff(np.concatenate([[y_pred_clean[0]], y_pred_clean])) > 0

    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy


def calculate_trading_signal(
    current_price: float,
    predicted_price: float,
    buy_threshold: float = 0.5,
    sell_threshold: float = -0.5
) -> Tuple[str, float]:
    """
    Determine trading signal based on predicted price change.

    Args:
        current_price: Current price
        predicted_price: Predicted future price
        buy_threshold: Minimum % change for BUY signal
        sell_threshold: Maximum % change for SELL signal

    Returns:
        Tuple of (signal, expected_change_percent)
        Signal is one of: "BUY", "SELL", "HOLD", "NEUTRAL"
    """
    if current_price == 0:
        return "NEUTRAL", 0.0

    expected_change = ((predicted_price - current_price) / current_price) * 100

    if expected_change >= buy_threshold:
        signal = "BUY"
    elif expected_change <= sell_threshold:
        signal = "SELL"
    elif abs(expected_change) < 0.1:
        signal = "NEUTRAL"
    else:
        signal = "HOLD"

    return signal, expected_change


def format_metrics_text(
    metrics: Dict[str, float],
    current_price: float,
    predicted_price: float,
    direction_accuracy: float,
    signal: str,
    expected_change: float
) -> str:
    """
    Format metrics as human-readable text.

    This format is compatible with the existing regex parsing in main.py.

    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        current_price: Current price
        predicted_price: Predicted price
        direction_accuracy: Direction accuracy percentage
        signal: Trading signal
        expected_change: Expected change percentage

    Returns:
        Formatted text string
    """
    # Remove prefix from metrics keys for display
    display_metrics = {}
    for key, value in metrics.items():
        clean_key = key.replace('test_', '').replace('train_', '')
        display_metrics[clean_key] = value

    text = f"""
===== Метрики на тестовой выборке =====
MSE: {display_metrics.get('mse', 0):.4f}
RMSE: {display_metrics.get('rmse', 0):.4f}
MAE: {display_metrics.get('mae', 0):.4f}
R²: {display_metrics.get('r2', 0):.4f}
MAPE: {display_metrics.get('mape', 0):.2f}%
Direction Accuracy: {direction_accuracy:.2f}%

Текущая цена: {current_price:.2f}
Прогнозируемая цена: {predicted_price:.2f}
Ожидаемое изменение: {expected_change:+.2f}%
Торговый сигнал: {signal}
"""
    return text.strip()


def calculate_all_metrics(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    current_prices_test: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for train and test sets.

    Args:
        y_train_true: True training values
        y_train_pred: Predicted training values
        y_test_true: True test values
        y_test_pred: Predicted test values
        current_prices_test: Current prices for test set

    Returns:
        Dictionary with all metrics
    """
    # Calculate base metrics
    train_metrics = calculate_metrics(y_train_true, y_train_pred, prefix="train_")
    test_metrics = calculate_metrics(y_test_true, y_test_pred, prefix="test_")

    # Calculate direction accuracy
    train_direction = calculate_direction_accuracy(y_train_true, y_train_pred)
    test_direction = calculate_direction_accuracy(
        y_test_true, y_test_pred, current_prices_test
    )

    # Combine all metrics
    all_metrics = {**train_metrics, **test_metrics}
    all_metrics['train_direction_accuracy'] = train_direction
    all_metrics['test_direction_accuracy'] = test_direction

    return all_metrics
