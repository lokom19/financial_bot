"""
Centralized feature engineering module.

This module consolidates all technical indicator calculations
that were previously duplicated across 13 model files.
"""

from enum import Enum
from typing import Set, Optional
import numpy as np
import pandas as pd


class FeatureSet(Enum):
    """Available feature sets for models."""
    BASIC = "basic"           # SMA, EMA, price changes (26 features)
    VOLUME = "volume"         # Volume indicators
    VOLATILITY = "volatility" # ATR, Bollinger Bands
    MOMENTUM = "momentum"     # RSI, MACD, Stochastic
    TREND = "trend"          # ADX, moving average crossovers
    EXTENDED = "extended"    # All of the above (~70 features)


def create_features(
    df: pd.DataFrame,
    feature_sets: Optional[Set[FeatureSet]] = None,
    include_target: bool = True
) -> pd.DataFrame:
    """
    Create technical indicators based on requested feature sets.

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume, timestamp)
        feature_sets: Set of feature categories to generate.
                     Default is {BASIC, VOLUME, VOLATILITY, MOMENTUM} for compatibility.
        include_target: Whether to create 'next_close' target column

    Returns:
        DataFrame with features and optionally 'next_close' target
    """
    if feature_sets is None:
        # Default feature set for backward compatibility
        feature_sets = {FeatureSet.BASIC, FeatureSet.VOLUME, FeatureSet.VOLATILITY, FeatureSet.MOMENTUM}

    df_features = df.copy()

    # Always add basic features
    if FeatureSet.BASIC in feature_sets or FeatureSet.EXTENDED in feature_sets:
        df_features = _add_basic_features(df_features)

    if FeatureSet.VOLUME in feature_sets or FeatureSet.EXTENDED in feature_sets:
        df_features = _add_volume_features(df_features)

    if FeatureSet.VOLATILITY in feature_sets or FeatureSet.EXTENDED in feature_sets:
        df_features = _add_volatility_features(df_features)

    if FeatureSet.MOMENTUM in feature_sets or FeatureSet.EXTENDED in feature_sets:
        df_features = _add_momentum_features(df_features)

    if FeatureSet.TREND in feature_sets or FeatureSet.EXTENDED in feature_sets:
        df_features = _add_trend_features(df_features)

    # Create target variable
    if include_target:
        df_features['next_close'] = df_features['close'].shift(-1)

    # Handle NaN and infinite values
    df_features = _handle_missing_values(df_features, exclude_cols=['next_close'] if include_target else [])

    return df_features


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical indicators: SMA, EMA, price changes."""
    # Simple Moving Averages
    for window in [5, 10, 20]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

    # Extended SMA for EXTENDED feature set
    for window in [50, 200]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

    # Exponential Moving Averages
    for span in [5, 10, 20]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    # Extended EMA
    for span in [50, 200]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    # Price relative to SMA
    for window in [5, 10, 20, 50, 200]:
        sma_col = f'sma_{window}'
        if sma_col in df.columns:
            df[f'close_minus_sma_{window}'] = df['close'] - df[sma_col]
            df[f'close_rel_sma_{window}'] = df['close'] / df[sma_col] - 1

    # Price changes
    for periods in [1, 2, 3, 5, 10, 20]:
        df[f'price_change_{periods}'] = df['close'].pct_change(periods=periods)

    # Cumulative returns
    price_change_1 = df['close'].pct_change(periods=1)
    for window in [5, 10, 20]:
        df[f'cum_return_{window}'] = (1 + price_change_1).rolling(window=window).apply(
            lambda x: np.prod(x) - 1, raw=True
        )

    # High-Low ratio
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based indicators."""
    # Log volume to reduce variance
    df['volume_log'] = np.log1p(df['volume'])

    # Volume moving averages
    for window in [5, 10, 20]:
        df[f'volume_sma_{window}'] = df['volume_log'].rolling(window=window).mean()

    # Volume ratio and change
    df['volume_ratio'] = df['volume_log'] / df['volume_sma_5']
    df['volume_change'] = df['volume_log'].pct_change(1)

    # Volume-price ratio
    df['volume_price_ratio'] = df['volume_log'] / np.log1p(df['close'])

    # Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']

    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)

    positive_mf_14 = pd.Series(positive_flow, index=df.index).rolling(window=14).sum()
    negative_mf_14 = pd.Series(negative_flow, index=df.index).rolling(window=14).sum()

    # Avoid division by zero
    mfi_ratio = np.where(negative_mf_14 != 0, positive_mf_14 / negative_mf_14, 100)
    df['mfi_14'] = 100 - (100 / (1 + mfi_ratio))

    return df


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators: ATR, Bollinger Bands."""
    # Volatility (normalized by SMA)
    for window in [5, 10, 20, 50]:
        sma_col = f'sma_{window}'
        if sma_col in df.columns:
            df[f'volatility_{window}'] = df['close'].rolling(window=window).std() / df[sma_col]

    # Volatility changes
    if 'volatility_5' in df.columns:
        df['volatility_change_5'] = df['volatility_5'].pct_change(periods=5)
    if 'volatility_10' in df.columns:
        df['volatility_change_10'] = df['volatility_10'].pct_change(periods=10)

    # True Range and ATR
    prev_close = df['close'].shift(1)
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hpc': abs(df['high'] - prev_close),
        'lpc': abs(df['low'] - prev_close)
    }).max(axis=1)

    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = bb_std.rolling(window=20).std()

    return df


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators: RSI, MACD, Stochastic."""
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    # Avoid division by zero
    rs = np.where(loss != 0, gain / loss, 100)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_change_5'] = pd.Series(df['rsi_14']).diff(periods=5)

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']
    df['macd_change'] = df['macd_histogram'].diff()

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    df['stoch_crossover'] = df['stoch_k'] - df['stoch_d']

    # Rate of Change (ROC)
    for periods in [5, 10, 20]:
        df[f'roc_{periods}'] = (df['close'] / df['close'].shift(periods) - 1) * 100

    return df


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend indicators: ADX, MA crossovers."""
    # Moving average crossovers
    if 'sma_5' in df.columns and 'sma_10' in df.columns:
        df['sma_5_10_cross'] = df['sma_5'] - df['sma_10']
    if 'sma_10' in df.columns and 'sma_20' in df.columns:
        df['sma_10_20_cross'] = df['sma_10'] - df['sma_20']
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        df['sma_20_50_cross'] = df['sma_20'] - df['sma_50']

    # EMA crossovers
    if 'ema_5' in df.columns and 'ema_10' in df.columns:
        df['ema_5_10_cross'] = df['ema_5'] - df['ema_10']
    if 'ema_10' in df.columns and 'ema_20' in df.columns:
        df['ema_10_20_cross'] = df['ema_10'] - df['ema_20']
    if 'ema_20' in df.columns and 'ema_50' in df.columns:
        df['ema_20_50_cross'] = df['ema_20'] - df['ema_50']

    # Average Directional Index (ADX)
    if 'atr_14' in df.columns:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].shift(1) - df['low']

        plus_dm_adj = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm_adj = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        plus_di_14 = 100 * (pd.Series(plus_dm_adj, index=df.index).ewm(alpha=1/14).mean() / df['atr_14'])
        minus_di_14 = 100 * (pd.Series(minus_dm_adj, index=df.index).ewm(alpha=1/14).mean() / df['atr_14'])

        df['dx'] = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
        df['adx'] = df['dx'].ewm(alpha=1/14).mean()
        df['adx_trend'] = df['adx'].diff()

    return df


def _handle_missing_values(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Handle NaN and infinite values properly.

    Uses modern pandas methods (ffill/bfill) instead of deprecated fillna(method=...).
    """
    if exclude_cols is None:
        exclude_cols = []

    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].isna().any():
            # First try forward fill, then backward fill
            df[col] = df[col].ffill().bfill()
            # If still NaN (e.g., all values were NaN), use median
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """
    Get list of feature columns (excluding metadata and target).

    Args:
        df: DataFrame with features
        exclude_cols: Additional columns to exclude

    Returns:
        List of feature column names
    """
    default_exclude = ['timestamp', 'next_close', 'open', 'high', 'low', 'close', 'volume', 'figi']
    if exclude_cols:
        default_exclude.extend(exclude_cols)

    return [col for col in df.columns if col not in default_exclude]
