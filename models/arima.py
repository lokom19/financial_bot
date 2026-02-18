import asyncio
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from utils.load_data_method import load_data

warnings.filterwarnings('ignore')


# Configuration
DEFAULT_TEST_SIZE = 60  # Number of days for testing
MAX_P = 3  # Maximum AR order
MAX_Q = 3  # Maximum MA order
MAX_D = 2  # Maximum differencing order

# Preprocessing functions
def preprocess_stock_data(df):
    """
    Preprocess stock data for time series analysis

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw stock data with timestamp and OHLCV columns

    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe with timestamp as index
    """
    df = df.copy()

    # Check and set the index
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)

    # Ensure index is datetime type
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort by date
    df = df.sort_index()

    # Handle missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found missing values: {missing_values[missing_values > 0]}")
        # Forward fill for financial time series (modern pandas syntax)
        df = df.ffill()
        # If there are still missing values at the beginning, backward fill
        df = df.bfill()

    # Check for duplicates in index
    if df.index.duplicated().any():
        print(f"Found {df.index.duplicated().sum()} duplicate timestamps. Keeping last values.")
        df = df[~df.index.duplicated(keep='last')]

    # Ensure the data has a regular frequency
    if df.index.inferred_freq is None:
        print("Irregular time series detected. Resampling to daily frequency.")
        # For financial data, business day frequency might be more appropriate
        df = df.asfreq('B', method='ffill')

    return df

# Stationarity analysis functions
def check_stationarity(series):
    """
    Test time series stationarity using Augmented Dickey-Fuller test

    Parameters:
    -----------
    series : pandas.Series
        Time series to test

    Returns:
    --------
    tuple
        (is_stationary, adf_result)
    """
    # Handle NaN values
    clean_series = series.dropna()

    if len(clean_series) < 10:
        print("Warning: Too few observations for reliable stationarity test")
        return False, None

    try:
        result = adfuller(clean_series)
        is_stationary = result[1] <= 0.05  # p-value threshold
        return is_stationary, result
    except Exception as e:
        print(f"Error in stationarity test: {e}")
        return False, None

def find_optimal_differencing(series, max_d=MAX_D):
    """
    Find the minimal differencing needed for stationarity

    Parameters:
    -----------
    series : pandas.Series
        Original time series
    max_d : int
        Maximum differencing order to try

    Returns:
    --------
    tuple
        (optimal_d, differenced_series)
    """
    original_series = series.copy()

    # Try each differencing order
    for d in range(max_d + 1):
        if d == 0:
            diff_series = original_series
        else:
            diff_series = original_series.diff(d).dropna()

        is_stationary, results = check_stationarity(diff_series)

        if is_stationary:
            return d, diff_series

    # If we couldn't achieve stationarity, return max_d
    print(f"Warning: Could not achieve stationarity with d≤{max_d}. Using d={max_d}.")
    return max_d, original_series.diff(max_d).dropna()

# ARIMA model selection functions
def find_optimal_arima_params(series, d, max_p=MAX_P, max_q=MAX_Q):
    """
    Find optimal ARIMA parameters using AIC criterion

    Parameters:
    -----------
    series : pandas.Series
        Time series to model
    d : int
        Differencing order
    max_p, max_q : int
        Maximum AR and MA orders to try

    Returns:
    --------
    tuple
        ((p,d,q), aic)
    """
    best_aic = float('inf')
    best_params = None

    # First try simple models (more efficient)
    simple_models = [(0,d,0), (1,d,0), (0,d,1), (1,d,1)]

    for order in simple_models:
        try:
            model = ARIMA(series, order=order)
            results = model.fit()
            aic = results.aic

            if aic < best_aic:
                best_aic = aic
                best_params = order
        except Exception as e:
            continue

    # If simple models work well, no need to try more complex ones
    if best_params is not None and best_params != (1,d,1):
        return best_params, best_aic

    # Try more complex models
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            # Skip already tried models
            if (p,d,q) in simple_models:
                continue

            try:
                model = ARIMA(series, order=(p,d,q))
                results = model.fit()
                aic = results.aic

                if aic < best_aic:
                    best_aic = aic
                    best_params = (p,d,q)
            except Exception as e:
                continue

    # If no model converged, use a simple default
    if best_params is None:
        print("Warning: No ARIMA models converged. Using default (1,d,1).")
        best_params = (1,d,1)

    return best_params, best_aic

# Analysis functions
def analyze_trend(series, window=30):
    """
    Analyze recent trend in time series

    Parameters:
    -----------
    series : pandas.Series
        Time series data
    window : int
        Window size for rolling statistics

    Returns:
    --------
    dict
        Trend analysis results
    """
    # Use last available data
    recent_data = series.dropna()[-window:]

    if len(recent_data) < window/2:
        return {
            'trend': "Insufficient data",
            'slope': None,
            'volatility': None,
            'range': None,
            'range_percent': None
        }

    # Calculate rolling mean
    rolling_mean = series.rolling(window=min(window, len(series)//2)).mean()

    # Calculate recent slope
    recent_mean = rolling_mean.dropna()
    if len(recent_mean) > 5:
        # Simple linear regression slope
        x = np.arange(len(recent_mean[-5:]))
        y = recent_mean[-5:].values
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0

    # Calculate recent volatility
    recent_std = recent_data.std()
    recent_range = recent_data.max() - recent_data.min()

    # Determine trend direction
    avg_price = recent_data.mean()
    if abs(slope) < 0.0001 * avg_price:
        trend = "Sideways"
    elif slope > 0.01 * avg_price:
        trend = "Strong Uptrend"
    elif slope > 0:
        trend = "Weak Uptrend"
    elif slope < -0.01 * avg_price:
        trend = "Strong Downtrend"
    else:
        trend = "Weak Downtrend"

    return {
        'trend': trend,
        'slope': slope,
        'volatility': recent_std,
        'range': recent_range,
        'range_percent': (recent_range / avg_price) * 100
    }

def calculate_prediction_metrics(actual, predicted):
    """
    Calculate common prediction error metrics

    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values

    Returns:
    --------
    dict
        Error metrics
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must be the same length")

    # Basic error metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE can cause division by zero errors
    try:
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    except (ZeroDivisionError, FloatingPointError):
        mape = None

    # Directional accuracy
    actual_diff = np.diff(actual)
    pred_diff = np.diff(predicted)
    directional_accuracy = np.mean((actual_diff > 0) == (pred_diff > 0)) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }

def determine_confidence(metrics):
    """
    Determine prediction confidence based on error metrics

    Parameters:
    -----------
    metrics : dict
        Error metrics

    Returns:
    --------
    str
        Confidence level
    """
    if metrics['mape'] is None:
        return "Unknown"

    mape = metrics['mape']
    directional_accuracy = metrics.get('directional_accuracy', 50)

    if mape < 1.5 and directional_accuracy > 65:
        confidence = "Very High"
    elif mape < 3 and directional_accuracy > 60:
        confidence = "High"
    elif mape < 7 and directional_accuracy > 55:
        confidence = "Moderate"
    elif mape < 15:
        confidence = "Low"
    else:
        confidence = "Very Low"

    return confidence

# Visualization functions
def plot_forecast_chart(historical, forecast_date, forecast_value, recent_days=60):
    """
    Plot recent price history with forecast

    Parameters:
    -----------
    historical : pandas.Series
        Historical time series
    forecast_date : pandas.Timestamp
        Date of the forecast
    forecast_value : float
        Forecasted value
    recent_days : int
        Number of recent days to display

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Get recent data
    recent = historical[-min(recent_days, len(historical)):]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot historical data
    ax.plot(recent.index, recent, 'b-', label='Historical', linewidth=2)

    # Add trend line
    x = np.arange(len(recent))
    y = recent.values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(recent.index, p(x), "g--", label="Trend", linewidth=1, alpha=0.7)

    # Calculate direction and change
    last_price = historical.iloc[-1]
    price_change = forecast_value - last_price
    price_change_percent = (price_change / last_price) * 100

    direction = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"

    # Plot forecast point and line
    ax.plot([historical.index[-1], forecast_date],
            [last_price, forecast_value],
            'r--', linewidth=2, label=f'Forecast: {direction}')
    ax.plot([forecast_date], [forecast_value], 'ro', markersize=8)

    # Add annotation
    ax.annotate(f'{forecast_value:.2f} ({price_change_percent:+.2f}%)',
                xy=(forecast_date, forecast_value),
                xytext=(forecast_date, forecast_value * 1.05 if price_change > 0 else forecast_value * 0.95),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center', fontsize=12)

    # Enhance chart
    ax.set_title('Stock Price Forecast', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True)

    plt.tight_layout()
    return fig

def plot_validation_results(actual, predicted, title='Model Validation'):
    """
    Plot actual vs predicted values

    Parameters:
    -----------
    actual : pandas.Series
        Actual values
    predicted : pandas.Series or array-like
        Predicted values
    title : str
        Chart title

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot actual and predicted
    ax.plot(actual.index, actual, 'b-', label='Actual', linewidth=2)

    if isinstance(predicted, pd.Series) and predicted.index.equals(actual.index):
        ax.plot(predicted.index, predicted, 'r-', label='Predicted', alpha=0.8)
    else:
        ax.plot(actual.index, predicted, 'r-', label='Predicted', alpha=0.8)

    # Add error area
    if isinstance(predicted, pd.Series) and predicted.index.equals(actual.index):
        error = actual - predicted
    else:
        error = actual - pd.Series(predicted, index=actual.index)

    ax.fill_between(actual.index, actual, predicted, color='red', alpha=0.2)

    # Enhance chart
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True)

    plt.tight_layout()
    return fig

# Main prediction function
async def predict_stock_with_arima(db_path, visualize=True, seasonal_analysis=False, test_size=DEFAULT_TEST_SIZE):
    """
    Predict next day's stock price using ARIMA model

    Parameters:
    -----------
    visualize : bool
        Whether to show visualizations
    seasonal_analysis : bool
        Whether to perform seasonal decomposition
    test_size : int
        Number of days to use for testing

    Returns:
    --------
    dict
        Prediction results and model metrics
    """
    try:
        # 1. Get and preprocess data
        print("Получение данных по акциям...")

        # Get data from the provided function
        # df = await get_df("BBG004730ZJ9")

        df = load_data(db_path)
        df = df.drop(["figi"], axis=1)

        print(f"Данные получены: {df.shape[0]} наблюдений")

        # Preprocess data
        df = preprocess_stock_data(df)

        # Extract close price for prediction
        close_prices = df['close']

        # 2. Perform exploratory analysis
        print("\n--- Анализ временных рядов ---")
        print(f"Диапазон дат: {df.index.min().date()} to {df.index.max().date()} ({len(df)} дней)")
        print(f"Диапазон цен: {close_prices.min():.2f} to {close_prices.max():.2f}")
        print(f"Текущая цена: {close_prices.iloc[-1]:.2f}")

        # Trend analysis
        trend_analysis = analyze_trend(close_prices)
        print(f"Недавний тренд: {trend_analysis['trend']}")
        print(f"Волатильность за 30 дней: {trend_analysis['volatility']:.2f} ({trend_analysis['range_percent']:.2f}%)")

        # Optional seasonal decomposition
        if seasonal_analysis and len(close_prices) > 2*252:  # Need at least 2 years for seasonal analysis
            try:
                print("\nPerforming seasonal decomposition...")
                decomposition = seasonal_decompose(close_prices, model='additive', period=252)  # ~252 trading days per year

                if visualize:
                    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
                    decomposition.observed.plot(ax=axes[0], title='Observed')
                    decomposition.trend.plot(ax=axes[1], title='Trend')
                    decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
                    decomposition.resid.plot(ax=axes[3], title='Residuals')
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"Seasonal decomposition failed: {e}")

        # 3. Check stationarity and apply differencing
        print("\n--- Анализ стационарности ---")
        is_stationary, adf_results = check_stationarity(close_prices)

        if is_stationary:
            print("Временной ряд уже стационарен.")
            d = 0
            diff_series = close_prices
        else:
            print("Временной ряд не стационарен. Поиск оптимального порядка дифференцирования...")
            d, diff_series = find_optimal_differencing(close_prices)
            print(f"Оптимальный порядок дифференцирования: d={d}")

        # 4. Find optimal ARIMA parameters
        print("\n--- Выбор модели ARIMA ---")
        optimal_params, aic = find_optimal_arima_params(close_prices, d)
        p, d, q = optimal_params
        print(f"Выбрана модель ARIMA({p},{d},{q}) с AIC: {aic:.2f}")

        # 5. Train-test split for validation
        test_size = min(test_size, len(close_prices) // 4)  # Limit test size
        train_data = close_prices[:-test_size]
        test_data = close_prices[-test_size:]

        print(f"\n--- Валидация модели (использование последних {test_size} наблюдений) ---")

        # 6. Train model on training data
        try:
            validation_model = ARIMA(train_data, order=optimal_params)
            validation_fit = validation_model.fit()

            # 7. Forecast test period
            forecast = validation_fit.forecast(steps=test_size)

            # 8. Calculate error metrics
            metrics = calculate_prediction_metrics(test_data.values, forecast)

            print(f"MAE: {metrics['mae']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")

            if metrics['mape'] is not None:
                print(f"MAPE: {metrics['mape']:.2f}%")

            print(f"Точность направления: {metrics['directional_accuracy']:.2f}%")

            # Determine prediction confidence
            confidence = determine_confidence(metrics)
            print(f"Уверенность: {confidence}" )
            # print(f"Точность направления: {confidence['directional_accuracy']:.2f}%")

            # Visualize validation results
            if visualize:
                fig = plot_validation_results(test_data, forecast)
                plt.show()
        except Exception as e:
            print(f"Error in model validation: {e}")
            metrics = {'mae': None, 'rmse': None, 'mape': None, 'directional_accuracy': None}
            confidence = "Unknown"

        # 9. Train final model on all data
        print("\n--- Финальный прогноз ---")
        try:
            final_model = ARIMA(close_prices, order=optimal_params)
            final_fit = final_model.fit()

            # 10. Forecast next day
            next_day_forecast = final_fit.forecast(steps=1)[0]
            last_price = close_prices.iloc[-1]
            price_change = next_day_forecast - last_price
            price_change_percent = (price_change / last_price) * 100

            # Next trading day (approximation)
            forecast_date = close_prices.index[-1] + pd.Timedelta(days=1)

            # Prediction direction
            if price_change > 0:
                direction = "UP ⬆️"
            elif price_change < 0:
                direction = "DOWN ⬇️"
            else:
                direction = "FLAT ➡️"

            print(f"Последняя цена закрытия ({close_prices.index[-1].date()}): {last_price:.4f}")
            print(f"Прогнозируемая следующая цена ({forecast_date.date()}): {next_day_forecast:.4f}")
            print(f"Изменение: {price_change:.4f} ({price_change_percent:.2f}%)")
            print(f"Прогнозируемое направление: {direction}")

            # Visualize forecast
            if visualize:
                fig = plot_forecast_chart(close_prices, forecast_date, next_day_forecast)
                plt.show()

            # 11. Return prediction results
            return {
                'last_date': close_prices.index[-1].strftime('%Y-%m-%d'),
                'last_price': float(last_price),
                'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                'forecast_price': float(next_day_forecast),
                'price_change': float(price_change),
                'price_change_percent': float(price_change_percent),
                'direction': direction,
                'confidence': confidence if metrics['mape'] is not None else "Unknown",
                'trend': trend_analysis['trend'],
                'model': {
                    'type': 'ARIMA',
                    'order': optimal_params,
                    'aic': float(aic)
                },
                'metrics': {
                    'mae': float(metrics['mae']) if metrics['mae'] is not None else None,
                    'rmse': float(metrics['rmse']) if metrics['rmse'] is not None else None,
                    'mape': float(metrics['mape']) if metrics['mape'] is not None else None,
                    'directional_accuracy': float(metrics['directional_accuracy']) if metrics['directional_accuracy'] is not None else None
                }
            }

        except Exception as e:
            print(f"Error in final prediction: {e}")
            return {
                'error': str(e),
                'last_date': close_prices.index[-1].strftime('%Y-%m-%d'),
                'last_price': float(close_prices.iloc[-1]),
                'model': {'type': 'ARIMA', 'order': optimal_params}
            }

    except Exception as e:
        print(f"Fatal error in prediction pipeline: {e}")
        return {'error': str(e)}

# Synchronous wrapper for train_models.py compatibility
def main(db_path):
    """
    Synchronous wrapper for ARIMA prediction.

    This function is called by train_models.py to maintain consistency
    with other models.

    ARIMA model doesn't have the same data leakage issue as ML models
    because it doesn't use a scaler - it works directly with the time series.
    """
    result = asyncio.run(predict_stock_with_arima(db_path=db_path, visualize=False))

    # Print results in format compatible with train_models.py metric extraction
    if 'error' not in result or result.get('prediction'):
        pred = result.get('prediction', {})
        metrics = result.get('metrics', {})

        print(f"\nТекущая цена: {pred.get('last_price', 0):.4f}")
        print(f"Прогнозируемая цена: {pred.get('price', 0):.4f}")
        print(f"Ожидаемое изменение: {pred.get('change_percent', 0):.2f}%")
        print(f"Торговый сигнал: {pred.get('signal', 'NEUTRAL')}")

        if metrics:
            print(f"\nМетрики модели:")
            if metrics.get('mae') is not None:
                print(f"MAE: {metrics['mae']:.6f}")
            if metrics.get('rmse') is not None:
                print(f"RMSE: {metrics['rmse']:.6f}")
            if metrics.get('mape') is not None:
                print(f"MAPE: {metrics['mape']:.2f}")
            if metrics.get('directional_accuracy') is not None:
                print(f"Direction Accuracy: {metrics['directional_accuracy']:.2f}")

    return result


# Run the prediction function if script is executed directly
if __name__ == "__main__":
    import time

    start_time = time.time()
    result = main("BBG000Q7ZZY2")
    end_time = time.time()

    print(f"\nPrediction completed in {end_time - start_time:.2f} seconds")
