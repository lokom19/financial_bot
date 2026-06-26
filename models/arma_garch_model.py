"""
ARMA + GARCH(1,1) модель для прогнозирования цены и волатильности.

Идея:
1. ARMA(p,q) моделирует среднее доходностей (логарифмических).
2. GARCH(1,1) моделирует условную волатильность остатков ARMA.
3. Прогноз цены: P_{t+1} = P_t * exp(mean_forecast).
4. Доверительный интервал: ±1.96 * sqrt(variance_forecast).

Подходит для финансовых рядов с кластеризацией волатильности
(периоды высокой/низкой дисперсии чередуются).
"""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import ARX, GARCH
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.load_data_method import load_data

warnings.filterwarnings("ignore")

# Параметры по умолчанию
DEFAULT_TEST_SIZE = 60
DEFAULT_AR = 2          # Порядок AR
DEFAULT_GARCH_P = 1     # Порядок ARCH (лаг квадратов остатков)
DEFAULT_GARCH_Q = 1     # Порядок GARCH (лаг условной дисперсии)


def _prepare_returns(df: pd.DataFrame) -> tuple:
    """Возвращает (close_series, log_returns) с DatetimeIndex."""
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    close = df["close"].astype(float)
    log_returns = np.log(close).diff().dropna() * 100  # в процентах для устойчивости GARCH
    return close, log_returns


def _fit_arma_garch(returns: pd.Series, ar: int = DEFAULT_AR,
                    p: int = DEFAULT_GARCH_P, q: int = DEFAULT_GARCH_Q):
    """Фитит ARMA(ar,0)-GARCH(p,q). Возвращает result-объект arch."""
    model = arch_model(
        returns,
        mean="AR", lags=ar,
        vol="Garch", p=p, q=q,
        dist="normal",
        rescale=False,
    )
    return model.fit(disp="off", show_warning=False)


def main(db_path: str):
    """
    Основная функция — совместима с train_models.py.
    Печатает метрики и прогноз в формате, который парсит extract_metrics.
    """
    print(f"Загрузка данных для {db_path}...")
    try:
        df = load_data(db_path)
        if "figi" in df.columns:
            df = df.drop(["figi"], axis=1)
    except Exception as e:
        print(f"ОШИБКА загрузки: {e}")
        return None

    if df.empty or len(df) < 100:
        print(f"ОШИБКА: Недостаточно данных. Минимум 100 наблюдений, есть {len(df)}.")
        return None

    print(f"Загружено {len(df)} записей за период с {df['timestamp'].min()} по {df['timestamp'].max()}")

    close, returns = _prepare_returns(df)
    print(f"Лог-доходностей: {len(returns)}, среднее: {returns.mean():.4f}%, std: {returns.std():.4f}%")

    # ----- Train / test split (хронологический) -----
    test_size = min(DEFAULT_TEST_SIZE, len(returns) // 5)
    train_returns = returns.iloc[:-test_size]
    test_returns = returns.iloc[-test_size:]
    train_close = close.iloc[:-test_size]
    test_close = close.iloc[-test_size:]

    # ----- Подгонка модели -----
    print(f"\nПодгонка ARMA({DEFAULT_AR},0)-GARCH({DEFAULT_GARCH_P},{DEFAULT_GARCH_Q})...")
    try:
        res = _fit_arma_garch(train_returns)
        print(f"AIC: {res.aic:.2f}, BIC: {res.bic:.2f}")
    except Exception as e:
        print(f"ОШИБКА подгонки: {e}")
        return None

    # ----- Walk-forward forecast по тестовой выборке -----
    # На каждом шаге используем расширяющееся окно, чтобы получить
    # последовательность одно-шаговых прогнозов.
    print(f"\nПрогноз на {test_size} тестовых наблюдений (one-step rolling)...")
    forecasted_returns = []
    for i in range(test_size):
        window = returns.iloc[: -test_size + i] if i > 0 else train_returns
        try:
            r = _fit_arma_garch(window)
            fc = r.forecast(horizon=1, reindex=False)
            mean_fc = float(fc.mean.iloc[-1, 0])
        except Exception:
            mean_fc = 0.0
        forecasted_returns.append(mean_fc)

    forecasted_returns = np.array(forecasted_returns)
    actual_returns_test = test_returns.values

    # Восстанавливаем прогнозные цены: P_pred[t+1] = P_actual[t] * exp(r/100)
    base_prices = np.concatenate([[train_close.iloc[-1]], test_close.values[:-1]])
    predicted_prices = base_prices * np.exp(forecasted_returns / 100.0)
    actual_prices = test_close.values

    # ----- Метрики -----
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(actual_prices, predicted_prices))
    try:
        r2 = float(r2_score(actual_prices, predicted_prices))
    except Exception:
        r2 = 0.0
    mape = float(np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100)

    # Direction accuracy (относительно текущей цены)
    direction_actual = actual_prices > base_prices
    direction_pred = predicted_prices > base_prices
    direction_acc = float(np.mean(direction_actual == direction_pred) * 100)

    print(f"\nМетрики на тестовой выборке:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"MAPE: {mape:.2f}")
    print(f"Direction Accuracy: {direction_acc:.2f}")

    # ----- Финальный прогноз на завтра -----
    print("\nФинальный прогноз...")
    try:
        final_res = _fit_arma_garch(returns)
        fc = final_res.forecast(horizon=1, reindex=False)
        next_return_pct = float(fc.mean.iloc[-1, 0])
        next_variance = float(fc.variance.iloc[-1, 0])
    except Exception as e:
        print(f"Ошибка финального прогноза: {e}")
        next_return_pct, next_variance = 0.0, 0.0

    current_price = float(close.iloc[-1])
    predicted_price = current_price * np.exp(next_return_pct / 100.0)
    expected_change = (predicted_price - current_price) / current_price * 100.0
    volatility = float(np.sqrt(next_variance))  # прогнозная std в %

    if expected_change >= 0.5:
        signal = "BUY"
    elif expected_change <= -0.5:
        signal = "SELL"
    elif abs(expected_change) < 0.1:
        signal = "NEUTRAL"
    else:
        signal = "HOLD"

    print(f"Текущая цена: {current_price:.4f}")
    print(f"Прогнозируемая цена: {predicted_price:.4f}")
    print(f"Ожидаемое изменение: {expected_change:+.2f}%")
    print(f"Прогнозная волатильность (1д): {volatility:.2f}%")
    print(f"Торговый сигнал: {signal}")

    # ----- Бэктест на тестовой выборке -----
    print("\nРетроспективная оценка торговых сигналов:")
    if len(predicted_prices) > 1:
        signals = np.sign(predicted_prices - base_prices)
        actual_pct_returns = (actual_prices - base_prices) / base_prices
        strategy_returns = signals * actual_pct_returns

        total_trades = int(np.sum(np.abs(np.diff(signals)) > 0) + 1)
        profitable = int(np.sum(strategy_returns > 0))
        win_rate = profitable / len(strategy_returns) * 100 if len(strategy_returns) else 0.0

        cumulative = (1 + strategy_returns).cumprod()
        cum_return = float((cumulative[-1] - 1) * 100) if len(cumulative) else 0.0

        profit_sum = float(np.sum(strategy_returns[strategy_returns > 0]))
        loss_sum = float(abs(np.sum(strategy_returns[strategy_returns < 0])))
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float("inf")

        if len(strategy_returns) > 1:
            sharpe = float(
                np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
            )
            peak = np.maximum.accumulate(cumulative)
            drawdowns = (peak - cumulative) / peak
            max_dd = float(np.max(drawdowns) * 100)
        else:
            sharpe, max_dd = 0.0, 0.0

        print(f"Всего сделок: {total_trades}")
        print(f"Прибыльных сделок: {profitable} ({win_rate:.2f}%)")
        print(f"Общая доходность: {cum_return:.2f}%")
        print(f"Коэффициент прибыли (Profit Factor): {profit_factor:.2f}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"Максимальная просадка: {max_dd:.2f}%")

    return {
        "metrics": {
            "mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
            "mape": mape, "directional_accuracy": direction_acc,
        },
        "prediction": {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "expected_change": expected_change,
            "signal": signal,
            "volatility": volatility,
        },
    }


if __name__ == "__main__":
    main("BBG004730N88")  # SBER
