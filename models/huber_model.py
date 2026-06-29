"""
Huber Regression — робастная линейная модель.

В отличие от Ridge (квадратичная функция потерь), Huber использует
комбинацию L1 (для выбросов) + L2 (для нормальных точек). Это делает
её устойчивой к выбросам в данных — что особенно важно для returns,
где иногда бывают большие движения (±5-10%) которые сбивают MSE.

Часто даёт лучшую Direction Accuracy чем Ridge на финансовых данных.
"""
from sklearn.linear_model import HuberRegressor

from core.feature_engineering import FeatureSet
from core.base_model import SklearnTradeModel


class HuberTradeModel(SklearnTradeModel):
    """Huber regression — robust linear для returns prediction."""

    REQUIRED_FEATURES = {
        FeatureSet.BASIC, FeatureSet.VOLUME,
        FeatureSet.VOLATILITY, FeatureSet.MOMENTUM,
    }
    MODEL_NAME = "huber"

    def __init__(
        self,
        # epsilon — порог между L2 (квадратичная) и L1 (линейная) потерей.
        # 1.35 — стандартное "robust" значение из литературы.
        epsilon: float = 1.35,
        # alpha — L2 регуляризация (как у Ridge), помогает с
        # коррелированными TA-фичами.
        alpha: float = 1.0,
        max_iter: int = 200,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(test_size=test_size, random_state=random_state)
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter

    def _create_model(self) -> HuberRegressor:
        return HuberRegressor(
            epsilon=self.epsilon,
            alpha=self.alpha,
            max_iter=self.max_iter,
        )


# Совместимость с train_models.py
def main(db_path):
    """Запускает обучение HuberTradeModel через универсальный раннер."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.load_data_method import load_data
    import numpy as np

    print(f"Загрузка данных для {db_path}...")
    df = load_data(db_path)
    if 'figi' in df.columns:
        df = df.drop(['figi'], axis=1)

    if df.empty or len(df) < 100:
        print(f"ОШИБКА: Недостаточно данных. Минимум 100 наблюдений.")
        return None

    print(
        f"Загружено {len(df)} записей за период с "
        f"{df['timestamp'].min()} по {df['timestamp'].max()}"
    )

    model = HuberTradeModel()

    try:
        metrics = model.train(df)
    except Exception as e:
        print(f"ОШИБКА при обучении: {e}")
        return None

    # Прогноз
    pred = model.predict_next(df)
    current_price = float(pred['current_price'])
    predicted_price = float(pred['predicted_price'])
    expected_change = float(pred['expected_change'])
    signal = pred['signal']

    print(f"\nТекущая цена: {current_price:.4f}")
    print(f"Прогнозируемая цена: {predicted_price:.4f}")
    print(f"Ожидаемое изменение: {expected_change:+.2f}%")
    print(f"Торговый сигнал: {signal}")

    print(f"\nМетрики на тестовой выборке:")
    print(f"MSE: {metrics.get('test_mse', 0):.6f}")
    print(f"RMSE: {metrics.get('test_rmse', 0):.6f}")
    print(f"MAE: {metrics.get('test_mae', 0):.6f}")
    print(f"R²: {metrics.get('test_r2', 0):.6f}")
    print(f"MAPE: {metrics.get('test_mape', 0):.2f}")
    print(f"Direction Accuracy: {metrics.get('test_direction_accuracy', 0):.2f}")

    # Простой бэктест
    print("\nРетроспективная оценка торговых сигналов:")
    df_features = model.prepare_features(df)
    test_size = int(len(df_features) * 0.2)
    if test_size >= 2:
        test_df = df_features.iloc[-test_size:]
        X_test = test_df[model.feature_columns]
        X_test_scaled = model.scaler.transform(X_test)
        preds = model._predict(X_test_scaled)
        # Для returns target преды это % изменения
        actual_returns = test_df['next_return'].values if 'next_return' in test_df.columns \
            else test_df['next_close'].pct_change().values * 100
        valid = ~np.isnan(actual_returns) & ~np.isnan(preds)
        actual_returns = actual_returns[valid]
        preds = preds[valid]
        if len(actual_returns) > 1:
            signals = np.sign(preds)
            strategy_returns = signals * actual_returns / 100.0
            total = len(strategy_returns)
            profitable = int(np.sum(strategy_returns > 0))
            cum_return = float((np.prod(1 + strategy_returns) - 1) * 100)
            profit_sum = float(np.sum(strategy_returns[strategy_returns > 0]))
            loss_sum = float(abs(np.sum(strategy_returns[strategy_returns < 0])))
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
            if len(strategy_returns) > 1:
                sharpe = float(np.mean(strategy_returns) /
                               (np.std(strategy_returns) + 1e-9) * np.sqrt(252))
                eq = (1 + strategy_returns).cumprod()
                peak = np.maximum.accumulate(eq)
                dd = (peak - eq) / peak
                max_dd = float(np.max(dd) * 100)
            else:
                sharpe, max_dd = 0.0, 0.0

            print(f"Всего сделок: {total}")
            print(f"Прибыльных сделок: {profitable} ({profitable / total * 100:.2f}%)")
            print(f"Общая доходность: {cum_return:.2f}%")
            print(f"Коэффициент прибыли (Profit Factor): {profit_factor:.2f}")
            print(f"Sharpe Ratio: {sharpe:.4f}")
            print(f"Максимальная просадка: {max_dd:.2f}%")

    return model


if __name__ == "__main__":
    main("BBG004730N88")  # SBER
