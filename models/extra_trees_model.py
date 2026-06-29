"""
Extra Trees Regressor — расширенный Random Forest.

Отличия от RandomForest:
- В каждом узле порог сплита выбирается СЛУЧАЙНО (не по best),
  что добавляет вариативности → меньше переобучения
- Часто слегка лучше RF на шумных данных типа финансовых

Использует ту же логику ensemble, но с большей рандомизацией.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from core.feature_engineering import FeatureSet
from core.base_model import SklearnTradeModel


class ExtraTreesTradeModel(SklearnTradeModel):
    """Extra Trees — рандомизированный аналог Random Forest."""

    REQUIRED_FEATURES = {
        FeatureSet.BASIC, FeatureSet.VOLUME,
        FeatureSet.VOLATILITY, FeatureSet.MOMENTUM,
    }
    MODEL_NAME = "extra_trees"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(test_size=test_size, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def _create_model(self) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )


def main(db_path):
    """Запускает обучение ExtraTreesTradeModel."""
    from utils.load_data_method import load_data

    print(f"Загрузка данных для {db_path}...")
    df = load_data(db_path)
    if 'figi' in df.columns:
        df = df.drop(['figi'], axis=1)

    if df.empty or len(df) < 100:
        print(f"ОШИБКА: Недостаточно данных.")
        return None

    print(
        f"Загружено {len(df)} записей за период с "
        f"{df['timestamp'].min()} по {df['timestamp'].max()}"
    )

    model = ExtraTreesTradeModel()
    try:
        metrics = model.train(df)
    except Exception as e:
        print(f"ОШИБКА при обучении: {e}")
        return None

    pred = model.predict_next(df)
    print(f"\nТекущая цена: {pred['current_price']:.4f}")
    print(f"Прогнозируемая цена: {pred['predicted_price']:.4f}")
    print(f"Ожидаемое изменение: {pred['expected_change']:+.2f}%")
    print(f"Торговый сигнал: {pred['signal']}")

    print(f"\nМетрики на тестовой выборке:")
    print(f"MSE: {metrics.get('test_mse', 0):.6f}")
    print(f"RMSE: {metrics.get('test_rmse', 0):.6f}")
    print(f"MAE: {metrics.get('test_mae', 0):.6f}")
    print(f"R²: {metrics.get('test_r2', 0):.6f}")
    print(f"MAPE: {metrics.get('test_mape', 0):.2f}")
    print(f"Direction Accuracy: {metrics.get('test_direction_accuracy', 0):.2f}")

    print("\nРетроспективная оценка торговых сигналов:")
    df_features = model.prepare_features(df)
    test_size = int(len(df_features) * 0.2)
    if test_size >= 2:
        test_df = df_features.iloc[-test_size:]
        X_test = test_df[model.feature_columns]
        X_test_scaled = model.scaler.transform(X_test)
        preds = model._predict(X_test_scaled)
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
