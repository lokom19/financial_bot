import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from utils.load_data_method import load_data

# Import core modules for new implementation
from core.feature_engineering import create_features as core_create_features, FeatureSet
from core.data_pipeline import DataPipeline
from core.metrics import calculate_metrics, calculate_direction_accuracy, calculate_trading_signal
from core.base_model import BaseTradeModel

# Отключаем все RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# NEW IMPLEMENTATION: XGBoostTradeModel using core modules
# ============================================================

class XGBoostTradeModelNew(BaseTradeModel):
    """XGBoost model for price prediction with no data leakage."""

    REQUIRED_FEATURES = {FeatureSet.BASIC, FeatureSet.VOLUME, FeatureSet.VOLATILITY, FeatureSet.MOMENTUM}
    MODEL_NAME = "xgboost_model"

    def __init__(self, params=None, test_size: float = 0.2, random_state: int = 42):
        super().__init__(test_size=test_size, random_state=random_state)
        self.params = params or {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        self.model = None

    def _create_model(self):
        return xgb.XGBRegressor(**self.params)

    def _fit_model(self, X_train, y_train, X_val, y_val):
        self.model = self._create_model()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        # Extract feature importances
        if self.feature_columns:
            self.feature_importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)

    def _predict(self, X):
        return self.model.predict(X)


# ============================================================
# LEGACY IMPLEMENTATION: Kept for backward compatibility
# ============================================================

# Функция для создания технических индикаторов (legacy)
def create_features(df):
    df_features = df.copy()
    df_features["sma_5"] = df["close"].rolling(window=5).mean()
    df_features["sma_10"] = df["close"].rolling(window=10).mean()
    df_features["sma_20"] = df["close"].rolling(window=20).mean()
    df_features["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df_features["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df_features["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df_features["close_minus_sma_5"] = df["close"] - df_features["sma_5"]
    df_features["close_minus_sma_10"] = df["close"] - df_features["sma_10"]
    df_features["close_rel_sma_5"] = df["close"] / df_features["sma_5"] - 1
    df_features["close_rel_sma_10"] = df["close"] / df_features["sma_10"] - 1
    df_features["volume_log"] = np.log1p(df["volume"])
    df_features["volume_sma_5"] = df_features["volume_log"].rolling(window=5).mean()
    df_features["volume_ratio"] = (
        df_features["volume_log"] / df_features["volume_sma_5"]
    )
    df_features["price_change_1"] = df["close"].pct_change(periods=1)
    df_features["price_change_3"] = df["close"].pct_change(periods=3)
    df_features["price_change_5"] = df["close"].pct_change(periods=5)
    df_features["volatility_5"] = (
        df["close"].rolling(window=5).std() / df_features["sma_5"]
    )
    df_features["volatility_10"] = (
        df["close"].rolling(window=10).std() / df_features["sma_10"]
    )
    df_features["high_low_ratio"] = df["high"] / df["low"]
    df_features["prev_close"] = df["close"].shift(1)
    df_features["tr"] = df_features.apply(
        lambda x: max(
            x["high"] - x["low"],
            abs(x["high"] - x["prev_close"]),
            abs(x["low"] - x["prev_close"]),
        )
        if not pd.isna(x["prev_close"])
        else np.nan,
        axis=1,
    )
    df_features["atr_14"] = df_features["tr"].rolling(window=14).mean()
    df_features = df_features.drop(["prev_close", "tr"], axis=1)
    df_features["next_close"] = df_features["close"].shift(-1)

    # df_features = df_features.dropna()
    for col in df_features.columns:
        if col != "next_close" and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())
    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # df_features = df_features.dropna()
    for col in df_features.columns:
        if col != "next_close" and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    return df_features


# Модель для прогнозирования с использованием XGBoost
class XGBoostTradeModel:
    """
    Модель для прогнозирования цен финансовых инструментов с использованием XGBoost.

    Класс реализует модель машинного обучения на основе градиентного бустинга
    для предсказания будущих цен закрытия. Включает функции создания технических
    индикаторов, проверки качества данных, обучения модели и генерации торговых сигналов.

    Attributes:
        params (dict): Параметры модели XGBoost (learning_rate, max_depth, и т.д.)
        n_estimators (int): Количество деревьев в ансамбле
        early_stopping_rounds (int): Количество раундов для ранней остановки
        model: Обученная модель XGBoost
        scaler: Масштабировщик признаков (RobustScaler)
        feature_columns: Список используемых признаков
        last_price: Последняя известная цена
        feature_importances: Важность признаков модели
    """

    def __init__(self, params=None):
        # Параметры XGBoost по умолчанию (без n_estimators и early_stopping_rounds в params)
        self.params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "alpha": 0.01,
            "lambda": 1,
            "seed": 42,
        }
        self.n_estimators = 100  # Отдельно
        self.early_stopping_rounds = 20  # Отдельно

        if params is not None:
            self.params.update(params)

        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.last_price = None
        self.feature_importances = None
        self.data_quality_checked = False

    def check_data_quality(self, features):
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        highly_correlated = [
            column for column in upper.columns if any(upper[column] > 0.95)
        ]
        if highly_correlated:
            print(
                f"ВНИМАНИЕ: Обнаружены сильно коррелирующие признаки: {highly_correlated}"
            )
            print(
                "Это может вызвать нестабильность в модели. Рассмотрите удаление некоторых из них."
            )
        for col in features.columns:
            if features[col].std() / (features[col].mean() + 1e-10) > 100:
                print(
                    f"ВНИМАНИЕ: Очень высокий разброс значений в колонке {col}. Ограничиваем выбросы."
                )
                q1 = features[col].quantile(0.01)
                q3 = features[col].quantile(0.99)
                features[col] = features[col].clip(q1, q3)
            if features[col].std() < 1e-6:
                print(
                    f"ВНИМАНИЕ: Колонка {col} имеет близкую к нулю дисперсию. Рассмотрите её удаление."
                )
        self.data_quality_checked = True
        return features

    def prepare_data(self, df):
        timestamps = df["timestamp"]
        y = df["next_close"]
        features = df.drop(["timestamp", "next_close"], axis=1)
        if "volume" in features.columns and "volume_log" in features.columns:
            features = features.drop(["volume"], axis=1)
        if not self.data_quality_checked:
            features = self.check_data_quality(features)
        features = features.replace([np.inf, -np.inf], np.nan)
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)
        for col in features.columns:
            q_low = features[col].quantile(0.01)
            q_high = features[col].quantile(0.99)
            features[col] = features[col].clip(lower=q_low, upper=q_high)
        cols_to_drop = [
            col
            for col in features.columns
            if features[col].std() < 1e-8 or features[col].isna().any()
        ]
        if cols_to_drop:
            print(f"Удаляем колонки: {cols_to_drop}")
            features = features.drop(cols_to_drop, axis=1)
            if len(features.columns) == 0:
                raise ValueError(
                    "После удаления проблемных колонок не осталось признаков"
                )
        self.feature_columns = features.columns
        print("\nСтатистика признаков после обработки выбросов:")
        print(features.describe().loc[["min", "max", "mean", "std"]].T.head())
        if features.isna().any().any():
            print("ВНИМАНИЕ: В данных остались NaN после обработки.")
            features = features.fillna(features.median())
        X_original = features.copy()
        try:
            X_scaled = self.scaler.fit_transform(features)
        except Exception as e:
            print(f"Ошибка при масштабировании: {e}")
            X_scaled = np.zeros(features.shape)
            for i, col in enumerate(features.columns):
                col_mean = features[col].mean()
                col_std = features[col].std()
                if col_std > 1e-10:
                    X_scaled[:, i] = (features[col] - col_mean) / col_std
                else:
                    X_scaled[:, i] = 0
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print(
                "ВНИМАНИЕ: После масштабирования обнаружены NaN или Inf. Заменяем на 0."
            )
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(X_scaled)) > 1e6:
            print("ВНИМАНИЕ: Большие значения после масштабирования. Ограничиваем.")
            X_scaled = np.clip(X_scaled, -1e6, 1e6)
        return X_original, X_scaled, y, timestamps

    def train(self, X_original, y, timestamps=None, X_scaled=None, df_features=None):
        nan_mask = y.isna()
        if nan_mask.any():
            print(
                f"ВНИМАНИЕ: Обнаружено {nan_mask.sum()} NaN значений в целевой переменной. Эти строки будут удалены."
            )
            X_original = X_original[~nan_mask]
            y = y[~nan_mask]
            if timestamps is not None:
                timestamps = timestamps[~nan_mask]

        if np.isnan(X_original.values).any() or np.isinf(X_original.values).any():
            print("КРИТИЧЕСКАЯ ОШИБКА: В данных обнаружены NaN или Inf.")
            X_original = X_original.replace([np.inf, -np.inf], np.nan).fillna(
                X_original.median()
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_original, y, test_size=0.2, random_state=42, shuffle=False
        )
        if timestamps is not None:
            _, timestamps_test = train_test_split(
                timestamps, test_size=0.2, random_state=42, shuffle=False
            )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        try:
            eval_set = [(dtrain, "train"), (dtest, "eval")]
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=100,
            )
        except Exception as e:
            print(f"Ошибка при обучении XGBoost: {e}")
            safe_params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.01,
                "max_depth": 3,
                "min_child_weight": 3,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "gamma": 0.1,
                "alpha": 1,
                "lambda": 10,
                "seed": 42,
            }
            try:
                self.model = xgb.train(
                    safe_params,
                    dtrain,
                    num_boost_round=50,
                    evals=eval_set,
                    early_stopping_rounds=10,
                    verbose_eval=100,
                )
            except Exception as e2:
                print(f"Повторная ошибка: {e2}")
                return None, None, None, None
        train_preds = self.model.predict(dtrain)
        test_preds = self.model.predict(dtest)
        if np.isnan(train_preds).any() or np.isinf(train_preds).any():
            print("ВНИМАНИЕ: NaN или Inf в предсказаниях train.")
            train_preds = np.nan_to_num(train_preds)
        if np.isnan(test_preds).any() or np.isinf(test_preds).any():
            print("ВНИМАНИЕ: NaN или Inf в предсказаниях test.")
            test_preds = np.nan_to_num(test_preds)
        # Исправление feature_importances
        importance_dict = self.model.get_score(importance_type="gain")
        importance_values = [importance_dict.get(f, 0.0) for f in self.feature_columns]
        self.feature_importances = pd.DataFrame(
            {"Feature": self.feature_columns, "Importance": importance_values}
        ).sort_values("Importance", ascending=False)
        train_metrics = self.calculate_metrics(y_train, train_preds)
        test_metrics = self.calculate_metrics(y_test, test_preds)
        print("\n===== Метрики на обучающей выборке =====")
        self.print_metrics(train_metrics)
        print("\n===== Метрики на тестовой выборке =====")
        self.print_metrics(test_metrics)

        if np.isnan(y.iloc[-1]) and df_features is not None:
            # Берем последнюю фактическую цену закрытия вместо NaN
            self.last_price = df_features.iloc[-1]["close"]
        else:
            self.last_price = y.iloc[-1]

        return X_test, y_test, test_preds, test_metrics

    def calculate_metrics(self, y_true, y_pred):
        try:
            # Фильтруем NaN значения перед вычислением метрик
            mask = ~np.isnan(y_true)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]

            if len(y_true_filtered) == 0:
                return {
                    "MSE": float("nan"),
                    "RMSE": float("nan"),
                    "MAE": float("nan"),
                    "R²": float("nan"),
                    "MAPE": float("nan"),
                    "Direction Accuracy": float("nan"),
                }

            mse = mean_squared_error(y_true_filtered, y_pred_filtered)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_filtered, y_pred_filtered)

            # Безопасное вычисление R² с обработкой исключений
            try:
                r2 = r2_score(y_true_filtered, y_pred_filtered)
            except Exception as e:
                print(f"Ошибка при вычислении R²: {e}")
                r2 = float("nan")

            # Безопасное вычисление MAPE с обработкой деления на ноль
            with np.errstate(divide="ignore", invalid="ignore"):
                mape_raw = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)
                mape_raw = np.where(
                    np.isinf(mape_raw) | np.isnan(mape_raw), 0, mape_raw
                )
                mape = np.mean(mape_raw) * 100

            # Направление цены (бинарный классификатор)
            if len(y_true_filtered) > 1:
                direction_true = (
                    np.diff(np.append(y_true_filtered.iloc[0], y_true_filtered.values))
                    > 0
                )
                direction_pred = (
                    np.diff(np.append(y_true_filtered.iloc[0], y_pred_filtered)) > 0
                )
                direction_accuracy = np.mean(direction_true == direction_pred) * 100
            else:
                direction_accuracy = float("nan")
            return {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2,
                "MAPE": mape,
                "Direction Accuracy": direction_accuracy,
            }
        except Exception as e:
            print(f"Ошибка при вычислении метрик: {e}")
            return {
                k: float("nan")
                for k in ["MSE", "RMSE", "MAE", "R²", "MAPE", "Direction Accuracy"]
            }

    def print_metrics(self, metrics):
        for name, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                print(f"{name}: Ошибка вычисления")
            else:
                print(f"{name}: {value:.4f}")

    def predict_next(self, current_data):
        try:
            if self.model is None:
                print("ОШИБКА: Модель не обучена. Запустите train() сначала.")
                return None

            current_price = current_data["close"].iloc[0]

            missing_cols = []
            for col in self.feature_columns:
                if col not in current_data.columns:
                    if col == "volume_log" and "volume" in current_data.columns:
                        # Создаем логарифмированную версию, если она отсутствует
                        current_data["volume_log"] = np.log1p(current_data["volume"])
                    else:
                        missing_cols.append(col)

            if missing_cols:
                print(f"ВНИМАНИЕ: В данных отсутствуют колонки: {missing_cols}")
                # Создаем отсутствующие колонки с нулями
                for col in missing_cols:
                    current_data[col] = 0

            features = current_data[self.feature_columns].copy()
            features = features.replace([np.inf, -np.inf], np.nan)
            for col in features.columns:
                if features[col].isna().any():
                    features[col] = features[col].fillna(features[col].median())
            for col in features.columns:
                q_low = (
                    features[col].quantile(0.01)
                    if len(features) > 10
                    else features[col].min()
                )
                q_high = (
                    features[col].quantile(0.99)
                    if len(features) > 10
                    else features[col].max()
                )
                features[col] = features[col].clip(lower=q_low, upper=q_high)
            dfeatures = xgb.DMatrix(features)
            predicted_price = self.model.predict(dfeatures)[0]
            if np.isnan(predicted_price) or np.isinf(predicted_price):
                print(
                    "ВНИМАНИЕ: Предсказанная цена - NaN или Inf. Используем текущую цену."
                )
                predicted_price = current_price

            price_change = (predicted_price - current_price) / current_price * 100

            if abs(price_change) > 10:
                print(
                    f"ВНИМАНИЕ: Изменение цены ({price_change:.2f}%) слишком большое. Ограничиваем до ±10%."
                )
                price_change = np.sign(price_change) * 10
                predicted_price = current_price * (1 + price_change / 100)
            confidence = 0.0
            return {
                "current_price": current_price,
                "predicted_price": predicted_price,
                "price_change_pct": price_change,
                "signal": "BUY" if price_change > 0 else "SELL",
                "confidence": confidence,
            }
        except Exception as e:
            print(f"Ошибка при прогнозировании: {e}")
            return {
                "current_price": self.last_price
                if hasattr(self, "last_price") and self.last_price is not None
                else 0,
                "predicted_price": self.last_price
                if hasattr(self, "last_price") and self.last_price is not None
                else 0,
                "price_change_pct": 0.0,
                "signal": "NEUTRAL",
                "confidence": 0.0,
            }

    def get_feature_importance(self, top_n=10):
        if self.feature_importances is None:
            print("Модель еще не обучена. Запустите train() сначала.")
            return None
        return self.feature_importances.head(top_n)

    def hyperparameter_tuning(self, X, y, param_grid=None, cv=3):
        from sklearn.model_selection import GridSearchCV

        print("Запуск подбора гиперпараметров XGBoost...")
        if param_grid is None:
            param_grid = {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [50, 100, 200],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "gamma": [0, 0.1, 0.2],
            }
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=cv,
            verbose=1,
            n_jobs=-1,
        )
        try:
            if X.isna().any().any() or np.isinf(X.values).any():
                print("ВНИМАНИЕ: В данных есть NaN или Inf. Выполняем предобработку...")
                X = X.replace([np.inf, -np.inf], np.nan)
                for col in X.columns:
                    if X[col].isna().any():
                        X[col] = X[col].fillna(X[col].median())
            for col in X.columns:
                q_low = X[col].quantile(0.01)
                q_high = X[col].quantile(0.99)
                X[col] = X[col].clip(q_low, q_high)
            zero_var_cols = [col for col in X.columns if X[col].std() < 1e-6]
            if zero_var_cols:
                print(
                    f"ВНИМАНИЕ: Колонки с нулевой дисперсией: {zero_var_cols}. Исключаем."
                )
                X = X.drop(columns=zero_var_cols)
            if X.shape[0] < cv * 2:
                print(
                    f"ВНИМАНИЕ: Мало данных ({X.shape[0]}) для {cv}-fold CV. Уменьшаем cv."
                )
                cv = min(2, X.shape[0] - 1)
                grid_search = GridSearchCV(
                    estimator=xgb_model,
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    cv=cv,
                    verbose=1,
                    n_jobs=-1,
                )
            print(f"Начинаем поиск с {cv}-fold кросс-валидацией...")
            grid_search.fit(X, y)
            print(f"Лучшие параметры: {grid_search.best_params_}")
            print(f"Лучший результат: {-grid_search.best_score_:.4f} (MSE)")
            self.params.update(grid_search.best_params_)
            self.n_estimators = grid_search.best_params_.get(
                "n_estimators", self.n_estimators
            )
            cv_results = pd.DataFrame(grid_search.cv_results_)
            print("\nТоп-3 лучших комбинации параметров:")
            top_indices = cv_results["rank_test_score"].argsort()[:3]
            for i, idx in enumerate(top_indices):
                print(f"{i + 1}. MSE: {-cv_results.loc[idx, 'mean_test_score']:.4f}")
                for param_name in param_grid.keys():
                    param_key = f"param_{param_name}"
                    if param_key in cv_results.columns:
                        print(f"   {param_name}: {cv_results.loc[idx, param_key]}")
            if "max_depth" in param_grid and "learning_rate" in param_grid:
                pivot_table = cv_results.pivot_table(
                    index="param_max_depth",
                    columns="param_learning_rate",
                    values="mean_test_score",
                )
                plt.figure(figsize=(10, 6))
                sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".4f")
                plt.title("MSE по глубине дерева и скорости обучения")
                plt.xlabel("Learning Rate")
                plt.ylabel("Max Depth")
                timestamp_now = datetime.now().strftime("%Y%m%d-%H%M%S")
                plt.savefig(f"hyperparameter_tuning_{timestamp_now}.png")
                print(f"График сохранен в hyperparameter_tuning_{timestamp_now}.png")
                plt.show()
            return grid_search.best_params_
        except Exception as e:
            print(f"Ошибка при подборе гиперпараметров: {e}")
            default_params = {
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
            }
            return default_params


# Основная функция
def main(db_path):
    """
    Main training function using the new XGBoostTradeModelNew.

    This implementation:
    - Uses core modules for proper data handling
    - Avoids data leakage (scaler fitted only on training data)
    - Provides comprehensive output for metrics parsing
    """
    print("Загрузка данных...")
    df = load_data(db_path)

    if 'figi' in df.columns:
        df = df.drop(['figi'], axis=1)

    print(df.tail(3))
    print(
        f"Загружено {len(df)} записей за период с {df['timestamp'].min() if not df.empty else 'N/A'} "
        f"по {df['timestamp'].max() if not df.empty else 'N/A'}"
    )

    if df.empty or len(df) < 50:
        print(f"ОШИБКА: Недостаточно данных для {db_path}. Минимум 50 записей.")
        return None

    print("\nОбучение модели XGBoost...")
    print("ВАЖНО: Scaler обучается ТОЛЬКО на тренировочных данных (без data leakage)")

    model = XGBoostTradeModelNew()

    try:
        metrics = model.train(df)
    except Exception as e:
        print(f"ОШИБКА при обучении: {e}")
        return None

    # Выводим метрики
    print("\n" + "=" * 50)
    print("МЕТРИКИ МОДЕЛИ")
    print("=" * 50)

    print("\nМетрики на тестовой выборке:")
    print(f"MSE: {metrics.get('test_mse', 0):.6f}")
    print(f"RMSE: {metrics.get('test_rmse', 0):.6f}")
    print(f"MAE: {metrics.get('test_mae', 0):.6f}")
    print(f"R²: {metrics.get('test_r2', 0):.6f}")
    print(f"MAPE: {metrics.get('test_mape', 0):.2f}")
    print(f"Direction Accuracy: {metrics.get('test_direction_accuracy', 0):.2f}")

    print("\nМетрики на тренировочной выборке:")
    print(f"MSE: {metrics.get('train_mse', 0):.6f}")
    print(f"RMSE: {metrics.get('train_rmse', 0):.6f}")
    print(f"MAE: {metrics.get('train_mae', 0):.6f}")
    print(f"R²: {metrics.get('train_r2', 0):.6f}")

    # Важность признаков
    feature_imp = model.get_feature_importance()
    if feature_imp is not None:
        print("\nВажность признаков (топ-10):")
        print(feature_imp.head(10).to_string(index=False))

    # Прогноз
    print("\n" + "=" * 50)
    print("ПРОГНОЗ НА СЛЕДУЮЩИЙ ВРЕМЕННОЙ ИНТЕРВАЛ")
    print("=" * 50)

    prediction = model.predict_next(df)

    print(f"Текущая цена: {prediction['current_price']:.4f}")
    print(f"Прогнозируемая цена: {prediction['predicted_price']:.4f}")
    print(f"Ожидаемое изменение: {prediction['expected_change']:.2f}%")
    print(f"Торговый сигнал: {prediction['signal']}")

    # Backtest
    print("\n" + "=" * 50)
    print("РЕТРОСПЕКТИВНАЯ ОЦЕНКА ТОРГОВЫХ СИГНАЛОВ")
    print("=" * 50)

    _run_backtest(model, df)

    return model


def _run_backtest(model: XGBoostTradeModelNew, df: pd.DataFrame):
    """Run simple backtest on test data."""
    df_features = model.prepare_features(df)
    test_size = int(len(df_features) * 0.2)

    if test_size < 2:
        print("Недостаточно данных для бэктеста")
        return

    test_df = df_features.iloc[-test_size:]
    X_test = test_df[model.feature_columns]
    X_test_scaled = model.scaler.transform(X_test)
    predictions = model._predict(X_test_scaled)

    # Drop NaN (last row has no next_close)
    valid_mask = ~np.isnan(test_df['next_close'].values)
    y_test = test_df['next_close'].values[valid_mask]
    current_prices = test_df['close'].values[valid_mask]
    predictions_valid = predictions[valid_mask]

    signals = np.sign(predictions_valid - current_prices)
    actual_returns = np.diff(y_test) / y_test[:-1]
    strategy_returns = signals[:-1] * actual_returns

    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    total_trades = int(np.sum(np.abs(np.diff(signals)) > 0) + 1)
    profitable_trades = int(np.sum(strategy_returns > 0))

    profit_sum = np.sum(strategy_returns[strategy_returns > 0])
    loss_sum = abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

    # Sharpe Ratio (annualized)
    if len(strategy_returns) > 1:
        sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
        equity_curve = (1 + strategy_returns).cumprod()
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        max_drawdown = float(np.max(drawdowns))
    else:
        sharpe_ratio = 0.0
        max_drawdown = 0.0

    print(f"Всего сделок: {total_trades}")
    print(f"Прибыльных сделок: {profitable_trades} ({profitable_trades / len(strategy_returns) * 100:.2f}%)")
    print(f"Общая доходность: {cumulative_returns[-1] * 100:.2f}%")
    print(f"Коэффициент прибыли (Profit Factor): {profit_factor:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Максимальная просадка: {max_drawdown * 100:.2f}%")


if __name__ == "__main__":
    main("BBG000QJW156")
