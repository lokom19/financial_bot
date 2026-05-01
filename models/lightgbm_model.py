import os
import sqlite3
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from utils.load_data_method import load_data

# Import core modules for new implementation
from core.feature_engineering import FeatureSet
from core.base_model import BaseTradeModel

# Отключаем все RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# NEW IMPLEMENTATION: LightGBMTradeModel using core modules
# ============================================================

class LightGBMTradeModelNew(BaseTradeModel):
    """LightGBM model for price prediction with no data leakage."""

    REQUIRED_FEATURES = {FeatureSet.EXTENDED}  # LightGBM uses all features
    MODEL_NAME = "lightgbm_model"

    def __init__(self, params=None, test_size: float = 0.2, random_state: int = 42):
        super().__init__(test_size=test_size, random_state=random_state)
        self.params = params or {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': -1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'verbose': -1
        }
        self.model = None

    def _create_model(self):
        return lgb.LGBMRegressor(**self.params)

    def _fit_model(self, X_train, y_train, X_val, y_val):
        self.model = self._create_model()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        if self.feature_columns:
            self.feature_importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)

    def _predict(self, X):
        return self.model.predict(X)


# Функция для загрузки данных из SQLite
# def load_data(db_path, table_name):
#     conn = sqlite3.connect(db_path)
#     query = f"SELECT * FROM {table_name} ORDER BY timestamp"
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#     return df


# Функция для создания технических индикаторов
def create_features(df):
    # Создаем копию DataFrame
    df_features = df.copy()

    # Простые скользящие средние
    df_features['sma_5'] = df['close'].rolling(window=5).mean()
    df_features['sma_10'] = df['close'].rolling(window=10).mean()
    df_features['sma_20'] = df['close'].rolling(window=20).mean()
    df_features['sma_50'] = df['close'].rolling(window=50).mean()
    df_features['sma_200'] = df['close'].rolling(window=200).mean()

    # Экспоненциальные скользящие средние
    df_features['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df_features['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df_features['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df_features['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df_features['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Разница между ценой закрытия и скользящими средними
    df_features['close_minus_sma_5'] = df['close'] - df_features['sma_5']
    df_features['close_minus_sma_10'] = df['close'] - df_features['sma_10']
    df_features['close_minus_sma_20'] = df['close'] - df_features['sma_20']
    df_features['close_minus_sma_50'] = df['close'] - df_features['sma_50']
    df_features['close_minus_sma_200'] = df['close'] - df_features['sma_200']

    # Относительные разницы
    df_features['close_rel_sma_5'] = df['close'] / df_features['sma_5'] - 1
    df_features['close_rel_sma_10'] = df['close'] / df_features['sma_10'] - 1
    df_features['close_rel_sma_20'] = df['close'] / df_features['sma_20'] - 1
    df_features['close_rel_sma_50'] = df['close'] / df_features['sma_50'] - 1
    df_features['close_rel_sma_200'] = df['close'] / df_features['sma_200'] - 1

    # Индикаторы схождения/расхождения скользящих средних
    df_features['sma_5_10_cross'] = df_features['sma_5'] - df_features['sma_10']
    df_features['sma_10_20_cross'] = df_features['sma_10'] - df_features['sma_20']
    df_features['sma_20_50_cross'] = df_features['sma_20'] - df_features['sma_50']
    df_features['ema_5_10_cross'] = df_features['ema_5'] - df_features['ema_10']
    df_features['ema_10_20_cross'] = df_features['ema_10'] - df_features['ema_20']
    df_features['ema_20_50_cross'] = df_features['ema_20'] - df_features['ema_50']

    # Логарифмированный объём для уменьшения разброса
    df_features['volume_log'] = np.log1p(df['volume'])

    # Объём - индикаторы
    df_features['volume_sma_5'] = df_features['volume_log'].rolling(window=5).mean()
    df_features['volume_sma_10'] = df_features['volume_log'].rolling(window=10).mean()
    df_features['volume_sma_20'] = df_features['volume_log'].rolling(window=20).mean()
    df_features['volume_ratio'] = df_features['volume_log'] / df_features['volume_sma_5']
    df_features['volume_change'] = df_features['volume_log'].pct_change(1)

    # Отношение объема к цене
    df_features['volume_price_ratio'] = df_features['volume_log'] / np.log1p(df['close'])

    # Money Flow Index (MFI) - упрощенная версия
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0))
    negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0))

    # 14-периодный MFI
    positive_mf_14 = positive_flow.rolling(window=14).sum()
    negative_mf_14 = negative_flow.rolling(window=14).sum()

    # Избегаем деление на ноль
    mfi_ratio = np.where(negative_mf_14 != 0, positive_mf_14 / negative_mf_14, 100)
    df_features['mfi_14'] = 100 - (100 / (1 + mfi_ratio))

    # Ценовые изменения
    df_features['price_change_1'] = df['close'].pct_change(periods=1)
    df_features['price_change_2'] = df['close'].pct_change(periods=2)
    df_features['price_change_3'] = df['close'].pct_change(periods=3)
    df_features['price_change_5'] = df['close'].pct_change(periods=5)
    df_features['price_change_10'] = df['close'].pct_change(periods=10)
    df_features['price_change_20'] = df['close'].pct_change(periods=20)

    # Накопленные ценовые изменения
    df_features['cum_return_5'] = (1 + df_features['price_change_1']).rolling(window=5).apply(lambda x: np.prod(x) - 1)
    df_features['cum_return_10'] = (1 + df_features['price_change_1']).rolling(window=10).apply(
        lambda x: np.prod(x) - 1)
    df_features['cum_return_20'] = (1 + df_features['price_change_1']).rolling(window=20).apply(
        lambda x: np.prod(x) - 1)

    # Волатильность
    df_features['volatility_5'] = df['close'].rolling(window=5).std() / df_features['sma_5']
    df_features['volatility_10'] = df['close'].rolling(window=10).std() / df_features['sma_10']
    df_features['volatility_20'] = df['close'].rolling(window=20).std() / df_features['sma_20']
    df_features['volatility_50'] = df['close'].rolling(window=50).std() / df_features['sma_50']

    # Изменения волатильности
    df_features['volatility_change_5'] = df_features['volatility_5'].pct_change(periods=5)
    df_features['volatility_change_10'] = df_features['volatility_10'].pct_change(periods=10)

    # Диапазон High-Low
    df_features['high_low_ratio'] = df['high'] / df['low']
    df_features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    # Диапазон Bollinger Bands
    df_features['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df_features['bb_upper'] = df_features['bb_middle'] + (bb_std * 2)
    df_features['bb_lower'] = df_features['bb_middle'] - (bb_std * 2)
    df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
    df_features['bb_position'] = (df['close'] - df_features['bb_lower']) / (
                df_features['bb_upper'] - df_features['bb_lower'])
    df_features['bb_squeeze'] = bb_std.rolling(window=20).std()

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    # Избегаем деление на ноль
    rs = np.where(loss != 0, gain / loss, 100)
    df_features['rsi_14'] = 100 - (100 / (1 + rs))

    # Изменения RSI
    df_features['rsi_change_5'] = df_features['rsi_14'].diff(periods=5)

    # True Range и ATR
    df_features['prev_close'] = df['close'].shift(1)
    df_features['tr'] = df_features.apply(
        lambda x: max(
            x['high'] - x['low'],
            abs(x['high'] - x['prev_close']),
            abs(x['low'] - x['prev_close'])
        ) if not pd.isna(x['prev_close']) else np.nan,
        axis=1
    )
    df_features['atr_14'] = df_features['tr'].rolling(window=14).mean()
    df_features['atr_ratio'] = df_features['atr_14'] / df['close']

    # MACD (Moving Average Convergence Divergence)
    df_features['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26,
                                                                                               adjust=False).mean()
    df_features['macd_signal'] = df_features['macd_line'].ewm(span=9, adjust=False).mean()
    df_features['macd_histogram'] = df_features['macd_line'] - df_features['macd_signal']
    df_features['macd_change'] = df_features['macd_histogram'].diff()

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df_features['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df_features['stoch_d'] = df_features['stoch_k'].rolling(window=3).mean()
    df_features['stoch_crossover'] = df_features['stoch_k'] - df_features['stoch_d']

    # Rate of Change (ROC)
    df_features['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
    df_features['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
    df_features['roc_20'] = (df['close'] / df['close'].shift(20) - 1) * 100

    # Average Directional Index (ADX)
    plus_dm = df['high'].diff()
    minus_dm = df['low'].shift(1) - df['low']
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(alpha=1 / 14).mean() / df_features['atr_14'])
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(alpha=1 / 14).mean() / df_features['atr_14'])
    df_features['dx'] = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    df_features['adx'] = df_features['dx'].ewm(alpha=1 / 14).mean()
    df_features['adx_trend'] = df_features['adx'].diff()

    # Удаляем временные колонки
    df_features = df_features.drop(['prev_close', 'tr'], axis=1)

    # Создаем целевую переменную - цена закрытия следующего периода
    df_features['next_close'] = df_features['close'].shift(-1)

    # Удаляем строки с NaN значениями
    # df_features = df_features.dropna()
    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    # Проверяем данные на бесконечные значения и заменяем их на NaN
    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # Удаляем строки с оставшимися NaN значениями
    # df_features = df_features.dropna()
    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    return df_features


# Модель для прогнозирования с использованием LightGBM
class LightGBMTradeModel:
    def __init__(self, params=None):
        # Параметры LightGBM по умолчанию
        self.params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': -1,  # -1 означает без ограничения глубины
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_gain_to_split': 0.0,
            'verbose': -1,
            'metric': 'rmse',
            'seed': 42
        }

        self.num_boost_round = 100
        self.early_stopping_rounds = 20

        if params is not None:
            self.params.update(params)

        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.last_price = None
        self.feature_importances = None
        self.data_quality_checked = False
        self.best_iteration = None
        self.train_metrics = None
        self.validation_metrics = None

    def check_data_quality(self, features):
        """Проверка качества данных и выявление потенциальных проблем"""
        # Проверка на корреляцию
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        highly_correlated = [column for column in upper.columns if any(upper[column] > 0.95)]

        if highly_correlated:
            print(f"ВНИМАНИЕ: Обнаружены сильно коррелирующие признаки: {highly_correlated[:5]}")
            print(f"Всего {len(highly_correlated)} коррелирующих признаков")
            print("Это может вызвать нестабильность в модели, но LightGBM обычно справляется с этим.")

        # Проверка на разброс значений и выбросы
        for col in features.columns:
            # Избегаем деления на очень маленькие числа
            mean_value = features[col].mean()
            if abs(mean_value) < 1e-10:  # Почти ноль
                ratio = features[col].std()
            else:
                ratio = features[col].std() / mean_value

            if ratio > 100:
                print(f"ВНИМАНИЕ: Очень высокий разброс значений в колонке {col}. Применяем ограничение выбросов.")
                # Применяем обрезку выбросов по квантилям
                q1 = features[col].quantile(0.01)
                q3 = features[col].quantile(0.99)
                features[col] = features[col].clip(q1, q3)

            # Проверка на нулевые или близкие к нулю дисперсии
            if features[col].std() < 1e-6:
                print(f"ВНИМАНИЕ: Колонка {col} имеет близкую к нулю дисперсию.")

        self.data_quality_checked = True
        return features

    def prepare_data(self, df):
        """Подготовка данных для модели LightGBM"""
        # Сохраняем timestamp для последующего анализа
        timestamps = df['timestamp']

        # Целевая переменная - следующая цена закрытия
        y = df['next_close']

        # Удаляем колонки, которые не нужны для обучения
        features = df.drop(['timestamp', 'next_close'], axis=1)

        # Удаляем 'volume' из признаков, так как мы используем его логарифмированную версию
        if 'volume' in features.columns and 'volume_log' in features.columns:
            features = features.drop(['volume'], axis=1)

        # Проверяем и исправляем качество данных перед обработкой
        if not self.data_quality_checked:
            features = self.check_data_quality(features)

        # Проверка на наличие бесконечных значений
        features = features.replace([np.inf, -np.inf], np.nan)

        # Заполняем отсутствующие значения медианой столбца
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)

        # Проверка на экстремальные значения
        for col in features.columns:
            # Применяем винсоризацию
            q_low = features[col].quantile(0.01)
            q_high = features[col].quantile(0.99)
            features[col] = features[col].clip(lower=q_low, upper=q_high)

        # Удаляем колонки с близкой к нулю дисперсией или бесконечными значениями
        cols_to_drop = []
        for col in features.columns:
            if features[col].std() < 1e-8 or features[col].isna().any():
                cols_to_drop.append(col)
                print(f"Удаляем колонку {col} из-за низкой дисперсии или наличия NaN")

        if cols_to_drop:
            features = features.drop(cols_to_drop, axis=1)
            if len(features.columns) == 0:
                raise ValueError("После удаления проблемных колонок не осталось признаков для модели")

        self.feature_columns = features.columns

        # Выводим базовую статистику для каждого признака после обработки
        print("\nСтатистика признаков после обработки выбросов:")
        print(f"Общее количество признаков: {len(features.columns)}")

        # Статистика только по нескольким признакам
        important_features = ['close', 'sma_20', 'rsi_14', 'macd_line', 'volatility_20']
        available_features = [f for f in important_features if f in features.columns]
        print(features[available_features].describe().loc[['min', 'max', 'mean', 'std']].T)

        # Проверка наличия NaN значений после замены
        if features.isna().any().any():
            print("ВНИМАНИЕ: В данных остались NaN значения после обработки.")
            features = features.fillna(features.median())  # Заполняем оставшиеся NaN медианой

        # Стандартизируем признаки
        try:
            features_scaled = self.scaler.fit_transform(features)
            features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
        except Exception as e:
            print(f"Ошибка при масштабировании данных: {e}")
            print("Применяем более простое масштабирование...")

            # Применяем более простое масштабирование
            features_scaled = np.zeros(features.shape)
            for i, col in enumerate(features.columns):
                col_mean = features[col].mean()
                col_std = features[col].std()
                if col_std > 1e-10:
                    features_scaled[:, i] = (features[col] - col_mean) / col_std
                else:
                    features_scaled[:, i] = 0

            features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

        # Проверка на NaN и Inf после масштабирования и замена проблемных значений
        if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
            print("ВНИМАНИЕ: После масштабирования обнаружены NaN или Inf значения. Заменяем их на 0.")
            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

        # Дополнительная проверка диапазона значений после масштабирования
        if np.max(np.abs(features_scaled)) > 1e6:
            print("ВНИМАНИЕ: Обнаружены очень большие значения после масштабирования. Ограничиваем их.")
            features_scaled = np.clip(features_scaled, -1e6, 1e6)
            features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

        return features, features_scaled_df, y, timestamps

    def train(self, X, y, timestamps=None, X_scaled=None, df_features=None):
        # Проверка на NaN и Inf перед обучением
        if X.isna().any().any() or np.isinf(X.values).any():
            print("Обнаружены NaN или Inf. Заменяем на медианные значения.")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        # Разделение на обучающую, валидационную и тестовую выборки
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]

        X_val = X.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]

        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]

        # Создание наборов данных LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Ведение журнала прогресса
        evals_result = {}

        # Обучение с мониторингом
        try:
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                    lgb.log_evaluation(100),
                    lgb.record_evaluation(evals_result)
                ]
            )

            self.best_iteration = self.model.best_iteration
            print(f"Лучшая итерация: {self.best_iteration}")

        except Exception as e:
            print(f"Ошибка при обучении модели: {e}")
            print("Попытка перезапуска с модифицированными параметрами...")

            # Безопасные параметры
            safe_params = {
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'num_leaves': 15,
                'max_depth': 5,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'metric': 'rmse'
            }

            self.model = lgb.train(
                safe_params,
                train_data,
                num_boost_round=50,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(10, verbose=True),
                    lgb.log_evaluation(100)
                ]
            )

            self.best_iteration = self.model.best_iteration

        # Предсказания
        train_preds = self.model.predict(X_train, num_iteration=self.best_iteration)
        val_preds = self.model.predict(X_val, num_iteration=self.best_iteration)
        test_preds = self.model.predict(X_test, num_iteration=self.best_iteration)

        # Проверка на NaN/Inf
        for preds, name in [(train_preds, "обучающей"), (val_preds, "валидационной"), (test_preds, "тестовой")]:
            if np.isnan(preds).any() or np.isinf(preds).any():
                print(f"ВНИМАНИЕ: В предсказаниях на {name} выборке есть NaN/Inf.")
                preds = np.nan_to_num(preds)

        # Сохранение важности признаков
        self.feature_importances = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('Importance', ascending=False)

        # Вычисление метрик
        self.train_metrics = self.calculate_metrics(y_train, train_preds)
        self.validation_metrics = self.calculate_metrics(y_val, val_preds)
        self.test_metrics = self.calculate_metrics(y_test, test_preds)

        print("\n===== Метрики на обучающей выборке =====")
        self.print_metrics(self.train_metrics)

        print("\n===== Метрики на валидационной выборке =====")
        self.print_metrics(self.validation_metrics)

        print("\n===== Метрики на тестовой выборке =====")
        self.print_metrics(self.test_metrics)

        # Вывод кривых обучения
        if evals_result and 'valid' in evals_result:
            print("\nЭволюция ошибки по итерациям:")
            for metric_name in evals_result['valid']:
                print(f"{metric_name}: начальное={evals_result['valid'][metric_name][0]:.4f}, "
                      f"лучшее={min(evals_result['valid'][metric_name]):.4f}")

        # Сохраняем последнюю цену для будущих прогнозов
        # self.last_price = y.iloc[-1]
        if np.isnan(y.iloc[-1]) and df_features is not None:
            self.last_price = df_features.iloc[-1]['close']
        else:
            self.last_price = y.iloc[-1]
        return X_test, y_test, test_preds, self.test_metrics


    def calculate_metrics(self, y_true, y_pred):
        """Расчет метрик качества прогнозирования"""
        # Обработка безопасных вычислений метрик
        try:
            # Фильтруем NaN значения перед вычислением метрик
            mask = ~np.isnan(y_true)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            if len(y_true_filtered) == 0:
                return {
                    'MSE': float('nan'),
                    'RMSE': float('nan'),
                    'MAE': float('nan'),
                    'R²': float('nan'),
                    'MAPE': float('nan'),
                    'Direction Accuracy': float('nan')
                }

            mse = mean_squared_error(y_true_filtered, y_pred_filtered)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_filtered, y_pred_filtered)

            # Безопасное вычисление R²
            try:
                r2 = r2_score(y_true_filtered, y_pred_filtered)
            except Exception as e:
                print(f"Ошибка при вычислении R²: {e}")
                r2 = float('nan')

            # Безопасное вычисление MAPE
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_raw = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)
                mape_raw = np.where(np.isinf(mape_raw) | np.isnan(mape_raw), 0, mape_raw)
                mape = np.mean(mape_raw) * 100

            # Направление цены (бинарный классификатор)
            if len(y_true_filtered) > 1:
                direction_true = np.diff(np.append(y_true_filtered.iloc[0], y_true_filtered.values)) > 0
                direction_pred = np.diff(np.append(y_true_filtered.iloc[0], y_pred_filtered)) > 0
                direction_accuracy = np.mean(direction_true == direction_pred) * 100
            else:
                direction_accuracy = float('nan')

            return {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape,
                'Direction Accuracy': direction_accuracy
            }
        except Exception as e:
            print(f"Ошибка при вычислении метрик: {e}")
            # Возвращаем NaN для всех метрик в случае ошибки
            return {
                'MSE': float('nan'),
                'RMSE': float('nan'),
                'MAE': float('nan'),
                'R²': float('nan'),
                'MAPE': float('nan'),
                'Direction Accuracy': float('nan')
            }

    def print_metrics(self, metrics):
        """Вывод метрик в консоль"""
        for name, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                print(f"{name}: Ошибка вычисления")
            else:
                print(f"{name}: {value:.4f}")

    def predict_next(self, current_data):
        """Предсказание следующей цены закрытия"""
        try:
            current_price = current_data['close'].iloc[0]

            # Проверяем, содержит ли current_data все необходимые колонки
            missing_cols = []
            for col in self.feature_columns:
                if col not in current_data.columns:
                    if col == 'volume_log' and 'volume' in current_data.columns:
                        # Создаем логарифмированную версию, если она отсутствует
                        current_data['volume_log'] = np.log1p(current_data['volume'])
                    else:
                        missing_cols.append(col)

            if missing_cols:
                print(f"ВНИМАНИЕ: В данных отсутствуют колонки: {missing_cols}")
                # Создаем отсутствующие колонки с нулями
                for col in missing_cols:
                    current_data[col] = 0

            # Подготавливаем данные
            features = current_data[self.feature_columns].copy()  # тут последняя строка из датасета, где есть next_close

            # Заменяем бесконечные значения на NaN и затем заполняем медианами
            features = features.replace([np.inf, -np.inf], np.nan)
            for col in features.columns:
                if features[col].isna().any():
                    features[col] = features[col].fillna(features[col].median())

            # Применяем ограничение выбросов перед масштабированием
            for col in features.columns:
                q_low = features[col].quantile(0.01) if len(features) > 10 else features[col].min()
                q_high = features[col].quantile(0.99) if len(features) > 10 else features[col].max()
                features[col] = features[col].clip(lower=q_low, upper=q_high)

            # Масштабируем данные
            try:
                scaled_features = self.scaler.transform(features)
            except Exception as e:
                print(f"Ошибка при масштабировании данных для прогноза: {e}")
                # Ручное масштабирование
                scaled_features = np.zeros(features.shape)
                for i, col in enumerate(features.columns):
                    if hasattr(self.scaler, 'center_') and hasattr(self.scaler, 'scale_'):
                        center = self.scaler.center_[i] if i < len(self.scaler.center_) else 0
                        scale = self.scaler.scale_[i] if i < len(self.scaler.scale_) else 1
                        if scale > 1e-10:
                            scaled_features[:, i] = (features[col].values - center) / scale
                        else:
                            scaled_features[:, i] = 0
                    else:
                        col_mean = features[col].mean()
                        col_std = features[col].std()
                        if col_std > 1e-10:
                            scaled_features[:, i] = (features[col] - col_mean) / col_std
                        else:
                            scaled_features[:, i] = 0

            # Проверка на NaN и Inf после масштабирования
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                print("ВНИМАНИЕ: После масштабирования обнаружены NaN или Inf значения. Заменяем их на 0.")
                scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Ограничиваем масштабированные значения
            scaled_features = np.clip(scaled_features, -1e6, 1e6)  # это просто наши данные последнего дня с изменением масштаба

            # Делаем прогноз
            predicted_price = self.model.predict(features, num_iteration=self.best_iteration)[0]

            # Проверяем, не является ли предсказание NaN или Inf
            if np.isnan(predicted_price) or np.isinf(predicted_price):
                print("ВНИМАНИЕ: Предсказанная цена - NaN или Inf. Используем текущую цену.")
                predicted_price = current_price

            # Определяем ожидаемое изменение цены
            price_change = (predicted_price - current_price) / current_price * 100

            # Ограничиваем слишком большие изменения цены
            if abs(price_change) > 10:
                print(f"ВНИМАНИЕ: Очень большое изменение цены ({price_change:.2f}%). Ограничиваем до ±10%.")
                price_change = np.sign(price_change) * 10
                predicted_price = current_price * (1 + price_change / 100)

            # Определение уровня доверия к прогнозу
            # Используем метрику Direction Accuracy с тестовой выборки как индикатор
            if self.validation_metrics and 'Direction Accuracy' in self.validation_metrics:
                base_confidence = self.validation_metrics['Direction Accuracy'] / 100
            else:
                base_confidence = 0.5

            # Корректируем уверенность в зависимости от размера изменения цены
            # Чем больше изменение, тем меньше уверенность
            confidence_adjustment = max(0, 1 - abs(price_change) / 20)
            confidence = base_confidence * confidence_adjustment

            # Создаем торговый сигнал на основе предсказания
            if price_change > 2:
                signal = 'BUY'
            elif price_change < -2:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change,
                'signal': signal,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Ошибка при прогнозировании следующей цены: {e}")
            # В случае ошибки возвращаем нейтральный прогноз
            return {
                'current_price': self.last_price if hasattr(self, 'last_price') and self.last_price is not None else 0,
                'predicted_price': self.last_price if hasattr(self,
                                                              'last_price') and self.last_price is not None else 0,
                'price_change_pct': 0.0,
                'signal': 'NEUTRAL',
                'confidence': 0.0
            }

    def get_feature_importance(self, top_n=10):
        """Получение важности признаков"""
        if self.feature_importances is None:
            print("Модель еще не обучена. Запустите метод train() сначала.")
            return None

        return self.feature_importances.head(top_n)

    def hyperparameter_tuning(self, X, y, param_grid=None, cv=3):
        """Подбор гиперпараметров для LightGBM с использованием кросс-валидации"""
        from sklearn.model_selection import GridSearchCV
        import lightgbm as lgb

        print("Запуск подбора гиперпараметров LightGBM...")

        # Если не предоставлена сетка параметров, используем стандартную
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.01, 0.03, 0.1],
                'num_leaves': [15, 31, 63],
                'max_depth': [5, -1],  # -1 означает без ограничения глубины
                'min_data_in_leaf': [10, 20, 50],
                'feature_fraction': [0.7, 0.8, 0.9],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'lambda_l1': [0.0, 0.1, 1.0],
                'lambda_l2': [0.0, 0.1, 1.0]
            }

        # Создаем базовую модель LightGBM
        estimator = lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            verbose=-1,
            metric='rmse',
            importance_type='gain',
            n_jobs=-1
        )

        # Настраиваем поиск по сетке
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=cv,
            verbose=1,
            n_jobs=-1
        )

        try:
            # Проверяем данные на наличие NaN и Inf
            if X.isna().any().any() or np.isinf(X.values).any():
                print("ВНИМАНИЕ: В данных есть NaN или Inf. Выполняем предобработку...")
                X = X.replace([np.inf, -np.inf], np.nan)
                for col in X.columns:
                    if X[col].isna().any():
                        X[col] = X[col].fillna(X[col].median())

            # Применяем ограничение выбросов
            for col in X.columns:
                q_low = X[col].quantile(0.01)
                q_high = X[col].quantile(0.99)
                X[col] = X[col].clip(q_low, q_high)

            # Проверяем на колонки с нулевой дисперсией
            zero_var_cols = [col for col in X.columns if X[col].std() < 1e-6]
            if zero_var_cols:
                print(f"ВНИМАНИЕ: Колонки с близкой к нулевой дисперсией: {zero_var_cols}. Исключаем их из анализа.")
                X = X.drop(columns=zero_var_cols)

            # Проверяем, достаточно ли данных для заданного CV
            if X.shape[0] < cv * 2:
                old_cv = cv
                cv = min(3, X.shape[0] // 2)
                print(f"ВНИМАНИЕ: Недостаточно данных ({X.shape[0]}) для {old_cv}-fold CV. Устанавливаем cv={cv}.")
                grid_search.cv = cv

            print(f"Начинаем поиск гиперпараметров с {cv}-fold кросс-валидацией...")
            grid_search.fit(X, y)

            best_params = grid_search.best_params_
            print(f"Лучшие параметры: {best_params}")
            print(f"Лучший результат: {-grid_search.best_score_:.4f} (MSE)")

            # Обновляем параметры модели
            self.params.update(best_params)

            # Выводим топ-3 лучших комбинаций параметров
            results_df = pd.DataFrame(grid_search.cv_results_)
            results_df['mean_score'] = -results_df['mean_test_score']
            results_df = results_df.sort_values('mean_score', ascending=True)

            print("\nТоп-3 лучших комбинации параметров:")
            for i in range(min(3, len(results_df))):
                print(f"{i + 1}. MSE: {results_df.iloc[i]['mean_score']:.4f}")
                for param_name in param_grid.keys():
                    param_key = f'param_{param_name}'
                    print(f"   {param_name}: {results_df.iloc[i][param_key]}")

            return best_params

        except Exception as e:
            print(f"Ошибка при подборе гиперпараметров: {e}")
            # Возвращаем параметры по умолчанию в случае ошибки
            default_params = {
                'learning_rate': 0.03,
                'num_leaves': 31,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1
            }
            return default_params


# Основная функция
def main(db_path):
    """
    Main training function using the new LightGBMTradeModelNew.

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

    print("\nОбучение модели LightGBM...")
    print("ВАЖНО: Scaler обучается ТОЛЬКО на тренировочных данных (без data leakage)")

    model = LightGBMTradeModelNew()

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


def _run_backtest(model: LightGBMTradeModelNew, df: pd.DataFrame):
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
    main("BBG000F6YPH8")