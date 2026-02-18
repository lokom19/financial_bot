import os
import sys

import pandas as pd
import numpy as np
import sqlite3
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from utils.load_data_method import load_data
from utils.load_crypto_data import load_data as load_crypto_data

# Import core modules
from core.feature_engineering import FeatureSet
from core.base_model import BaseTradeModel

# Отключаем все RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# NEW IMPLEMENTATION: CatBoostTradeModel using core modules
# ============================================================

class CatBoostTradeModelNew(BaseTradeModel):
    """CatBoost model for price prediction with no data leakage."""

    REQUIRED_FEATURES = {FeatureSet.BASIC, FeatureSet.VOLUME, FeatureSet.VOLATILITY, FeatureSet.MOMENTUM}
    MODEL_NAME = "cat_boost_model"

    def __init__(self, params=None, test_size: float = 0.2, random_state: int = 42):
        super().__init__(test_size=test_size, random_state=random_state)
        self.params = params or {
            'iterations': 300,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'RMSE',
            'random_seed': random_state,
            'verbose': False
        }
        self.model = None

    def _create_model(self):
        return CatBoostRegressor(**self.params)

    def _fit_model(self, X_train, y_train, X_val, y_val):
        self.model = self._create_model()
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        if self.feature_columns:
            self.feature_importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)

    def _predict(self, X):
        return self.model.predict(X)


# Функция для создания технических индикаторов
def create_features(df):
    # Создаем копию DataFrame
    df_features = df.copy()

    # Простые скользящие средние
    df_features['sma_5'] = df['close'].rolling(window=5).mean()
    df_features['sma_10'] = df['close'].rolling(window=10).mean()
    df_features['sma_20'] = df['close'].rolling(window=20).mean()

    # Экспоненциальные скользящие средние (менее чувствительны к выбросам)
    df_features['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df_features['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df_features['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Разница между ценой закрытия и скользящими средними
    df_features['close_minus_sma_5'] = df['close'] - df_features['sma_5']
    df_features['close_minus_sma_10'] = df['close'] - df_features['sma_10']

    # Относительные разницы (более стабильные)
    df_features['close_rel_sma_5'] = df['close'] / df_features['sma_5'] - 1
    df_features['close_rel_sma_10'] = df['close'] / df_features['sma_10'] - 1

    # Обработка объёма для уменьшения разброса (логарифмирование)
    df_features['volume_log'] = np.log1p(df['volume'])  # log(1+x) для избежания log(0)

    # Объём - простые индикаторы с логарифмированием
    df_features['volume_sma_5'] = df_features['volume_log'].rolling(window=5).mean()
    df_features['volume_ratio'] = df_features['volume_log'] / df_features['volume_sma_5']

    # Ценовые изменения
    df_features['price_change_1'] = df['close'].pct_change(periods=1)
    df_features['price_change_3'] = df['close'].pct_change(periods=3)
    df_features['price_change_5'] = df['close'].pct_change(periods=5)

    # Волатильность
    df_features['volatility_5'] = df['close'].rolling(window=5).std() / df_features['sma_5']
    df_features['volatility_10'] = df['close'].rolling(window=10).std() / df_features['sma_10']

    # Диапазон High-Low
    df_features['high_low_ratio'] = df['high'] / df['low']

    # True Range - классический индикатор волатильности
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

    # Удаляем временные колонки
    df_features = df_features.drop(['prev_close', 'tr'], axis=1)

    # Создаем целевую переменную - цена закрытия следующего периода
    df_features['next_close'] = df_features['close'].shift(-1)

    # Заполняем NaN значения
    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    # Проверяем данные на бесконечные значения и заменяем их на NaN
    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # Заполняем оставшиеся NaN значения
    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    return df_features


# Модель для прогнозирования с использованием CatBoost
class CatBoostTradeModel:
    def __init__(self, iterations=1000, learning_rate=0.1, depth=6, l2_leaf_reg=3.0):
        # Используем CatBoost вместо Ridge регрессии
        self.model = CatBoostRegressor(
            iterations=500,           # Уменьшено с 1500
            learning_rate=0.05,       # Уменьшено с 0.03
            depth=6,                  # Уменьшено с 12
            l2_leaf_reg=10.0,         # Увеличено с 3.0 (больше регуляризации)
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            # Дополнительные параметры для предотвращения переобучения
            min_data_in_leaf=20,      # Минимум образцов в листе
            max_leaves=64,            # Ограничение количества листьев
            subsample=0.8,            # Случайная выборка данных
            colsample_bylevel=0.8,    # Случайная выборка признаков
            bootstrap_type='Bernoulli' # Тип бутстрэпа
        )

        # Оставляем скалер для нормализации данных
        self.scaler = RobustScaler()

        self.feature_columns = None
        self.last_price = None
        self.feature_importances = None

        # Добавляем флаг для проверки качества данных
        self.data_quality_checked = False

    def check_data_quality(self, features):
        """Проверка качества данных и выявление потенциальных проблем"""
        # Проверка на корреляцию
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        highly_correlated = [column for column in upper.columns if any(upper[column] > 0.95)]

        if highly_correlated:
            print(f"ВНИМАНИЕ: Обнаружены сильно коррелирующие признаки: {highly_correlated}")
            print("CatBoost хорошо справляется с коррелирующими признаками, но рассмотрите удаление некоторых из них.")

        # Проверка на разброс значений и выбросы
        for col in features.columns:
            if features[col].std() / (features[col].mean() + 1e-10) > 100:
                print(
                    f"ВНИМАНИЕ: Очень высокий разброс значений в колонке {col}. Будет применено ограничение выбросов.")
                # Применяем обрезку выбросов по квантилям
                q1 = features[col].quantile(0.01)
                q3 = features[col].quantile(0.99)
                features[col] = features[col].clip(q1, q3)

            # Проверка на нулевые или близкие к нулю дисперсии
            if features[col].std() < 1e-6:
                print(f"ВНИМАНИЕ: Колонка {col} имеет близкую к нулю дисперсию. Рассмотрите её удаление.")

        self.data_quality_checked = True
        return features

    def prepare_data(self, df):
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

        # Заполняем отсутствующие значения медианой столбца вместо удаления строк
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)

        # Проверка на очень большие или малые значения, которые могут вызвать проблемы
        for col in features.columns:
            # Если в колонке есть экстремальные значения, применяем винсоризацию
            q_low = features[col].quantile(0.01)
            q_high = features[col].quantile(0.99)
            features[col] = features[col].clip(lower=q_low, upper=q_high)

        # Удаляем колонки с близкой к нулю дисперсией
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
        print(features.describe().loc[['min', 'max', 'mean', 'std']].T.head())

        # Проверка наличия NaN значений после замены
        if features.isna().any().any():
            print("ВНИМАНИЕ: В данных остались NaN значения после обработки.")
            features = features.fillna(features.median())

        # CatBoost может работать с сырыми данными, но мы всё же применим масштабирование
        # для стабильности предсказаний
        try:
            X = self.scaler.fit_transform(features)
        except Exception as e:
            print(f"Ошибка при масштабировании данных: {e}")
            print("Используем сырые данные без масштабирования...")
            X = features.values

        # Проверка на NaN и Inf после масштабирования
        if np.isnan(X).any() or np.isinf(X).any():
            print("ВНИМАНИЕ: После масштабирования обнаружены NaN или Inf значения. Заменяем их на 0.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, timestamps

    def train(self, X, y, timestamps=None, df_features=None):
        # Проверка на NaN и Inf перед обучением
        if np.isnan(X).any() or np.isinf(X).any():
            print("КРИТИЧЕСКАЯ ОШИБКА: В данных для обучения обнаружены NaN или Inf значения.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Проверка целевой переменной
        if np.isnan(y).any() or np.isinf(y).any():
            print("КРИТИЧЕСКАЯ ОШИБКА: В целевой переменной обнаружены NaN или Inf значения.")
            # Фильтруем NaN значения из целевой переменной
            valid_mask = ~(np.isnan(y) | np.isinf(y))
            X = X[valid_mask]
            y = y[valid_mask]
            if timestamps is not None:
                timestamps = timestamps[valid_mask]

        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Если timestamps предоставлены, разделим их тоже
        if timestamps is not None:
            _, timestamps_test = train_test_split(
                timestamps, test_size=0.2, random_state=42, shuffle=False
            )

        # Обучаем модель CatBoost
        try:
            print("Обучение CatBoost модели...")

            # Создаем валидационную выборку из тренировочных данных
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, shuffle=False
            )

            # Обучаем с валидацией для избежания переобучения
            self.model.fit(
                X_train_fit, y_train_fit,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )

        except Exception as e:
            print(f"Ошибка при обучении CatBoost модели: {e}")
            print("Попытка обучения без валидации...")

            # Простое обучение без валидации
            self.model = CatBoostRegressor(
                iterations=500,
                learning_rate=0.1,
                depth=4,
                l2_leaf_reg=5.0,
                loss_function='RMSE',
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )
            self.model.fit(X_train, y_train)

        # Оцениваем модель
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        # Проверяем, нет ли NaN или Inf в предсказаниях
        if np.isnan(train_preds).any() or np.isinf(train_preds).any():
            print("ВНИМАНИЕ: В предсказаниях на обучающей выборке есть NaN или Inf значения!")
            train_preds = np.nan_to_num(train_preds)

        if np.isnan(test_preds).any() or np.isinf(test_preds).any():
            print("ВНИМАНИЕ: В предсказаниях на тестовой выборке есть NaN или Inf значения!")
            test_preds = np.nan_to_num(test_preds)

        # Сохраняем важность признаков из CatBoost
        try:
            feature_importance = self.model.get_feature_importance()
            self.feature_importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
        except Exception as e:
            print(f"Ошибка при получении важности признаков: {e}")
            self.feature_importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': [0] * len(self.feature_columns)
            })

        # Вычисляем метрики
        train_metrics = self.calculate_metrics(y_train, train_preds)
        test_metrics = self.calculate_metrics(y_test, test_preds)

        print("\n===== Метрики на обучающей выборке =====")
        self.print_metrics(train_metrics)

        print("\n===== Метрики на тестовой выборке =====")
        self.print_metrics(test_metrics)

        # Сохраняем последнюю цену для будущих прогнозов
        if np.isnan(y.iloc[-1]) and df_features is not None:
            self.last_price = df_features.iloc[-1]['close']
        else:
            self.last_price = y.iloc[-1]

        return X_test, y_test, test_preds, test_metrics

    def calculate_metrics(self, y_true, y_pred):
        """Вычисление метрик качества модели"""
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
            return {
                'MSE': float('nan'),
                'RMSE': float('nan'),
                'MAE': float('nan'),
                'R²': float('nan'),
                'MAPE': float('nan'),
                'Direction Accuracy': float('nan')
            }

    def print_metrics(self, metrics):
        """Вывод метрик на экран"""
        for name, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                print(f"{name}: Ошибка вычисления")
            else:
                print(f"{name}: {value:.4f}")

    def visualize_predictions(self, timestamps, y_true, y_pred):
        """Визуализация предсказаний"""
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, y_true, label='Фактическая цена', color='blue')
        plt.plot(timestamps, y_pred, label='Предсказанная цена', color='red', linestyle='--')
        plt.title('Сравнение фактических и предсказанных цен (CatBoost)')
        plt.xlabel('Время')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)

        # Сохраняем график
        timestamp_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(f'catboost_predictions_{timestamp_now}.png')
        print(f"График сохранен в файл catboost_predictions_{timestamp_now}.png")

        # Показываем график
        plt.show()

    def predict_next(self, current_data):
        """Предсказание следующей цены"""
        try:
            # Получаем текущую цену закрытия из данных
            current_price = current_data['close'].iloc[0]

            # Проверяем, содержит ли current_data все необходимые колонки
            missing_cols = []
            for col in self.feature_columns:
                if col not in current_data.columns:
                    if col == 'volume_log' and 'volume' in current_data.columns:
                        current_data['volume_log'] = np.log1p(current_data['volume'])
                    else:
                        missing_cols.append(col)

            if missing_cols:
                print(f"ВНИМАНИЕ: В данных отсутствуют колонки: {missing_cols}")
                for col in missing_cols:
                    current_data[col] = 0

            # Подготавливаем данные
            features = current_data[self.feature_columns].copy()

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
                print("Используем сырые данные...")
                scaled_features = features.values

            # Проверка на NaN и Inf после масштабирования
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                print("ВНИМАНИЕ: После масштабирования обнаружены NaN или Inf значения. Заменяем их на 0.")
                scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Делаем прогноз с помощью CatBoost
            predicted_price = self.model.predict(scaled_features)[0]

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

            # Определяем доверительность прогноза на основе важности признаков
            confidence = min(np.sum(self.feature_importances['Importance'][:5]) / 100, 1.0)

            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change,
                'signal': 'BUY' if price_change > 0 else 'SELL',
                'confidence': confidence
            }
        except Exception as e:
            print(f"Ошибка при прогнозировании следующей цены: {e}")
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


# Основная функция
def main(db_path):
    """
    Main training function using the new CatBoostTradeModelNew.

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

    print("\nОбучение модели CatBoost...")
    print("ВАЖНО: Scaler обучается ТОЛЬКО на тренировочных данных (без data leakage)")

    model = CatBoostTradeModelNew(params={
        'iterations': 300,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'RMSE',
        'verbose': False
    })

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


def _run_backtest(model: CatBoostTradeModelNew, df: pd.DataFrame):
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

    y_test = test_df['next_close'].values
    current_prices = test_df['close'].values
    signals = np.sign(predictions - current_prices)
    actual_returns = np.diff(y_test) / y_test[:-1]
    strategy_returns = signals[:-1] * actual_returns

    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    total_trades = np.sum(np.abs(np.diff(signals)) > 0) + 1
    profitable_trades = np.sum(strategy_returns > 0)

    profit_sum = np.sum(strategy_returns[strategy_returns > 0])
    loss_sum = abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

    print(f"Всего сделок: {total_trades}")
    print(f"Прибыльных сделок: {profitable_trades} ({profitable_trades / total_trades * 100:.2f}%)")
    print(f"Общая доходность: {cumulative_returns[-1] * 100:.2f}%")
    print(f"Коэффициент прибыли (Profit Factor): {profit_factor:.2f}")


if __name__ == "__main__":
    main("BBG0029SG1C1")
    # main("ETHUSDT_250328_5M")
