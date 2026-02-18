import os
import sqlite3
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from utils.load_data_method import load_data
from utils.load_crypto_data import load_data as load_crypto_data

# Import core modules
from core.feature_engineering import create_features as core_create_features, FeatureSet

# Отключаем предупреждения
warnings.filterwarnings("ignore")


# ============================================================
# NEW IMPLEMENTATION: RandomForestClassifierNew
# This fixes data leakage by fitting scaler only on training data
# ============================================================

class RandomForestClassifierNew:
    """
    Random Forest Classifier for direction prediction with no data leakage.

    Key fix: Scaler is fitted ONLY on training data, not the entire dataset.
    """

    MODEL_NAME = "rf_classifier"
    REQUIRED_FEATURES = {FeatureSet.BASIC, FeatureSet.VOLUME, FeatureSet.VOLATILITY, FeatureSet.MOMENTUM}

    def __init__(self, params=None, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

        self.default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': random_state,
            'class_weight': 'balanced',
            'n_jobs': -1
        }

        self.params = self.default_params.copy()
        if params:
            self.params.update(params)

        self.model = RandomForestClassifier(**self.params)
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.is_fitted = False
        self.feature_importances = None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features using core module."""
        return core_create_features(df, feature_sets=self.REQUIRED_FEATURES)

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the classifier with proper data handling.

        IMPORTANT: Scaler is fitted ONLY on training data to prevent data leakage.
        """
        # Create features
        df_features = self.prepare_features(df)

        # Create target variable (1 if next close > current close)
        df_features['price_up'] = (df_features['next_close'] > df_features['close']).astype(int)

        # Prepare feature matrix
        exclude_cols = ['timestamp', 'price_up', 'next_close', 'volume']
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]
        self.feature_columns = feature_cols

        X = df_features[feature_cols].copy()
        y = df_features['price_up']

        # Handle infinities and NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Clip outliers
        for col in X.columns:
            q_low, q_high = X[col].quantile([0.01, 0.99])
            X[col] = X[col].clip(lower=q_low, upper=q_high)

        # Split data BEFORE scaling (critical for no data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )

        # Fit scaler ONLY on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Extract feature importances
        self.feature_importances = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.5
        }

        # Store for backtest
        self._last_y_test = y_test
        self._last_y_pred = y_test_pred
        self._last_y_proba = y_test_proba
        self._last_prices = df_features['close'].iloc[-len(y_test):].values

        return metrics

    def predict_next(self, df: pd.DataFrame) -> dict:
        """Predict direction for the next period."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        df_features = self.prepare_features(df)
        X = df_features[self.feature_columns].iloc[[-1]].copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        X_scaled = self.scaler.transform(X)
        prob_up = self.model.predict_proba(X_scaled)[0, 1]
        prob_down = 1 - prob_up

        direction = "UP" if prob_up > 0.5 else "DOWN"
        confidence = max(prob_up, prob_down)

        # Signal based on confidence threshold
        if confidence > 0.6:
            signal = "BUY" if direction == "UP" else "SELL"
        else:
            signal = "HOLD"

        return {
            'current_price': float(df['close'].iloc[-1]),
            'direction': direction,
            'probability_up': prob_up,
            'probability_down': prob_down,
            'confidence': confidence,
            'signal': signal
        }

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get top N important features."""
        if self.feature_importances is None:
            return None
        return self.feature_importances.head(top_n)


# ============================================================
# LEGACY IMPLEMENTATION: Kept for backward compatibility
# ============================================================

# Функция для создания технических индикаторов
def create_features(df):
    df_features = df.copy()

    # Базовые технические индикаторы
    df_features['sma_5'] = df['close'].rolling(window=5).mean()
    df_features['sma_10'] = df['close'].rolling(window=10).mean()
    df_features['sma_20'] = df['close'].rolling(window=20).mean()
    df_features['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df_features['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df_features['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Разница между ценой и скользящими средними
    df_features['close_minus_sma_5'] = df['close'] - df_features['sma_5']
    df_features['close_minus_sma_10'] = df['close'] - df_features['sma_10']
    df_features['close_rel_sma_5'] = df['close'] / df_features['sma_5'] - 1
    df_features['close_rel_sma_10'] = df['close'] / df_features['sma_10'] - 1

    # Объем
    df_features['volume_log'] = np.log1p(df['volume'])
    df_features['volume_sma_5'] = df_features['volume_log'].rolling(window=5).mean()
    df_features['volume_ratio'] = df_features['volume_log'] / df_features['volume_sma_5']

    # Ценовые изменения
    df_features['price_change_1'] = df['close'].pct_change(periods=1)
    df_features['price_change_3'] = df['close'].pct_change(periods=3)
    df_features['price_change_5'] = df['close'].pct_change(periods=5)

    # Волатильность
    df_features['volatility_5'] = df['close'].rolling(window=5).std() / df_features['sma_5']
    df_features['volatility_10'] = df['close'].rolling(window=10).std() / df_features['sma_10']

    # High-Low диапазон
    df_features['high_low_ratio'] = df['high'] / df['low']

    # True Range для расчета ATR
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

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)  # Избегаем деления на ноль
    rs = rs.fillna(100)  # Заполняем NaN значения
    df_features['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df_features['macd'] = ema_12 - ema_26
    df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
    df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']

    # Bollinger Bands
    df_features['bb_middle'] = df['close'].rolling(window=20).mean()
    df_features['bb_std'] = df['close'].rolling(window=20).std()
    df_features['bb_upper'] = df_features['bb_middle'] + (df_features['bb_std'] * 2)
    df_features['bb_lower'] = df_features['bb_middle'] - (df_features['bb_std'] * 2)
    df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
    df_features['bb_pct'] = (df['close'] - df_features['bb_lower']) / (
                df_features['bb_upper'] - df_features['bb_lower'])

    # Создание целевой переменной - цена закрытия следующего дня
    df_features['next_close'] = df_features['close'].shift(-1)

    # Создание бинарной целевой переменной (1 - цена выросла, 0 - цена упала или не изменилась)
    df_features['price_up'] = (df_features['next_close'] > df_features['close']).astype(int)

    # Удаляем временные колонки
    df_features = df_features.drop(['prev_close', 'tr'], axis=1)

    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    # Удаляем строки с NaN значениями и бесконечностями
    # df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()

    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    return df_features


# Модель Random Forest для бинарной классификации направления цены
class RandomForestDirectionModel:
    def __init__(self, params=None):
        # Параметры по умолчанию
        self.default_params = {
            'n_estimators': 100,  # Количество деревьев
            'max_depth': None,  # Максимальная глубина деревьев
            'min_samples_split': 2,  # Минимальное количество образцов для разделения
            'min_samples_leaf': 1,  # Минимальное количество образцов в листе
            'max_features': 'sqrt',  # Количество признаков для поиска наилучшего разделения
            'bootstrap': True,  # Использовать бутстрап
            'random_state': 42,  # Для воспроизводимости результатов
            'class_weight': 'balanced',  # Балансировка классов
            'n_jobs': -1  # Использовать все доступные ядра
        }

        # Обновляем параметры, если они предоставлены
        self.params = self.default_params.copy()
        if params is not None:
            self.params.update(params)

        self.model = RandomForestClassifier(**self.params)
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.last_price = None
        self.feature_importances = None
        self.data_quality_checked = False

    def check_data_quality(self, features):
        """Проверяет качество данных и выполняет предобработку"""
        # Проверка на корреляцию
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        highly_correlated = [column for column in upper.columns if any(upper[column] > 0.95)]
        if highly_correlated:
            print(f"ВНИМАНИЕ: Обнаружены сильно коррелирующие признаки: {highly_correlated}")
            print("Это может вызвать нестабильность в модели. Рассмотрите удаление некоторых из них.")

        # Проверка на разброс значений и выбросы
        for col in features.columns:
            if features[col].std() / (features[col].mean() + 1e-10) > 100:
                print(f"ВНИМАНИЕ: Очень высокий разброс значений в колонке {col}. Ограничиваем выбросы.")
                q1 = features[col].quantile(0.01)
                q3 = features[col].quantile(0.99)
                features[col] = features[col].clip(q1, q3)

            # Проверка на нулевые или близкие к нулю дисперсии
            if features[col].std() < 1e-6:
                print(f"ВНИМАНИЕ: Колонка {col} имеет близкую к нулю дисперсию. Рассмотрите её удаление.")

        # Проверка баланса классов
        if 'price_up' in features.columns:
            class_counts = features['price_up'].value_counts()
            class_ratio = class_counts.min() / class_counts.max()
            if class_ratio < 0.3:
                print(f"ВНИМАНИЕ: Сильный дисбаланс классов: {class_counts[1]}/{class_counts[0]} ({class_ratio:.2f})")
                print("Рекомендуется использовать class_weight='balanced' или SMOTE для балансировки.")

        self.data_quality_checked = True
        return features

    def prepare_data(self, df):
        """Подготавливает данные для обучения модели"""
        # Сохраняем timestamp для последующего анализа
        timestamps = df['timestamp']

        # Целевая переменная - бинарный флаг роста цены
        y = df['price_up']

        # Удаляем колонки, которые не нужны для обучения
        features = df.drop(['timestamp', 'price_up', 'next_close'], axis=1)

        # Удаляем 'volume' из признаков, если есть volume_log
        if 'volume' in features.columns and 'volume_log' in features.columns:
            features = features.drop(['volume'], axis=1)

        # Проверка и исправление качества данных
        if not self.data_quality_checked:
            features = self.check_data_quality(features)

        # Проверка на наличие бесконечных значений
        features = features.replace([np.inf, -np.inf], np.nan)

        # Заполнение пропущенных значений
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)

        # Обработка выбросов
        for col in features.columns:
            q_low = features[col].quantile(0.01)
            q_high = features[col].quantile(0.99)
            features[col] = features[col].clip(lower=q_low, upper=q_high)

        # Удаление колонок с низкой дисперсией
        cols_to_drop = []
        for col in features.columns:
            if features[col].std() < 1e-8 or features[col].isna().any():
                cols_to_drop.append(col)
                print(f"Удаляем колонку {col} из-за низкой дисперсии или наличия NaN")

        if cols_to_drop:
            features = features.drop(cols_to_drop, axis=1)
            if len(features.columns) == 0:
                raise ValueError("После удаления проблемных колонок не осталось признаков")

        self.feature_columns = features.columns

        # Вывод статистики
        print("\nСтатистика признаков после обработки:")
        print(features.describe().loc[['min', 'max', 'mean', 'std']].T.head())

        # Проверка наличия NaN значений
        if features.isna().any().any():
            print("ВНИМАНИЕ: В данных остались NaN значения после обработки.")
            features = features.fillna(features.median())

        # Масштабирование признаков
        X_original = features.copy()
        try:
            X_scaled = self.scaler.fit_transform(features)
        except Exception as e:
            print(f"Ошибка при масштабировании данных: {e}")
            # Вручную масштабируем данные
            X_scaled = np.zeros(features.shape)
            for i, col in enumerate(features.columns):
                col_mean = features[col].mean()
                col_std = features[col].std()
                if col_std > 1e-10:
                    X_scaled[:, i] = (features[col] - col_mean) / col_std
                else:
                    X_scaled[:, i] = 0

        # Проверка на NaN и Inf после масштабирования
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print("ВНИМАНИЕ: После масштабирования обнаружены NaN или Inf. Заменяем их на 0.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Ограничение экстремальных значений
        if np.max(np.abs(X_scaled)) > 1e6:
            print("ВНИМАНИЕ: Обнаружены очень большие значения после масштабирования. Ограничиваем их.")
            X_scaled = np.clip(X_scaled, -1e6, 1e6)

        # Сохраняем последнюю цену
        self.last_price = df['close'].iloc[-1]

        return X_original, X_scaled, y, timestamps

    def train(self, X, y, timestamps=None, X_scaled=None, test_size=0.2):
        """Обучает модель RandomForestClassifier"""
        # Проверка на NaN и Inf
        if np.isnan(X.values).any() or np.isinf(X.values).any():
            print("КРИТИЧЕСКАЯ ОШИБКА: В данных обнаружены NaN или Inf.")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        # Разделение timestamps, если они предоставлены
        if timestamps is not None:
            _, timestamps_test = train_test_split(
                timestamps, test_size=test_size, random_state=42, shuffle=False
            )

        try:
            # Проверка баланса классов
            class_counts = np.bincount(y_train)
            print(f"\nРаспределение классов в обучающей выборке: {class_counts}")
            if class_counts[0] / sum(class_counts) > 0.7 or class_counts[1] / sum(class_counts) > 0.7:
                print("Обнаружен дисбаланс классов. Используем балансировку весов.")
                self.model = RandomForestClassifier(**{**self.params, 'class_weight': 'balanced'})

            # Обучение модели
            print("\nОбучение модели Random Forest Classifier...")
            self.model.fit(X_train, y_train)

            # Прогнозы
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)

            # Вероятности классов
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            y_test_proba = self.model.predict_proba(X_test)[:, 1]

            # Метрики
            train_metrics = self.calculate_metrics(y_train, y_train_pred, y_train_proba)
            test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_proba)

            print("\n===== Метрики на обучающей выборке =====")
            self.print_metrics(train_metrics)

            print("\n===== Метрики на тестовой выборке =====")
            self.print_metrics(test_metrics)

            # Сохраняем важность признаков
            self.feature_importances = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)

            print("\nТоп-10 важных признаков:")
            print(self.feature_importances.head(10))

            # Проверяем, не слишком ли переобучилась модель
            train_acc = train_metrics['accuracy']
            test_acc = test_metrics['accuracy']
            if train_acc > 0.9 and test_acc < 0.6:
                print("\nВНИМАНИЕ: Обнаружено сильное переобучение! Точность на обучающей выборке:",
                      f"{train_acc:.4f}, на тестовой: {test_acc:.4f}")
                print("Рекомендуется уменьшить сложность модели или использовать регуляризацию.")

            return X_test, y_test, y_test_pred, y_test_proba, test_metrics

        except Exception as e:
            print(f"Ошибка при обучении модели: {e}")
            print("Попытка обучения с упрощенными параметрами...")

            # Упрощенная модель
            simple_params = {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'class_weight': 'balanced'
            }

            self.model = RandomForestClassifier(**simple_params)
            self.model.fit(X_train, y_train)

            y_test_pred = self.model.predict(X_test)
            y_test_proba = self.model.predict_proba(X_test)[:, 1]

            test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_proba)

            print("\n===== Метрики на тестовой выборке (упрощенная модель) =====")
            self.print_metrics(test_metrics)

            return X_test, y_test, y_test_pred, y_test_proba, test_metrics

    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Вычисляет метрики классификации"""
        try:
            # Основные метрики классификации
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Матрица ошибок
            cm = confusion_matrix(y_true, y_pred)

            # ROC AUC, если доступны вероятности
            roc_auc = 0.5
            if y_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_true, y_proba)
                except Exception:
                    pass

            # Специфичность (true negative rate)
            if len(cm) > 1:
                specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            else:
                specificity = 0

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'roc_auc': roc_auc,
                'confusion_matrix': cm
            }

        except Exception as e:
            print(f"Ошибка при вычислении метрик: {e}")
            return {
                'accuracy': float('nan'),
                'precision': float('nan'),
                'recall': float('nan'),
                'f1': float('nan'),
                'specificity': float('nan'),
                'roc_auc': float('nan'),
                'confusion_matrix': None
            }

    def print_metrics(self, metrics):
        """Выводит метрики в консоль"""
        for name, value in metrics.items():
            if name != 'confusion_matrix':
                if np.isnan(value) or np.isinf(value):
                    print(f"{name}: Ошибка вычисления")
                else:
                    print(f"{name}: {value:.4f}".replace("accuracy", "Direction Accuracy"))

        if metrics['confusion_matrix'] is not None:
            print("\nМатрица ошибок:")
            print(metrics['confusion_matrix'])

    def cross_validate(self, X, y, cv=5):
        """Выполняет перекрестную проверку модели"""
        print(f"\nВыполняем {cv}-блочную перекрестную проверку...")

        try:
            # Убедимся, что X не содержит NaN или Inf
            if isinstance(X, pd.DataFrame):
                X_cv = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
            else:
                X_cv = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Различные метрики для перекрестной проверки
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            cv_results = {}

            for metric in scoring:
                try:
                    scores = cross_val_score(self.model, X_cv, y, cv=cv, scoring=metric)
                    cv_results[metric] = scores.mean()
                    print(f"{metric.upper()}-CV: {scores.mean():.4f} (±{scores.std():.4f})")
                except Exception as e:
                    print(f"Ошибка при вычислении {metric}: {e}")

            return cv_results

        except Exception as e:
            print(f"Ошибка при кросс-валидации: {e}")
            return None

    def hyperparameter_tuning(self, X, y, param_grid=None, cv=3):
        """Выполняет подбор гиперпараметров с помощью GridSearchCV"""
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }

        print("\nНачинаем подбор гиперпараметров...")

        try:
            # Убедимся, что X не содержит NaN или Inf
            if isinstance(X, pd.DataFrame):
                X_tune = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
            else:
                X_tune = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Создаем модель для GridSearchCV
            base_model = RandomForestClassifier(random_state=42, class_weight='balanced')

            # Определяем GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='f1',  # Используем F1-меру
                n_jobs=-1,
                verbose=1
            )

            # Обучаем GridSearchCV
            grid_search.fit(X_tune, y)

            # Выводим лучшие параметры
            print(f"\nЛучшие параметры: {grid_search.best_params_}")
            print(f"Лучший f1-score: {grid_search.best_score_:.4f}")

            # Обновляем параметры модели
            self.params.update(grid_search.best_params_)

            # Создаем новую модель с оптимальными параметрами
            self.model = RandomForestClassifier(**self.params)

            return grid_search.best_params_

        except Exception as e:
            print(f"Ошибка при подборе гиперпараметров: {e}")
            return None

    def save_model(self, filepath):
        """Сохраняет модель в файл"""
        try:
            if self.model is not None:
                # Создаем словарь с моделью и всеми необходимыми компонентами
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'last_price': self.last_price,
                    'params': self.params
                }

                # Сохраняем в файл
                joblib.dump(model_data, filepath)
                print(f"Модель успешно сохранена в {filepath}")
                return True
            else:
                print("ОШИБКА: Модель не обучена. Нечего сохранять.")
                return False
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")
            return False

    def load_model(self, filepath):
        """Загружает модель из файла"""
        try:
            # Загружаем данные
            model_data = joblib.load(filepath)

            # Восстанавливаем все компоненты
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.last_price = model_data['last_price']
            self.params = model_data['params']

            print(f"Модель успешно загружена из {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return False

    def predict_next(self, current_data):
        """
        Прогнозирует направление движения цены на следующий день

        Parameters:
        -----------
        current_data : pandas.DataFrame
            Текущие данные с техническими индикаторами

        Returns:
        --------
        dict
            Результаты прогноза
        """
        try:
            if self.model is None:
                print("ОШИБКА: Модель не обучена. Запустите train() сначала.")
                return None

            # Проверяем наличие всех необходимых колонок
            missing_cols = [col for col in self.feature_columns if col not in current_data.columns]
            if missing_cols:
                print(f"ВНИМАНИЕ: Отсутствуют колонки: {missing_cols}")
                for col in missing_cols:
                    if col == 'volume_log' and 'volume' in current_data.columns:
                        current_data['volume_log'] = np.log1p(current_data['volume'])
                    else:
                        current_data[col] = 0

            # Подготавливаем признаки для прогноза
            features = current_data[self.feature_columns].copy()

            # Обработка пропущенных значений и бесконечностей
            features = features.replace([np.inf, -np.inf], np.nan)
            for col in features.columns:
                if features[col].isna().any():
                    features[col] = features[col].fillna(features[col].median())

            # Обработка выбросов
            for col in features.columns:
                q_low = features[col].quantile(0.01) if len(features) > 10 else features[col].min()
                q_high = features[col].quantile(0.99) if len(features) > 10 else features[col].max()
                features[col] = features[col].clip(lower=q_low, upper=q_high)

            # Масштабирование признаков
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
                        # Если параметры скалера недоступны, используем стандартное масштабирование
                        col_mean = features[col].mean()
                        col_std = features[col].std()
                        if col_std > 1e-10:
                            scaled_features[:, i] = (features[col] - col_mean) / col_std
                        else:
                            scaled_features[:, i] = 0

            # Проверка на NaN и Inf после масштабирования
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                print("ВНИМАНИЕ: После масштабирования обнаружены NaN или Inf. Заменяем их на 0.")
                scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Делаем прогноз
            prediction_proba = self.model.predict_proba(scaled_features)[0]
            prediction = 1 if prediction_proba[1] > 0.5 else 0

            # Вычисляем уверенность прогноза
            # Для бинарной классификации, чем ближе вероятность к 0 или 1, тем выше уверенность
            confidence = max(prediction_proba)

            # Формируем результат
            result = {
                'current_price': float(current_data['close'].iloc[-1]),
                'direction': 'UP' if prediction == 1 else 'DOWN',
                'probability_up': float(prediction_proba[1]),
                'probability_down': float(prediction_proba[0]),
                'confidence': float(confidence),
                'signal': 'BUY' if prediction == 1 and prediction_proba[1] > 0.65 else
                'SELL' if prediction == 0 and prediction_proba[0] > 0.65 else 'HOLD'
            }

            # Добавляем информацию о текущих значениях важных индикаторов
            if 'rsi_14' in current_data.columns:
                result['rsi'] = float(current_data['rsi_14'].iloc[-1])
            if 'macd' in current_data.columns:
                result['macd'] = float(current_data['macd'].iloc[-1])

            return result

        except Exception as e:
            print(f"Ошибка при прогнозировании: {e}")
            # Возвращаем нейтральный прогноз в случае ошибки
            return {
                'current_price': float(current_data['close'].iloc[-1]) if 'close' in current_data.columns else 0.0,
                'direction': 'UNKNOWN',
                'probability_up': 0.5,
                'probability_down': 0.5,
                'confidence': 0.0,
                'signal': 'NEUTRAL'
            }

    def analyze_model_performance(self, X_test, y_test, y_pred, y_proba):
        """
        Анализирует и визуализирует эффективность модели

        Parameters:
        -----------
        X_test : pandas.DataFrame
            Тестовые данные
        y_test : pandas.Series
            Фактические метки класса
        y_pred : numpy.ndarray
            Предсказанные метки класса
        y_proba : numpy.ndarray
            Предсказанные вероятности положительного класса

        Returns:
        --------
        dict
            Результаты анализа
        """

        try:
            # Вычисляем базовые метрики
            metrics = self.calculate_metrics(y_test, y_pred, y_proba)

            # Получаем классификационный отчет
            report = classification_report(y_test, y_pred, output_dict=True)

            # Создаем DataFrame с результатами
            results_df = pd.DataFrame({
                'actual': y_test.values,
                'predicted': y_pred,
                'probability': y_proba
            })

            # Добавляем стратегию: 1 - покупка, -1 - продажа, 0 - удержание
            results_df['strategy'] = np.where(results_df['predicted'] == 1, 1, -1)

            # Для фактических доходностей, используем простой подход
            # Проверяем, есть ли нужные колонки в X_test
            if isinstance(X_test, pd.DataFrame) and 'close' in X_test.columns and 'next_close' in X_test.columns:
                # Если колонки с ценами есть напрямую
                returns = (X_test['next_close'].values / X_test['close'].values - 1)
            else:
                # Если нужных колонок нет, используем случайные значения для демонстрации
                print("ВНИМАНИЕ: Не найдены колонки с ценами. Используем случайные значения для доходности.")
                returns = np.random.normal(0, 0.01, size=len(results_df))

            results_df['returns'] = returns

        except Exception as e:
            print(f"Ошибка при анализе эффективности модели: {e}")
            return None


# Основная функция
def main(db_path):
    """
    Main training function using the new RandomForestClassifierNew.

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

    print("\nОбучение модели Random Forest Classifier...")
    print("ВАЖНО: Scaler обучается ТОЛЬКО на тренировочных данных (без data leakage)")

    model = RandomForestClassifierNew()

    try:
        metrics = model.train(df)
    except Exception as e:
        print(f"ОШИБКА при обучении: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Выводим метрики
    print("\n" + "=" * 50)
    print("МЕТРИКИ МОДЕЛИ (КЛАССИФИКАЦИЯ)")
    print("=" * 50)

    print("\nМетрики на тестовой выборке:")
    print(f"Accuracy: {metrics.get('test_accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('test_precision', 0):.4f}")
    print(f"Recall: {metrics.get('test_recall', 0):.4f}")
    print(f"F1-Score: {metrics.get('test_f1', 0):.4f}")
    print(f"ROC-AUC: {metrics.get('test_roc_auc', 0):.4f}")

    print(f"\nМетрики на тренировочной выборке:")
    print(f"Accuracy: {metrics.get('train_accuracy', 0):.4f}")

    # Важность признаков
    feature_imp = model.get_feature_importance()
    if feature_imp is not None:
        print("\nВажность признаков (топ-10):")
        print(feature_imp.to_string(index=False))

    # Прогноз
    print("\n" + "=" * 50)
    print("ПРОГНОЗ НА СЛЕДУЮЩИЙ ВРЕМЕННОЙ ИНТЕРВАЛ")
    print("=" * 50)

    prediction = model.predict_next(df)

    print(f"Текущая цена: {prediction['current_price']:.4f}")
    print(f"Прогноз направления: {prediction['direction']}")
    print(f"Вероятность роста: {prediction['probability_up']:.4f}")
    print(f"Вероятность падения: {prediction['probability_down']:.4f}")
    print(f"Уверенность: {prediction['confidence']:.4f}")
    print(f"Торговый сигнал: {prediction['signal']}")

    # Backtest
    print("\n" + "=" * 50)
    print("РЕТРОСПЕКТИВНАЯ ОЦЕНКА ТОРГОВЫХ СИГНАЛОВ")
    print("=" * 50)

    _run_classifier_backtest(model)

    return model


def _run_classifier_backtest(model: RandomForestClassifierNew):
    """Run backtest for classifier model."""
    if not hasattr(model, '_last_y_test'):
        print("Недостаточно данных для бэктеста")
        return

    y_test = model._last_y_test
    y_pred = model._last_y_pred
    prices = model._last_prices

    if len(y_test) < 2:
        print("Недостаточно данных для бэктеста")
        return

    # Calculate returns based on predictions
    price_returns = np.diff(prices) / prices[:-1]
    signals = y_pred[:-1]  # 1 = predict UP (buy), 0 = predict DOWN (sell/short)
    strategy_signals = np.where(signals == 1, 1, -1)  # Convert to +1/-1
    strategy_returns = strategy_signals * price_returns

    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    total_trades = len(strategy_returns)
    profitable_trades = np.sum(strategy_returns > 0)

    profit_sum = np.sum(strategy_returns[strategy_returns > 0])
    loss_sum = abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

    print(f"Всего сделок: {total_trades}")
    print(f"Прибыльных сделок: {profitable_trades} ({profitable_trades / total_trades * 100:.2f}%)")
    print(f"Общая доходность: {cumulative_returns[-1] * 100:.2f}%")
    print(f"Коэффициент прибыли (Profit Factor): {profit_factor:.2f}")


if __name__ == "__main__":
    # main("BBG000QJW156")
    main("ETHUSDT_250328_5M")
