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



# Отключаем предупреждения
warnings.filterwarnings("ignore")

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
    # Загружаем данные
    print("Загрузка данных...")

    df = load_data(db_path)
    # df = load_crypto_data(db_path)
    # df["timestamp"] = df["open_time"]
    # df = df.drop(["open_time"], axis=1)
    # df = df.drop(["close_time"], axis=1)
    try:
        df = df.drop(["figi"], axis=1)
    except KeyError:
        pass  # Column doesn't exist

    print(
        f"Загружено {len(df)} записей за период с {df['timestamp'].min() if not df.empty else 'N/A'} по {df['timestamp'].max() if not df.empty else 'N/A'}")

    # Проверка на пустой DataFrame
    if df.empty:
        print(f"ОШИБКА: Файл {db_path} не содержит данных. Пропускаем обработку.")
        return None, None

    # Создаем признаки
    print("\nСоздание признаков...")
    df_features = create_features(df)
    print(f"Создано признаков: {len(df_features.columns) - 3}")  # -3 для timestamp, next_close и price_up

    # Проверка на пустой DataFrame после создания признаков
    if df_features.empty:
        print(
            f"ОШИБКА: После создания признаков для {db_path} не осталось данных (возможно, из-за NaN). Пропускаем обработку.")
        return None, None

    # Информация о признаках
    print("\nСтатистика признаков:")
    print(df_features.describe().T[['mean', 'min', 'max', 'std']])

    # Проверка на бесконечные значения
    inf_check = np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()
    if inf_check > 0:
        print(f"ВНИМАНИЕ: обнаружено {inf_check} бесконечных значений. Они будут обработаны.")

    # Информация о целевой переменной
    up_count = df_features['price_up'].sum()
    down_count = len(df_features) - up_count
    up_percentage = up_count / len(df_features) * 100

    print(f"\nРаспределение целевой переменной:")
    print(f"UP (рост): {up_count} ({up_percentage:.2f}%)")
    print(f"DOWN (падение): {down_count} ({100 - up_percentage:.2f}%)")

    # Создаем и обучаем модель
    print("\nОбучение модели Random Forest Classifier...")
    model = RandomForestDirectionModel()

    # Подготавливаем данные
    X_original, X_scaled, y, timestamps = model.prepare_data(df_features)

    # Обучаем модель
    X_test, y_test, y_pred, y_proba, test_metrics = model.train(X_original, y, timestamps)

    # Проводим кросс-валидацию
    print("\nПроводим кросс-валидацию модели...")
    cv_results = model.cross_validate(X_original, y, cv=5)

    # Анализируем результаты
    print("\nЛучшие признаки по важности:")
    print(model.feature_importances.head(10))

    # Выводим последние 5 прогнозов и сравниваем с фактическими значениями
    if len(y_test) > 0:
        print("\nПоследние 5 прогнозов:")
        last_n = min(5, len(y_test))
        for i in range(last_n):
            idx = len(y_test) - last_n + i
            actual = "UP" if y_test.iloc[idx] == 1 else "DOWN"
            predicted = "UP" if y_pred[idx] == 1 else "DOWN"
            prob_up = y_proba[idx] #  это вероятность роста (класс "UP"), предсказанная моделью Random Forest
            prob_down = 1 - prob_up     # это вероятность падения (класс "DOWN"),

            print(f"#{idx} Фактически: {actual}, Прогноз: {predicted}, "
                  f"Вероятность UP: {prob_up:.4f}, Вероятность DOWN: {prob_down:.4f}, "
                  f"Уверенность: {max(prob_up, prob_down):.4f}")   # Уверенность в выводе - это просто максимальная из двух вероятностей.

    # Прогноз на следующий день
    print("\nПрогноз на следующий день:")
    # Берем последнюю строку из оригинального датасета
    latest_row = df.iloc[-1:].copy()

    # Создаем признаки только для этой строки
    latest_features = create_features(
        pd.concat([df.iloc[-30:].iloc[:-1], latest_row]))  # Берем предыдущие 30 дней для расчета индикаторов
    latest_features = latest_features.iloc[-1:].copy()  # Оставляем только последнюю строку с рассчитанными признаками

    # Если в latest_features есть NaN в next_close, заменяем его на 0 или другое значение
    if 'next_close' in latest_features.columns and latest_features['next_close'].isna().any():
        latest_features['next_close'] = 0  # или другое подходящее значение
    # last_data = df_features.iloc[-1:].copy()
    """
    берется вчерашний день, по нему предсказывается next_close для сегодняшнего дня, что означает, 
    что предсказывается цена актива на завтрашний день => лучше делать прогноз вечером, т.к. для вчерашнего дня 
    есть актуальная next_close (т.е. сегодняшняя цена закрытия она уже не будет менять после закрытия биржи)
    и по результатам прогноза мы получим next_close для сегодняшнего дня => цену актива на завтра  
    """
    print(latest_features)
    prediction = model.predict_next(latest_features)

    print(f"Текущая цена: {prediction['current_price']:.4f}")
    print(f"Прогноз направления: {prediction['direction']}")
    print(f"Вероятность роста: {prediction['probability_up']:.4f}")
    print(f"Вероятность падения: {prediction['probability_down']:.4f}")
    print(f"Уверенность: {prediction['confidence']:.4f}")
    print(f"Торговый сигнал: {prediction['signal']}")

    # Анализ эффективности модели
    print("\nАнализ эффективности модели на тестовых данных:")
    perf_analysis = model.analyze_model_performance(X_test, y_test, y_pred, y_proba)

    if perf_analysis:
        trading_metrics = perf_analysis['trading_metrics']
        print(f"Всего сделок: {trading_metrics['total_trades']}")
        print(f"Прибыльных сделок: {trading_metrics['winning_trades']} ({trading_metrics['win_rate'] * 100:.2f}%)")
        print(f"Коэффициент выигрыша (Profit Factor): {trading_metrics['profit_factor']:.2f}")
        print(f"Средняя прибыльная сделка: {trading_metrics['avg_win'] * 100:.2f}%")
        print(f"Средняя убыточная сделка: {trading_metrics['avg_loss'] * 100:.2f}%")
        print(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}")
        print(f"Максимальная просадка: {trading_metrics['max_drawdown'] * 100:.2f}%")
        print(f"Общая доходность: {trading_metrics['final_return'] * 100:.2f}%")

    # Сохраняем модель
    model_filename = f"rf_direction_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    # model.save_model(model_filename)

    return model, df_features


if __name__ == "__main__":
    # main("BBG000QJW156")
    main("ETHUSDT_250328_5M")
