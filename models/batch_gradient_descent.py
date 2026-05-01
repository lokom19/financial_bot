import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from utils.load_data_method import load_data


# Отключаем все RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


class BatchGradientDescentRegressor:
    """
    Линейная регрессия с использованием Batch Gradient Descent
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6,
                 regularization='ridge', lambda_reg=1.0, verbose=False):
        """
        Параметры:
        - learning_rate: скорость обучения
        - max_iterations: максимальное количество итераций
        - tolerance: критерий остановки (изменение функции потерь)
        - regularization: тип регуляризации ('ridge', 'lasso', 'elastic', None)
        - lambda_reg: коэффициент регуляризации
        - verbose: вывод процесса обучения
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.verbose = verbose

        # Параметры модели
        self.weights = None
        self.bias = None

        # История обучения
        self.cost_history = []
        self.weight_history = []

    def _add_bias_term(self, X):
        """Добавляет столбец единиц для bias термина"""
        return np.column_stack([np.ones(X.shape[0]), X])

    def _compute_cost(self, y_true, y_pred, weights):
        """Вычисляет функцию потерь с регуляризацией"""
        m = len(y_true)

        # Основная функция потерь (MSE)
        mse_cost = np.mean((y_true - y_pred) ** 2)

        # Добавляем регуляризацию
        reg_cost = 0
        if self.regularization == 'ridge':
            # L2 регуляризация (исключаем bias из регуляризации)
            reg_cost = self.lambda_reg * np.sum(weights[1:] ** 2)
        elif self.regularization == 'lasso':
            # L1 регуляризация
            reg_cost = self.lambda_reg * np.sum(np.abs(weights[1:]))
        elif self.regularization == 'elastic':
            # Elastic Net (комбинация L1 и L2)
            l1_reg = 0.5 * self.lambda_reg * np.sum(np.abs(weights[1:]))
            l2_reg = 0.5 * self.lambda_reg * np.sum(weights[1:] ** 2)
            reg_cost = l1_reg + l2_reg

        return mse_cost + reg_cost

    def _compute_gradients(self, X, y_true, y_pred, weights):
        """Вычисляет градиенты функции потерь"""
        m = X.shape[0]

        # Градиент основной функции потерь
        error = y_pred - y_true
        gradients = (2 / m) * X.T.dot(error)

        # Добавляем градиенты регуляризации
        if self.regularization == 'ridge':
            # L2 регуляризация (не применяем к bias)
            reg_gradients = np.zeros_like(gradients)
            reg_gradients[1:] = 2 * self.lambda_reg * weights[1:]
            gradients += reg_gradients
        elif self.regularization == 'lasso':
            # L1 регуляризация (субградиент)
            reg_gradients = np.zeros_like(gradients)
            reg_gradients[1:] = self.lambda_reg * np.sign(weights[1:])
            gradients += reg_gradients
        elif self.regularization == 'elastic':
            # Elastic Net
            reg_gradients = np.zeros_like(gradients)
            reg_gradients[1:] = self.lambda_reg * (np.sign(weights[1:]) + weights[1:])
            gradients += reg_gradients

        return gradients

    def fit(self, X, y):
        """Обучение модели с использованием batch gradient descent"""
        # Добавляем bias терм
        X_with_bias = self._add_bias_term(X)
        m, n = X_with_bias.shape

        # Инициализация весов (Xavier/Glorot инициализация)
        self.weights = np.random.normal(0, np.sqrt(2.0 / n), n)

        # История для анализа
        self.cost_history = []
        self.weight_history = []

        prev_cost = float('inf')

        for iteration in range(self.max_iterations):
            # Прямой проход (forward pass)
            y_pred = X_with_bias.dot(self.weights)

            # Вычисляем функцию потерь
            current_cost = self._compute_cost(y, y_pred, self.weights)
            self.cost_history.append(current_cost)

            # Сохраняем веса для анализа
            if iteration % 10 == 0:  # Сохраняем каждые 10 итераций
                self.weight_history.append(self.weights.copy())

            # Вычисляем градиенты
            gradients = self._compute_gradients(X_with_bias, y, y_pred, self.weights)

            # Проверяем на NaN или Inf в градиентах
            if np.any(np.isnan(gradients)) or np.any(np.isinf(gradients)):
                print(f"ВНИМАНИЕ: Обнаружены NaN или Inf в градиентах на итерации {iteration}")
                # Применяем gradient clipping
                gradients = np.clip(gradients, -1e6, 1e6)

            # Обновляем веса
            self.weights -= self.learning_rate * gradients

            # Gradient clipping для весов
            self.weights = np.clip(self.weights, -1e6, 1e6)

            # Вывод прогресса
            if self.verbose and iteration % 100 == 0:
                print(f"Итерация {iteration}: Cost = {current_cost:.6f}")

            # Проверка критерия остановки
            if abs(prev_cost - current_cost) < self.tolerance:
                if self.verbose:
                    print(f"Сходимость достигнута на итерации {iteration}")
                break

            prev_cost = current_cost

        # Сохраняем отдельно bias и веса
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

        if self.verbose:
            print(f"Обучение завершено. Финальная функция потерь: {current_cost:.6f}")

        return self

    def predict(self, X):
        """Предсказание"""
        if self.weights is None:
            raise ValueError("Модель не обучена. Вызовите метод fit() сначала.")

        # Проверяем размерности
        if X.shape[1] != len(self.weights):
            raise ValueError(f"Ожидается {len(self.weights)} признаков, получено {X.shape[1]}")

        predictions = X.dot(self.weights) + self.bias

        # Проверяем на NaN или Inf
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            print("ВНИМАНИЕ: В предсказаниях обнаружены NaN или Inf значения")
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)

        return predictions

    def get_feature_importance(self):
        """Возвращает важность признаков (абсолютные значения весов)"""
        if self.weights is None:
            raise ValueError("Модель не обучена")

        return np.abs(self.weights)

    def plot_training_history(self):
        """Визуализация процесса обучения"""
        if not self.cost_history:
            print("История обучения пуста")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # График функции потерь
        ax1.plot(self.cost_history)
        ax1.set_title('Функция потерь во время обучения')
        ax1.set_xlabel('Итерация')
        ax1.set_ylabel('Cost')
        ax1.grid(True)

        # График изменения весов
        if self.weight_history:
            weight_history_array = np.array(self.weight_history)
            for i in range(min(5, weight_history_array.shape[1])):  # Показываем первые 5 весов
                ax2.plot(weight_history_array[:, i], label=f'Weight {i}')
            ax2.set_title('Изменение весов во время обучения')
            ax2.set_xlabel('Итерация (каждые 10)')
            ax2.set_ylabel('Значение веса')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.show()


class ImprovedTradeModelBGD:
    """
    Улучшенная торговая модель с использованием Batch Gradient Descent
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization='ridge',
                 lambda_reg=1.0, tolerance=1e-6, verbose=False):
        # Модель градиентного спуска
        self.model = BatchGradientDescentRegressor(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
            lambda_reg=lambda_reg,
            verbose=verbose
        )

        # Скалер для нормализации
        self.scaler = RobustScaler()

        self.feature_columns = None
        self.last_price = None
        self.feature_importances = None
        self.data_quality_checked = False

    def check_data_quality(self, features):
        """Проверка качества данных"""
        # Проверка на корреляцию
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        highly_correlated = [column for column in upper.columns if any(upper[column] > 0.95)]

        if highly_correlated:
            print(f"ВНИМАНИЕ: Обнаружены сильно коррелирующие признаки: {highly_correlated}")

        # Проверка на разброс значений и выбросы
        for col in features.columns:
            if features[col].std() / (features[col].mean() + 1e-10) > 100:
                print(f"ВНИМАНИЕ: Очень высокий разброс значений в колонке {col}")
                # Применяем обрезку выбросов
                q1 = features[col].quantile(0.01)
                q3 = features[col].quantile(0.99)
                features[col] = features[col].clip(q1, q3)

        self.data_quality_checked = True
        return features

    def prepare_data(self, df):
        """Подготовка данных для обучения"""
        timestamps = df['timestamp']
        y = df['next_close']

        # Удаляем ненужные колонки
        features = df.drop(['timestamp', 'next_close'], axis=1)

        if 'volume' in features.columns and 'volume_log' in features.columns:
            features = features.drop(['volume'], axis=1)

        # Проверка качества данных
        if not self.data_quality_checked:
            features = self.check_data_quality(features)

        # Обработка выбросов и пропущенных значений
        features = features.replace([np.inf, -np.inf], np.nan)

        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)

        # Винсоризация
        for col in features.columns:
            q_low = features[col].quantile(0.01)
            q_high = features[col].quantile(0.99)
            features[col] = features[col].clip(lower=q_low, upper=q_high)

        # Удаление колонок с низкой дисперсией
        cols_to_drop = []
        for col in features.columns:
            if features[col].std() < 1e-8:
                cols_to_drop.append(col)

        if cols_to_drop:
            features = features.drop(cols_to_drop, axis=1)
            print(f"Удалены колонки с низкой дисперсией: {cols_to_drop}")

        self.feature_columns = features.columns

        # Масштабирование
        X = self.scaler.fit_transform(features)

        # Проверка на NaN и Inf после масштабирования
        if np.isnan(X).any() or np.isinf(X).any():
            print("ВНИМАНИЕ: Обнаружены NaN или Inf после масштабирования")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, timestamps

    def train(self, X, y, timestamps=None, df_features=None):
        """Обучение модели"""
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        if timestamps is not None:
            _, timestamps_test = train_test_split(
                timestamps, test_size=0.2, random_state=42, shuffle=False
            )

        # Обучение модели
        print("Начало обучения с Batch Gradient Descent...")
        self.model.fit(X_train, y_train)

        # Предсказания
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        # Важность признаков
        importance_values = self.model.get_feature_importance()
        self.feature_importances = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False)

        # Метрики
        train_metrics = self.calculate_metrics(y_train, train_preds)
        test_metrics = self.calculate_metrics(y_test, test_preds)

        print("\n===== Метрики на обучающей выборке =====")
        self.print_metrics(train_metrics)

        print("\n===== Метрики на тестовой выборке =====")
        self.print_metrics(test_metrics)

        # Сохраняем последнюю цену
        if np.isnan(y.iloc[-1]) and df_features is not None:
            self.last_price = df_features.iloc[-1]['close']
        else:
            self.last_price = y.iloc[-1]

        return X_test, y_test, test_preds, test_metrics

    def calculate_metrics(self, y_true, y_pred):
        """Вычисление метрик качества"""
        try:
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

            try:
                r2 = r2_score(y_true_filtered, y_pred_filtered)
            except ValueError:
                r2 = float('nan')

            # MAPE
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_raw = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)
                mape_raw = np.where(np.isinf(mape_raw) | np.isnan(mape_raw), 0, mape_raw)
                mape = np.mean(mape_raw) * 100

            # Точность направления
            if len(y_true_filtered) > 1:
                if hasattr(y_true_filtered, 'iloc'):
                    direction_true = np.diff(np.append(y_true_filtered.iloc[0], y_true_filtered.values)) > 0
                else:
                    direction_true = np.diff(np.append(y_true_filtered[0], y_true_filtered)) > 0
                direction_pred = np.diff(np.append(y_true_filtered[0] if hasattr(y_true_filtered, 'iloc')
                                                   else y_true_filtered[0], y_pred_filtered)) > 0
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
        """Вывод метрик"""
        for name, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                print(f"{name}: Ошибка вычисления")
            else:
                print(f"{name}: {value:.4f}")

    def predict_next(self, current_data):
        """Предсказание следующей цены"""
        try:
            current_price = current_data['close'].iloc[0]

            # Проверяем наличие нужных колонок
            missing_cols = []
            for col in self.feature_columns:
                if col not in current_data.columns:
                    if col == 'volume_log' and 'volume' in current_data.columns:
                        current_data['volume_log'] = np.log1p(current_data['volume'])
                    else:
                        missing_cols.append(col)

            if missing_cols:
                print(f"ВНИМАНИЕ: Отсутствуют колонки: {missing_cols}")
                for col in missing_cols:
                    current_data[col] = 0

            # Подготовка данных
            features = current_data[self.feature_columns].copy()
            features = features.replace([np.inf, -np.inf], np.nan)

            for col in features.columns:
                if features[col].isna().any():
                    features[col] = features[col].fillna(features[col].median())

            # Масштабирование
            try:
                scaled_features = self.scaler.transform(features)
            except Exception as e:
                print(f"Ошибка масштабирования: {e}")
                return {
                    'current_price': current_price,
                    'predicted_price': current_price,
                    'price_change_pct': 0.0,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0
                }

            # Проверка на NaN и Inf
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Предсказание
            predicted_price = self.model.predict(scaled_features)[0]

            if np.isnan(predicted_price) or np.isinf(predicted_price):
                predicted_price = current_price

            # Изменение цены
            price_change = (predicted_price - current_price) / current_price * 100

            # Ограничение изменений
            if abs(price_change) > 10:
                price_change = np.sign(price_change) * 10
                predicted_price = current_price * (1 + price_change / 100)

            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change,
                'signal': 'BUY' if price_change > 0 else 'SELL',
                'confidence': 0.0
            }

        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            return {
                'current_price': self.last_price if hasattr(self, 'last_price') else 0,
                'predicted_price': self.last_price if hasattr(self, 'last_price') else 0,
                'price_change_pct': 0.0,
                'signal': 'NEUTRAL',
                'confidence': 0.0
            }

    def get_feature_importance(self, top_n=10):
        """Получение важности признаков"""
        if self.feature_importances is None:
            print("Модель не обучена")
            return None
        return self.feature_importances.head(top_n)

    def plot_training_history(self):
        """Визуализация процесса обучения"""
        self.model.plot_training_history()


# Функция создания признаков (из оригинального кода)
def create_features(df):
    """Создание технических индикаторов"""
    df_features = df.copy()

    # Простые скользящие средние
    df_features['sma_5'] = df['close'].rolling(window=5).mean()
    df_features['sma_10'] = df['close'].rolling(window=10).mean()
    df_features['sma_20'] = df['close'].rolling(window=20).mean()

    # Экспоненциальные скользящие средние
    df_features['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df_features['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df_features['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Разности
    df_features['close_minus_sma_5'] = df['close'] - df_features['sma_5']
    df_features['close_minus_sma_10'] = df['close'] - df_features['sma_10']

    # Относительные разности
    df_features['close_rel_sma_5'] = df['close'] / df_features['sma_5'] - 1
    df_features['close_rel_sma_10'] = df['close'] / df_features['sma_10'] - 1

    # Объём
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

    # Диапазон High-Low
    df_features['high_low_ratio'] = df['high'] / df['low']

    # True Range
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

    # Целевая переменная
    df_features['next_close'] = df_features['close'].shift(-1)

    # Заполнение пропущенных значений
    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    # Обработка бесконечных значений
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    return df_features


# Пример использования
def main_bgd(df, learning_rate=0.01, max_iterations=1000, regularization='ridge', lambda_reg=1.0):
    """
    Основная функция для обучения модели с Batch Gradient Descent

    Параметры:
    - df: DataFrame с данными
    - learning_rate: скорость обучения (по умолчанию 0.01)
    - max_iterations: максимальное количество итераций (по умолчанию 1000)
    - regularization: тип регуляризации ('ridge', 'lasso', 'elastic', None)
    - lambda_reg: коэффициент регуляризации (по умолчанию 1.0)
    """

    print("=== Обучение торговой модели с Batch Gradient Descent ===")
    print(f"Параметры: lr={learning_rate}, max_iter={max_iterations}, reg={regularization}, lambda={lambda_reg}")

    # Удаляем ненужные колонки
    try:
        df = df.drop(["figi"], axis=1)
    except KeyError:
        pass  # Column doesn't exist

    print(f"Загружено {len(df)} записей")

    if df.empty:
        print("ОШИБКА: DataFrame пуст")
        return None, None

    # Создание признаков
    print("\nСоздание признаков...")
    df_features = create_features(df)
    print(f"Создано признаков: {len(df_features.columns) - 2}")

    if df_features.empty:
        print("ОШИБКА: После создания признаков данные пусты")
        return None, None

    # Создание и обучение модели
    print("\nИнициализация модели BGD...")
    model = ImprovedTradeModelBGD(
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        regularization=regularization,
        lambda_reg=lambda_reg,
        tolerance=1e-6,
        verbose=True
    )

    # Подготовка данных
    X, y, timestamps = model.prepare_data(df_features)

    if X.shape[0] == 0:
        print("ОШИБКА: Нет данных для обучения")
        return None, None

    # Обучение
    X_test, y_test, predictions, test_metrics = model.train(X, y, timestamps, df_features)

    # Важность признаков
    print("\nВажность признаков (топ-10):")
    print(model.get_feature_importance(10))

    # Визуализация процесса обучения
    print("\nВизуализация процесса обучения:")
    model.plot_training_history()

    # Последние предсказания
    print("\nПоследние 5 предсказаний:")
    last_indices = range(len(y_test) - 5, len(y_test))
    for i in last_indices:
        real_price = y_test.iloc[i]
        pred_price = predictions[i]
        error_pct = (pred_price - real_price) / real_price * 100 if not np.isnan(
            real_price) and real_price != 0 else float('nan')
        print(f"Реальная цена: {real_price:.4f}, Предсказанная: {pred_price:.4f}, Ошибка: {error_pct:.2f}%")

    # Прогноз на следующий период
    print("\nПрогноз на следующий временной интервал:")
    latest_row = df.iloc[-1:].copy()

    # Создаем признаки для последней строки
    latest_features = create_features(
        pd.concat([df.iloc[-30:].iloc[:-1], latest_row])
    )
    latest_features = latest_features.iloc[-1:].copy()

    if 'next_close' in latest_features.columns and latest_features['next_close'].isna().any():
        latest_features['next_close'] = 0

    prediction = model.predict_next(latest_features)

    print(f"Текущая цена: {prediction['current_price']:.4f}")
    print(f"Прогнозируемая цена: {prediction['predicted_price']:.4f}")
    print(f"Ожидаемое изменение: {prediction['price_change_pct']:.2f}%")
    print(f"Торговый сигнал: {prediction['signal']}")

    # Оценка торговых сигналов
    print("\nРетроспективная оценка торговых сигналов:")
    valid_mask = ~np.isnan(y_test)
    y_test_valid = y_test[valid_mask]
    predictions_valid = predictions[valid_mask]

    if len(y_test_valid) > 1:
        y_test_shifted = y_test_valid.shift(1).fillna(y_test_valid.iloc[0])
        signals = np.sign(np.clip(predictions_valid - y_test_shifted, -1e10, 1e10))
        actual_returns = y_test_valid.pct_change().fillna(0)
        strategy_returns = signals[:-1] * actual_returns[1:].values
        cumulative_returns = (1 + strategy_returns).cumprod() - 1
        total_trades = np.sum(np.abs(np.diff(signals)) > 0) + 1
        profitable_trades = np.sum(strategy_returns > 0)
        profit_sum = np.sum(strategy_returns[strategy_returns > 0])
        loss_sum = abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

        # Sharpe Ratio (annualized) — filter NaN
        clean_returns = strategy_returns[~np.isnan(strategy_returns)]
        if len(clean_returns) > 1:
            sharpe_ratio = np.mean(clean_returns) / (np.std(clean_returns) + 1e-9) * np.sqrt(252)
            equity_curve = (1 + clean_returns).cumprod()
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = (peak - equity_curve) / peak
            max_drawdown = np.max(drawdowns)
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0

        print(f"Всего сделок: {total_trades}")
        print(f"Прибыльных сделок: {profitable_trades} ({profitable_trades / total_trades * 100:.2f}%)")
        print(f"Общая доходность: {cumulative_returns.iloc[-1] * 100:.2f}%")
        print(f"Коэффициент прибыли (Profit Factor): {profit_factor:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Максимальная просадка: {max_drawdown * 100:.2f}%")
    else:
        print("Недостаточно данных для оценки торговых сигналов")

    return model, df_features


# Дополнительные функции для анализа и сравнения моделей

def compare_models(df, configurations):
    """
    Сравнение различных конфигураций модели BGD

    Параметры:
    - df: DataFrame с данными
    - configurations: список словарей с параметрами моделей
    """
    results = []

    print("=== Сравнение различных конфигураций модели ===")

    for i, config in enumerate(configurations):
        print(f"\n--- Конфигурация {i + 1}: {config} ---")

        try:
            model, df_features = main_bgd(df, **config)

            if model is not None:
                # Получаем метрики последнего обучения
                X, y, _ = model.prepare_data(df_features)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )

                test_preds = model.model.predict(X_test)
                test_metrics = model.calculate_metrics(y_test, test_preds)

                results.append({
                    'config': config,
                    'model': model,
                    'metrics': test_metrics,
                    'converged': len(model.model.cost_history) < config.get('max_iterations', 1000)
                })
            else:
                results.append({
                    'config': config,
                    'model': None,
                    'metrics': None,
                    'converged': False
                })

        except Exception as e:
            print(f"Ошибка в конфигурации {i + 1}: {e}")
            results.append({
                'config': config,
                'model': None,
                'metrics': None,
                'converged': False,
                'error': str(e)
            })

    # Сравнительная таблица
    print("\n=== Сравнительная таблица результатов ===")
    print(f"{'Config':<8} {'LR':<8} {'MaxIter':<8} {'Reg':<8} {'Lambda':<8} {'RMSE':<10} {'R²':<8} {'Converged':<10}")
    print("-" * 80)

    for i, result in enumerate(results):
        config = result['config']
        metrics = result['metrics']

        if metrics:
            rmse = metrics.get('RMSE', float('nan'))
            r2 = metrics.get('R²', float('nan'))
        else:
            rmse = float('nan')
            r2 = float('nan')

        converged = result.get('converged', False)

        print(f"{i + 1:<8} {config.get('learning_rate', 'N/A'):<8} {config.get('max_iterations', 'N/A'):<8} "
              f"{config.get('regularization', 'N/A'):<8} {config.get('lambda_reg', 'N/A'):<8} "
              f"{rmse:<10.4f} {r2:<8.4f} {converged}")

    return results


def adaptive_learning_rate_bgd(df, initial_lr=0.1, decay_factor=0.95, min_lr=1e-6,
                               max_iterations=2000, regularization='ridge', lambda_reg=1.0):
    """
    Модель BGD с адаптивной скоростью обучения
    """
    print("=== Обучение с адаптивной скоростью обучения ===")

    class AdaptiveBGDRegressor(BatchGradientDescentRegressor):
        def __init__(self, initial_lr, decay_factor, min_lr, **kwargs):
            super().__init__(learning_rate=initial_lr, **kwargs)
            self.initial_lr = initial_lr
            self.decay_factor = decay_factor
            self.min_lr = min_lr
            self.lr_history = []

        def fit(self, X, y):
            X_with_bias = self._add_bias_term(X)
            m, n = X_with_bias.shape

            self.weights = np.random.normal(0, np.sqrt(2.0 / n), n)
            self.cost_history = []
            self.lr_history = []

            current_lr = self.learning_rate
            prev_cost = float('inf')
            patience = 50
            no_improvement = 0

            for iteration in range(self.max_iterations):
                # Прямой проход
                y_pred = X_with_bias.dot(self.weights)
                current_cost = self._compute_cost(y, y_pred, self.weights)
                self.cost_history.append(current_cost)
                self.lr_history.append(current_lr)

                # Вычисляем градиенты
                gradients = self._compute_gradients(X_with_bias, y, y_pred, self.weights)

                # Проверка на NaN/Inf
                if np.any(np.isnan(gradients)) or np.any(np.isinf(gradients)):
                    gradients = np.clip(gradients, -1e6, 1e6)

                # Обновляем веса с текущей скоростью обучения
                self.weights -= current_lr * gradients
                self.weights = np.clip(self.weights, -1e6, 1e6)

                # Адаптация скорости обучения
                if current_cost >= prev_cost:
                    no_improvement += 1
                    if no_improvement > patience and current_lr > self.min_lr:
                        current_lr *= self.decay_factor
                        current_lr = max(current_lr, self.min_lr)
                        no_improvement = 0
                        if self.verbose:
                            print(f"Итерация {iteration}: Уменьшение LR до {current_lr:.6f}")
                else:
                    no_improvement = 0

                if self.verbose and iteration % 100 == 0:
                    print(f"Итерация {iteration}: Cost = {current_cost:.6f}, LR = {current_lr:.6f}")

                # Критерий остановки
                if abs(prev_cost - current_cost) < self.tolerance:
                    if self.verbose:
                        print(f"Сходимость достигнута на итерации {iteration}")
                    break

                prev_cost = current_cost

            # Сохраняем финальные параметры
            self.bias = self.weights[0]
            self.weights = self.weights[1:]

            return self

        def plot_training_history(self):
            """Расширенная визуализация с историей LR"""
            if not self.cost_history:
                print("История обучения пуста")
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # График функции потерь
            ax1.plot(self.cost_history, 'b-', label='Cost')
            ax1.set_title('Функция потерь во время обучения')
            ax1.set_xlabel('Итерация')
            ax1.set_ylabel('Cost')
            ax1.legend()
            ax1.grid(True)

            # График скорости обучения
            ax2.plot(self.lr_history, 'r-', label='Learning Rate')
            ax2.set_title('Адаптация скорости обучения')
            ax2.set_xlabel('Итерация')
            ax2.set_ylabel('Learning Rate')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

    # Создаем модель с адаптивной скоростью обучения
    class AdaptiveTradeModel(ImprovedTradeModelBGD):
        def __init__(self, initial_lr, decay_factor, min_lr, **kwargs):
            super().__init__(**kwargs)
            self.model = AdaptiveBGDRegressor(
                initial_lr=initial_lr,
                decay_factor=decay_factor,
                min_lr=min_lr,
                max_iterations=kwargs.get('max_iterations', 2000),
                tolerance=kwargs.get('tolerance', 1e-6),
                regularization=kwargs.get('regularization', 'ridge'),
                lambda_reg=kwargs.get('lambda_reg', 1.0),
                verbose=kwargs.get('verbose', True)
            )

    # Обучение адаптивной модели
    try:
        df = df.drop(["figi"], axis=1)
    except KeyError:
        pass  # Column doesn't exist

    print(f"Загружено {len(df)} записей")

    df_features = create_features(df)
    print(f"Создано признаков: {len(df_features.columns) - 2}")

    model = AdaptiveTradeModel(
        initial_lr=initial_lr,
        decay_factor=decay_factor,
        min_lr=min_lr,
        max_iterations=max_iterations,
        regularization=regularization,
        lambda_reg=lambda_reg,
        verbose=True
    )

    X, y, timestamps = model.prepare_data(df_features)
    X_test, y_test, predictions, test_metrics = model.train(X, y, timestamps, df_features)

    print("\nВажность признаков (топ-10):")
    print(model.get_feature_importance(10))

    print("\nВизуализация обучения с адаптивной скоростью:")
    model.plot_training_history()

    return model, df_features


# Пример использования различных конфигураций
if __name__ == "__main__":
    # Загрузка данных (замените на свою функцию загрузки)
    df = load_data("BBG000Q7ZZY2")

    # Пример 1: Базовое использование
    print("=== Пример 1: Базовое использование ===")
    # model, df_features = main_bgd(df)

    # Пример 2: Сравнение различных конфигураций
    print("\n=== Пример 2: Сравнение конфигураций ===")
    configurations = [
        {'learning_rate': 0.01, 'max_iterations': 1000, 'regularization': 'ridge', 'lambda_reg': 1.0},
        {'learning_rate': 0.001, 'max_iterations': 2000, 'regularization': 'ridge', 'lambda_reg': 5.0},
        {'learning_rate': 0.05, 'max_iterations': 500, 'regularization': 'lasso', 'lambda_reg': 0.1},
        {'learning_rate': 0.01, 'max_iterations': 1500, 'regularization': 'elastic', 'lambda_reg': 1.0},
    ]
    # results = compare_models(df, configurations)

    # Пример 3: Адаптивная скорость обучения
    print("\n=== Пример 3: Адаптивная скорость обучения ===")
    # adaptive_model, df_features = adaptive_learning_rate_bgd(df, initial_lr=0.1, decay_factor=0.95)

    print("\nВсе примеры готовы к использованию!")
    print("Раскомментируйте нужные строки и подставьте свои данные.")

