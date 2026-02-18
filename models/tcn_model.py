import os
import sys

import pandas as pd
import numpy as np
import sqlite3
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Add, Activation, BatchNormalization, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from utils.load_data_method import load_data


# Отключаем все RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Функция для загрузки данных из SQLite


# Функция для создания технических индикаторов
def create_features(df):
    # Создаем копию DataFrame
    df_features = df.copy()

    # Простые скользящие средние
    df_features['sma_5'] = df['close'].rolling(window=5).mean()
    df_features['sma_10'] = df['close'].rolling(window=10).mean()
    df_features['sma_20'] = df['close'].rolling(window=20).mean()

    # Экспоненциальные скользящие средние
    df_features['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df_features['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df_features['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Разница между ценой закрытия и скользящими средними
    df_features['close_minus_sma_5'] = df['close'] - df_features['sma_5']
    df_features['close_minus_sma_10'] = df['close'] - df_features['sma_10']

    # Относительные разницы
    df_features['close_rel_sma_5'] = df['close'] / df_features['sma_5'] - 1
    df_features['close_rel_sma_10'] = df['close'] / df_features['sma_10'] - 1

    # Логарифмированный объём для уменьшения разброса
    df_features['volume_log'] = np.log1p(df['volume'])

    # Объём - простые индикаторы
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
    df_features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    # Избегаем деление на ноль
    rs = np.where(loss != 0, gain / loss, 100)
    df_features['rsi_14'] = 100 - (100 / (1 + rs))

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

    # MACD (Moving Average Convergence Divergence)
    df_features['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26,
                                                                                               adjust=False).mean()
    df_features['macd_signal'] = df_features['macd_line'].ewm(span=9, adjust=False).mean()
    df_features['macd_histogram'] = df_features['macd_line'] - df_features['macd_signal']

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


def residual_block(x, filters, kernel_size=3, dilation_rate=1, dropout_rate=0.1):
    """
    Блок остаточной связи для временной сверточной сети

    Параметры:
    ----------
    x : Tensor
        Входной тензор
    filters : int
        Количество фильтров в сверточном слое
    kernel_size : int
        Размер ядра свертки
    dilation_rate : int
        Скорость расширения (dilation rate)
    dropout_rate : float
        Вероятность dropout

    Возвращает:
    -----------
    Tensor
        Выходной тензор
    """
    # Сохраняем вход для добавления к выходу (skip connection)
    input_x = x
    input_shape = tf.keras.backend.int_shape(input_x)

    # Первая свертка с расширением
    x = Conv1D(filters=filters, kernel_size=kernel_size,
               dilation_rate=dilation_rate, padding='causal',
               activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    # Вторая свертка с расширением
    x = Conv1D(filters=filters, kernel_size=kernel_size,
               dilation_rate=dilation_rate, padding='causal',
               activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    # Если размерности не совпадают, применяем проекцию к входу
    if input_shape[-1] != filters:
        input_x = Conv1D(filters=filters, kernel_size=1, padding='same')(input_x)

    # Суммируем вход и выход (skip connection)
    x = Add()([x, input_x])
    return x


# Модель TCN (Temporal Convolutional Network)
class TCNModel:
    def __init__(self, params=None):
        # Параметры по умолчанию
        # НУЖНО ИГРАТЬСЯ С ЭТИМИ ПАРАМЕТРАМИ
        # self.default_params = {
        #     'seq_length': 50,  # Длина последовательности для входных данных
        #     'filters': 256,  # Количество фильтров в сверточных слоях
        #     'kernel_size': 7,  # Размер ядра свертки
        #     'num_blocks': 8,  # Количество блоков TCN
        #     'dropout_rate': 0.1,  # Вероятность dropout
        #     'learning_rate': 0.0001,  # Скорость обучения
        #     'epochs': 100,  # Максимальное количество эпох
        #     'batch_size': 128,  # Размер батча
        #     'patience': 25  # Количество эпох для early stopping
        # }
        self.default_params = {
            'seq_length': 20,  # Длина последовательности для входных данных
            'filters': 64,  # Количество фильтров в сверточных слоях
            'kernel_size': 3,  # Размер ядра свертки
            'num_blocks': 4,  # Количество блоков TCN
            'dropout_rate': 0.2,  # Вероятность dropout
            'learning_rate': 0.001,  # Скорость обучения
            'epochs': 100,  # Максимальное количество эпох
            'batch_size': 32,  # Размер батча
            'patience': 15  # Количество эпох для early stopping
        }

        # Обновляем параметры, если они предоставлены
        self.params = self.default_params.copy()
        if params is not None:
            self.params.update(params)

        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.last_price = None
        self.feature_importances = None
        self.data_quality_checked = False
        self.history = None

    def check_data_quality(self, features):
        """Проверка качества данных и выявление потенциальных проблем"""
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
        """Подготовка данных для TCN модели"""
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
        print(features.describe().loc[['min', 'max', 'mean', 'std']].T.head())

        # Проверка наличия NaN значений после замены
        if features.isna().any().any():
            print("ВНИМАНИЕ: В данных остались NaN значения после обработки.")
            features = features.fillna(features.median())  # Заполняем оставшиеся NaN медианой

        # Сохраняем оригинальные данные
        X_original = features.copy()

        # Стандартизируем признаки с осторожностью
        try:
            X_scaled = self.scaler.fit_transform(features)
        except Exception as e:
            print(f"Ошибка при масштабировании данных: {e}")
            print("Применяем более простое масштабирование...")
            # Применяем более простое масштабирование без использования скалера
            X_scaled = np.zeros(features.shape)
            for i, col in enumerate(features.columns):
                col_mean = features[col].mean()
                col_std = features[col].std()
                if col_std > 1e-10:  # Проверка на нулевое стандартное отклонение
                    X_scaled[:, i] = (features[col] - col_mean) / col_std
                else:
                    X_scaled[:, i] = 0  # Если стандартное отклонение близко к нулю, устанавливаем колонку в нули

        # Проверка на NaN и Inf после масштабирования и замена проблемных значений
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print("ВНИМАНИЕ: После масштабирования обнаружены NaN или Inf значения. Заменяем их на 0.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Дополнительная проверка диапазона значений после масштабирования
        if np.max(np.abs(X_scaled)) > 1e6:
            print("ВНИМАНИЕ: Обнаружены очень большие значения после масштабирования. Ограничиваем их.")
            X_scaled = np.clip(X_scaled, -1e6, 1e6)

        return X_original, X_scaled, y, timestamps

    def create_sequences(self, X, y=None):
        """Создание последовательностей для TCN"""
        seq_length = self.params['seq_length']
        X_seq = []
        y_seq = []

        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            if y is not None:
                y_seq.append(y.iloc[i + seq_length])

        X_seq = np.array(X_seq)

        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        else:
            return X_seq

    def build_tcn_model(self, input_shape):
        """Построение модели TCN"""
        filters = self.params['filters']
        kernel_size = self.params['kernel_size']
        num_blocks = self.params['num_blocks']
        dropout_rate = self.params['dropout_rate']

        input_layer = Input(shape=input_shape)
        x = input_layer

        # Инициализируем dilation_rate для первого блока
        dilation_rate = 1

        # Строим блоки TCN с возрастающим dilation_rate
        for i in range(num_blocks):
            x = residual_block(x, filters, kernel_size, dilation_rate, dropout_rate)
            # Увеличиваем dilation_rate экспоненциально
            dilation_rate *= 2

        # Применяем операцию Flatten для получения одномерного выхода
        x = Flatten()(x)

        # Добавляем полносвязные слои для предсказания
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        output_layer = Dense(1, activation='linear')(x)

        # Создаем модель
        model = Model(inputs=input_layer, outputs=output_layer)

        # Компилируем модель
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse')

        print(model.summary())
        return model

    def train(self, X_original, y, timestamps=None, X_scaled=None, df_features=None):
        """Обучение модели TCN"""
        if np.isnan(X_original.values).any() or np.isinf(X_original.values).any():
            print("КРИТИЧЕСКАЯ ОШИБКА: В данных обнаружены NaN или Inf.")
            X_original = X_original.replace([np.inf, -np.inf], np.nan).fillna(X_original.median())

        # Используем масштабированные данные для TCN
        if X_scaled is None:
            X_scaled = self.scaler.fit_transform(X_original)

        # Создаем последовательности для TCN
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        # Train/test split - соблюдаем хронологический порядок
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # Также получаем оригинальные данные для вычисления метрик
        # Обратите внимание на смещение индексов из-за создания последовательностей
        offset = self.params['seq_length']
        X_train_orig = X_original.iloc[offset:split_idx + offset]
        X_test_orig = X_original.iloc[split_idx + offset:len(X_seq) + offset]
        y_train_orig = y.iloc[offset:split_idx + offset]
        y_test_orig = y.iloc[split_idx + offset:len(X_seq) + offset]

        # Timestamp для тестового набора
        if timestamps is not None:
            timestamps_test = timestamps.iloc[split_idx + offset:len(X_seq) + offset]

        # Строим и обучаем модель TCN
        try:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_tcn_model(input_shape)

            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.params['patience'],
                restore_best_weights=True,
                verbose=1
            )

            # Checkpoint для сохранения лучшей модели
            checkpoint_filepath = 'best_tcn_model.h5'
            model_checkpoint = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_best_only=False,
                monitor='val_loss',
                verbose=1,

            )

            # Адаптивное изменение скорости обучения
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )

            # Обучаем модель
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_split=0.2,
                callbacks=[early_stopping, model_checkpoint, lr_scheduler],
                verbose=1
            )

        except Exception as e:
            print(f"Ошибка при обучении TCN модели: {e}")

            # Пробуем упрощенную модель в случае ошибки
            try:
                print("Попытка использования упрощенной модели...")
                self.params['filters'] = 32
                self.params['num_blocks'] = 2
                self.params['dropout_rate'] = 0.1

                self.model = Sequential([
                    Conv1D(filters=32, kernel_size=3, padding='causal', input_shape=input_shape),
                    Activation('relu'),
                    Dropout(0.1),
                    Flatten(),
                    Dense(1)
                ])

                self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

                self.history = self.model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1
                )
            except Exception as e2:
                print(f"Вторая ошибка: {e2}")
                return None, None, None, None

        # Прогнозы
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        # Проверяем на наличие NaN и Inf в прогнозах
        if np.isnan(train_preds).any() or np.isinf(train_preds).any():
            print("ВНИМАНИЕ: NaN или Inf в прогнозах на обучающих данных.")
            train_preds = np.nan_to_num(train_preds)

        if np.isnan(test_preds).any() or np.isinf(test_preds).any():
            print("ВНИМАНИЕ: NaN или Inf в прогнозах на тестовых данных.")
            test_preds = np.nan_to_num(test_preds)

        # Вычисляем метрики
        train_metrics = self.calculate_metrics(y_train, train_preds.flatten())
        test_metrics = self.calculate_metrics(y_test, test_preds.flatten())

        print("\n===== Метрики на обучающей выборке =====")
        self.print_metrics(train_metrics)

        print("\n===== Метрики на тестовой выборке =====")
        self.print_metrics(test_metrics)

        # Сохраняем последнюю цену для будущих прогнозов
        if np.isnan(y.iloc[-1]) and df_features is not None:
            # Берем последнюю фактическую цену закрытия вместо NaN
            self.last_price = df_features.iloc[-1]['close']
        else:
            self.last_price = y.iloc[-1]


        # Создаем и заполняем feature_importances для совместимости с другими моделями
        # TCN не имеет прямого механизма определения важности признаков как в случае с лесами
        # Поэтому мы имитируем его методом пермутационной важности
        if X_test_orig is not None and len(X_test_orig) > 100:
            try:
                self.compute_permutation_importance(X_test_orig, y_test_orig)
            except Exception as e:
                print(f"Ошибка при вычислении важности признаков: {e}")
                # Создаем заглушку
                self.feature_importances = pd.DataFrame({
                    'Feature': self.feature_columns,
                    'Importance': np.ones(len(self.feature_columns))
                })

        return X_test_orig, y_test_orig, test_preds.flatten(), test_metrics

    def compute_permutation_importance(self, X_test, y_test, n_repeats=10):
        """
        Вычисление важности признаков методом пермутации

        Идея: для каждого признака случайно перемешиваем его значения и смотрим,
        насколько ухудшается качество предсказания
        """
        # Создаем последовательности для предсказания
        X_scaled = self.scaler.transform(X_test)
        X_seq = self.create_sequences(X_scaled)

        # Базовая ошибка
        base_preds = self.model.predict(X_seq)
        base_error = mean_squared_error(y_test.iloc[len(X_test) - len(base_preds):], base_preds)

        importance = np.zeros(len(self.feature_columns))

        # Для каждого признака
        for i, col in enumerate(self.feature_columns):
            error_increases = []

            # Проводим n_repeats экспериментов с перемешиванием
            for _ in range(n_repeats):
                # Копируем данные
                X_permuted = X_test.copy()
                # Перемешиваем значения признака
                X_permuted[col] = np.random.permutation(X_permuted[col].values)

                # Масштабируем и создаем последовательности
                X_scaled_perm = self.scaler.transform(X_permuted)
                X_seq_perm = self.create_sequences(X_scaled_perm)

                # Предсказываем с перемешанным признаком
                perm_preds = self.model.predict(X_seq_perm)

                # Вычисляем, насколько увеличилась ошибка
                perm_error = mean_squared_error(y_test.iloc[len(X_test) - len(perm_preds):], perm_preds)
                error_increases.append(perm_error - base_error)

            # Среднее увеличение ошибки - это важность признака
            importance[i] = np.mean(error_increases)

        # Нормализуем важности, чтобы сумма была равна 1
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)

        # Создаем DataFrame с результатами
        self.feature_importances = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        return self.feature_importances

    def calculate_metrics(self, y_true, y_pred):
        """Расчет метрик качества модели"""
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

            # Безопасное вычисление R² с обработкой исключений
            try:
                r2 = r2_score(y_true_filtered, y_pred_filtered)
            except Exception as e:
                print(f"Ошибка при вычислении R²: {e}")
                r2 = float('nan')

            # Безопасное вычисление MAPE с обработкой деления на ноль
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_raw = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)
                mape_raw = np.where(np.isinf(mape_raw) | np.isnan(mape_raw), 0, mape_raw)
                mape = np.mean(mape_raw) * 100

            # Направление цены (бинарный классификатор)
            if len(y_true_filtered) > 1:
                # Используем первое значение вместо .iloc[0]
                direction_true = np.diff(np.append(y_true_filtered[0], y_true_filtered)) > 0
                direction_pred = np.diff(np.append(y_true_filtered[0], y_pred_filtered)) > 0
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
            return {k: float('nan') for k in ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE', 'Direction Accuracy']}


    def print_metrics(self, metrics):
        """Вывод метрик в консоль"""
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
        plt.title('Сравнение фактических и предсказанных цен')
        plt.xlabel('Время')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)

        # Сохраняем график
        timestamp_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(f'predictions_tcn_{timestamp_now}.png')
        print(f"График сохранен в файл predictions_tcn_{timestamp_now}.png")

        # Дополнительный график для анализа ошибок
        plt.figure(figsize=(12, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title('Фактические vs Предсказанные значения')
        plt.xlabel('Фактические значения')
        plt.ylabel('Предсказанные значения')
        plt.grid(True)
        plt.savefig(f'residuals_tcn_{timestamp_now}.png')
        print(f"График остатков сохранен в файл residuals_tcn_{timestamp_now}.png")

        # График важности признаков
        if self.feature_importances is not None:
            plt.figure(figsize=(12, 10))
            top_features = self.feature_importances.head(15)
            plt.barh(top_features['Feature'], top_features['Importance'])
            plt.title('Важность признаков')
            plt.xlabel('Важность')
            plt.gca().invert_yaxis()  # Инвертируем ось, чтобы самый важный признак был вверху
            plt.tight_layout()
            plt.savefig(f'feature_importance_tcn_{timestamp_now}.png')
            print(f"График важности признаков сохранен в файл feature_importance_tcn_{timestamp_now}.png")

        plt.close('all')  # Закрываем все окна графиков

    def predict_next(self, current_data):
        """Предсказание следующего значения цены"""
        try:
            if self.model is None:
                print("ОШИБКА: Модель не обучена. Запустите метод train() сначала.")
                return None

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
            scaled_features = np.clip(scaled_features, -1e6,
                                      1e6)
            # Создаем последовательность необходимой длины
            seq_length = self.params['seq_length']

            # Если есть достаточно данных
            if len(scaled_features) >= seq_length:
                input_sequence = scaled_features[-seq_length:].reshape(1, seq_length, scaled_features.shape[1])
            else:
                # Если недостаточно данных, дополняем последовательность
                print(f"ВНИМАНИЕ: Недостаточно данных для последовательности длиной {seq_length}. Дополняем.")
                padding_size = seq_length - len(scaled_features)
                # Дублируем первую строку нужное количество раз
                padding = np.tile(scaled_features[0], (padding_size, 1))
                padded_sequence = np.vstack([padding, scaled_features])
                input_sequence = padded_sequence.reshape(1, seq_length, scaled_features.shape[1])

            # Делаем прогноз
            predicted_price = self.model.predict(input_sequence)[0][0]

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

            # Простая метрика доверия на основе потерь при валидации
            confidence = 0.0
            if self.history is not None:
                val_losses = self.history.history['val_loss']
                last_val_loss = val_losses[-1]
                # Преобразуем потери в показатель доверия (чем меньше потери, тем выше доверие)
                confidence = max(0, min(1, 1 / (1 + last_val_loss)))

            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change,
                'signal': 'BUY' if price_change > 0 else 'SELL',
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


# Основная функция
def main(db_path):
    # Загружаем данные
    print("Загрузка данных...")

    df = load_data(db_path)
    df = df.drop(["figi"], axis=1)

    print(
        f"Загружено {len(df)} записей за период с {df['timestamp'].min() if not df.empty else 'N/A'} по {df['timestamp'].max() if not df.empty else 'N/A'}")

    # Проверка на пустой DataFrame
    if df.empty:
        print(f"ОШИБКА: Файл {db_path} не содержит данных. Пропускаем обработку.")
        return None, None

    # Создаем признаки
    print("\nСоздание признаков...")
    df_features = create_features(df)
    print(f"Создано признаков: {len(df_features.columns) - 2}")  # -2 для timestamp и next_close

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

    # Создаем и обучаем модель
    print("\nОбучение TCN модели...")
    model = TCNModel()  # Используем параметры по умолчанию
    X_original, X_scaled, y, timestamps = model.prepare_data(df_features)

    # Проверка на пустые данные после подготовки
    if X_original.shape[0] == 0:
        print(f"ОШИБКА: После подготовки данных для {db_path} не осталось записей для обучения.")
        return None, None

    # Обучаем модель и получаем результаты
    X_test, y_test, predictions, test_metrics = model.train(X_original, y, timestamps, X_scaled, df_features)

    # Выводим важность признаков
    print("\nВажность признаков (топ-10):")
    print(model.get_feature_importance(10))

    # Выводим кривые обучения
    # model.plot_learning_curves()

    # Выводим последние 5 предсказаний и реальные значения
    print("\nПоследние 5 предсказаний:")
    if X_test is not None and y_test is not None and len(predictions) > 0:
        last_indices = range(len(y_test) - min(5, len(y_test)), len(y_test))
        for i in last_indices:
            real_price = y_test.iloc[i]
            pred_price = predictions[i]
            error_pct = (pred_price - real_price) / real_price * 100 if not np.isnan(
                real_price) and real_price != 0 else float('nan')
            print(
                f"Реальная цена next_close: {real_price:.4f}, Предсказанная цена next_close: {pred_price:.4f}, Ошибка: {error_pct:.2f}%")

    # Предсказываем цену на следующий временной интервал
    print("\nПрогноз на следующий временной интервал:")

    # Берем последнюю строку из оригинального датасета
    latest_row = df.iloc[-1:].copy()

    # Создаем признаки только для этой строки
    latest_features = create_features(
        pd.concat([df.iloc[-30:].iloc[:-1], latest_row]))  # Берем предыдущие 30 дней для расчета индикаторов
    latest_features = latest_features.iloc[-1:].copy()  # Оставляем только последнюю строку с рассчитанными признаками

    # Если в latest_features есть NaN в next_close, заменяем его на 0 или другое значение
    if 'next_close' in latest_features.columns and latest_features['next_close'].isna().any():
        latest_features['next_close'] = 0  # или другое подходящее значение

    prediction = model.predict_next(latest_features)

    print(f"Текущая цена: {prediction['current_price']:.4f}")
    print(f"Прогнозируемая цена: {prediction['predicted_price']:.4f}")
    print(f"Ожидаемое изменение: {prediction['price_change_pct']:.2f}%")
    print(f"Торговый сигнал: {prediction['signal']}")
    print(f"Уверенность в прогнозе: {prediction['confidence']:.2f}")

    # Оценка эффективности торговых сигналов
    print("\nРетроспективная оценка торговых сигналов:")
    if y_test is not None and len(predictions) > 0:
        y_test_adjusted = y_test.iloc[len(y_test) - len(predictions):]
        y_test_shifted = y_test_adjusted.shift(1).fillna(y_test_adjusted.iloc[0])
        signals = np.sign(np.clip(predictions - y_test_shifted, -1e10, 1e10))
        actual_returns = y_test_adjusted.pct_change().fillna(0)

        if len(signals) > 1 and len(actual_returns) > 1:
            strategy_returns = signals[:-1] * actual_returns[1:].values
            cumulative_returns = (1 + strategy_returns).cumprod() - 1

            total_trades = np.sum(np.abs(np.diff(signals)) > 0) + 1
            profitable_trades = np.sum(strategy_returns > 0)

            profit_sum = np.sum(strategy_returns[strategy_returns > 0])
            loss_sum = abs(np.sum(strategy_returns[strategy_returns < 0]))
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

            print(f"Всего сделок: {total_trades}")
            print(
                f"Прибыльных сделок: {profitable_trades} ({profitable_trades / total_trades * 100:.2f}% от общего числа)")
            print(f"Общая доходность: {cumulative_returns.iloc[-1] * 100:.2f}%")
            print(f"Коэффициент прибыли (Profit Factor): {profit_factor:.2f}")
        else:
            print("Недостаточно данных для ретроспективной оценки.")
    else:
        print("Недостаточно данных для ретроспективной оценки.")

    # Визуализация предсказаний
    if X_test is not None and y_test is not None and len(predictions) > 0 and timestamps is not None:
        timestamps_test = timestamps.iloc[len(timestamps) - len(y_test):].iloc[len(y_test) - len(predictions):]
        # model.visualize_predictions(timestamps_test, y_test.iloc[len(y_test) - len(predictions):], predictions)

    return model, df_features


if __name__ == "__main__":
    main("BBG000F6YPH8")