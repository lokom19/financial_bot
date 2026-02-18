import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta
import logging
from utils.load_data_method import load_data

# Отключаем предупреждения
warnings.filterwarnings("ignore")
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)



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

    # Целевая переменная - цена закрытия следующего дня
    df_features['next_close'] = df_features['close'].shift(-1)

    # Удаляем временные колонки
    df_features = df_features.drop(['prev_close', 'tr'], axis=1)

    # Удаляем строки с NaN значениями и бесконечностями
    df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()

    return df_features


# Модель Prophet для прогнозирования цен акций
class ProphetTradeModel:
    def __init__(self, params=None):
        # Параметры по умолчанию
        self.default_params = {
            'changepoint_prior_scale': 0.05,  # Гибкость тренда
            'seasonality_prior_scale': 10.0,  # Сила сезонных компонентов
            'holidays_prior_scale': 10.0,  # Влияние праздников
            'seasonality_mode': 'multiplicative',  # Для финансовых рядов часто лучше multiplicative
            'daily_seasonality': False,  # Автоматическое определение
            'weekly_seasonality': True,  # Еженедельные эффекты часто важны
            'yearly_seasonality': True  # Годовые эффекты могут быть важны
        }

        # Обновляем параметры, если они предоставлены
        self.params = self.default_params.copy()
        if params is not None:
            self.params.update(params)

        self.model = None
        self.scaler = RobustScaler()
        self.last_price = None
        self.feature_importances = None
        self.data_quality_checked = False
        self.forecast = None
        self.train_end_date = None
        self.exogenous_features = []

    def check_data_quality(self, df):
        """Проверяет качество данных и выполняет предобработку"""
        # Проверяем временной интервал
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            date_diffs = df['timestamp'].diff().dropna()

            if date_diffs.min() < timedelta(days=1):
                print("ВНИМАНИЕ: В данных обнаружены интервалы меньше дня. Prophet лучше работает с дневными данными.")

            if date_diffs.max() > timedelta(days=7):
                print(
                    f"ВНИМАНИЕ: Обнаружены большие пропуски во временном ряде (до {date_diffs.max().days} дней). Возможны проблемы с прогнозом.")

        # Проверяем выбросы в ценах
        price_mean = df['close'].mean()
        price_std = df['close'].std()
        outliers = df[abs(df['close'] - price_mean) > 3 * price_std]

        if len(outliers) > 0:
            print(f"ВНИМАНИЕ: Обнаружено {len(outliers)} потенциальных выбросов в ценах.")

        # Проверяем логарифмические доходности на нормальность (используем z-оценку как приближение)
        log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        z_scores = (log_returns - log_returns.mean()) / log_returns.std()
        extreme_returns = log_returns[abs(z_scores) > 3]

        if len(extreme_returns) > 0:
            print(f"ВНИМАНИЕ: Обнаружено {len(extreme_returns)} экстремальных дневных изменений цены.")

        self.data_quality_checked = True
        return df

    def prepare_data(self, df, split_ratio=0.8, add_regressor_columns=None):
        """
        Подготавливает данные для Prophet

        Parameters:
        -----------
        df : pandas.DataFrame
            Исходный DataFrame с данными
        split_ratio : float
            Доля данных для обучения (0.0-1.0)
        add_regressor_columns : list
            Список колонок для использования в качестве регрессоров

        Returns:
        --------
        tuple
            (prophet_train_df, prophet_test_df, original_train, original_test)
        """
        # Проверка и предобработка данных
        if not self.data_quality_checked:
            df = self.check_data_quality(df)

        # Для Prophet нужны колонки 'ds' и 'y'
        prophet_df = df.copy()
        prophet_df.rename(columns={'timestamp': 'ds', 'close': 'y'}, inplace=True)

        # Регрессоры (экзогенные переменные)
        self.exogenous_features = []
        if add_regressor_columns is not None:
            for col in add_regressor_columns:
                if col in prophet_df.columns:
                    self.exogenous_features.append(col)

        # Сохраняем самую последнюю цену для прогнозов
        self.last_price = prophet_df['y'].iloc[-1]

        # Разделение на train и test
        split_idx = int(len(prophet_df) * split_ratio)
        train_df = prophet_df.iloc[:split_idx].copy()
        test_df = prophet_df.iloc[split_idx:].copy()

        self.train_end_date = train_df['ds'].iloc[-1]

        # Масштабирование регрессоров для лучшей сходимости
        if self.exogenous_features:
            scaler_data = prophet_df[self.exogenous_features].copy()
            scaled_features = self.scaler.fit_transform(scaler_data)

            for i, col in enumerate(self.exogenous_features):
                prophet_df[col] = scaled_features[:, i]
                train_df[col] = prophet_df[col].iloc[:split_idx]
                test_df[col] = prophet_df[col].iloc[split_idx:]

        # Возвращаем Prophet-форматированные данные и оригинальные для оценки
        return train_df, test_df, df.iloc[:split_idx], df.iloc[split_idx:]

    def train(self, train_df, test_df=None, periods=30, frequency='D'):
        """
        Обучает модель Prophet

        Parameters:
        -----------
        train_df : pandas.DataFrame
            Данные для обучения в формате Prophet (ds, y)
        test_df : pandas.DataFrame
            Данные для тестирования (optional)
        periods : int
            Количество периодов для прогноза
        frequency : str
            Частота прогноза ('D' - дни, 'W' - недели и т.д.)

        Returns:
        --------
        tuple
            (forecast_df, metrics) если есть test_df, иначе только forecast_df
        """
        try:
            # Создаем и настраиваем модель Prophet
            self.model = Prophet(**self.params)

            # Добавляем регрессоры
            for feature in self.exogenous_features:
                self.model.add_regressor(feature)

            # Обучаем модель
            print("Обучение модели Prophet...")
            self.model.fit(train_df)

            # Создаем DataFrame для прогноза
            if test_df is not None:
                # Если есть тестовые данные, используем их период
                future = test_df.copy()
            else:
                # Иначе делаем прогноз на заданное количество периодов вперед
                future = self.model.make_future_dataframe(periods=periods, freq=frequency)

            # Добавляем значения регрессоров в future DataFrame
            if self.exogenous_features:
                for feature in self.exogenous_features:
                    if test_df is not None and feature in test_df.columns:
                        future[feature] = test_df[feature].values
                    else:
                        # Используем последнее значение для прогноза, если регрессоры неизвестны
                        future[feature] = train_df[feature].iloc[-1]

            # Делаем прогноз
            forecast = self.model.predict(future)
            self.forecast = forecast

            # Если есть тестовые данные, вычисляем метрики
            if test_df is not None and 'y' in test_df.columns:
                # Объединяем прогноз с реальными значениями
                evaluation = pd.merge(
                    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                    test_df[['ds', 'y']],
                    on='ds',
                    how='inner'
                )

                # Вычисляем метрики
                metrics = self.calculate_metrics(evaluation['y'], evaluation['yhat'])

                print("\n===== Метрики на тестовой выборке =====")
                self.print_metrics(metrics)

                return forecast, metrics
            else:
                return forecast, None

        except Exception as e:
            print(f"Ошибка при обучении модели Prophet: {e}")
            # Пробуем с упрощенными параметрами
            try:
                print("Повторная попытка с упрощенными параметрами...")
                simple_params = {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 1.0,
                    'daily_seasonality': False,
                    'weekly_seasonality': False,
                    'yearly_seasonality': False
                }

                self.model = Prophet(**simple_params)
                self.model.fit(train_df)

                if test_df is not None:
                    future = test_df.copy()
                else:
                    future = self.model.make_future_dataframe(periods=periods, freq=frequency)

                forecast = self.model.predict(future)
                return forecast, None

            except Exception as e2:
                print(f"Вторая ошибка при обучении: {e2}")
                return None, None

    def calculate_metrics(self, y_true, y_pred):
        """Вычисляет метрики точности прогноза"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)

            # R-squared
            try:
                r2 = r2_score(y_true, y_pred)
            except Exception:
                r2 = float('nan')

            # MAPE
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                if np.isnan(mape) or np.isinf(mape):
                    mape = float('nan')

            # Направление (рост/падение)
            y_true_diff = np.diff(np.append(y_true.iloc[0], y_true))
            y_pred_diff = np.diff(np.append(y_pred.iloc[0], y_pred))
            direction_accuracy = np.mean((y_true_diff > 0) == (y_pred_diff > 0)) * 100

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
        """Выводит метрики в консоль"""
        for name, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                print(f"{name}: Ошибка вычисления")
            else:
                print(f"{name}: {value:.4f}")

    def cross_validate_model(self, df, initial='730 days', period='180 days', horizon='30 days'):
        """
        Проводит кросс-валидацию модели на исторических данных

        Parameters:
        -----------
        df : pandas.DataFrame
            Данные в формате Prophet (ds, y)
        initial : str
            Размер начального обучающего периода
        period : str
            Шаг между последовательными обучающими наборами
        horizon : str
            Горизонт прогнозирования для каждого обучающего набора

        Returns:
        --------
        tuple
            (cv_results, performance)
        """
        try:
            # Создаем и обучаем модель
            model = Prophet(**self.params)

            # Добавляем регрессоры
            for feature in self.exogenous_features:
                model.add_regressor(feature)

            model.fit(df)

            # Выполняем кросс-валидацию
            print(f"Выполняем кросс-валидацию с горизонтом {horizon}...")
            cv_results = cross_validation(
                model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="processes"
            )

            # Вычисляем метрики производительности
            performance = performance_metrics(cv_results)

            print("\n===== Метрики кросс-валидации =====")
            print(f"RMSE: {performance['rmse'].mean():.4f}")
            print(f"MAE: {performance['mae'].mean():.4f}")
            print(f"MAPE: {performance['mape'].mean():.4f}%")

            return cv_results, performance

        except Exception as e:
            print(f"Ошибка при кросс-валидации: {e}")
            return None, None

    def hyperparameter_tuning(self, df, params_grid=None):
        """
        Выполняет подбор гиперпараметров модели

        Parameters:
        -----------
        df : pandas.DataFrame
            Данные в формате Prophet (ds, y)
        params_grid : dict
            Сетка параметров для перебора

        Returns:
        --------
        dict
            Лучшие параметры
        """
        if params_grid is None:
            params_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                'seasonality_mode': ['additive', 'multiplicative']
            }

        # Готовим данные для кросс-валидации
        train_size = int(len(df) * 0.8)
        df_train = df.iloc[:train_size].reset_index(drop=True)

        best_rmse = float('inf')
        best_params = None

        print("Начинаем подбор гиперпараметров...")

        # Перебираем комбинации параметров
        for changepoint_prior_scale in params_grid['changepoint_prior_scale']:
            for seasonality_prior_scale in params_grid['seasonality_prior_scale']:
                for seasonality_mode in params_grid['seasonality_mode']:

                    params = {
                        'changepoint_prior_scale': changepoint_prior_scale,
                        'seasonality_prior_scale': seasonality_prior_scale,
                        'seasonality_mode': seasonality_mode
                    }

                    try:
                        # Создаем модель с текущими параметрами
                        model = Prophet(**params)

                        # Добавляем регрессоры
                        for feature in self.exogenous_features:
                            model.add_regressor(feature)

                        model.fit(df_train)

                        # Делаем валидационный набор для оставшихся 20% данных
                        df_val = df.iloc[train_size:].copy()

                        # Прогнозируем
                        future = model.make_future_dataframe(periods=len(df_val), freq='D')

                        # Добавляем значения регрессоров в future DataFrame
                        if self.exogenous_features:
                            for feature in self.exogenous_features:
                                future[feature] = pd.concat([df_train[feature], df_val[feature]]).reset_index(drop=True)

                        forecast = model.predict(future)

                        # Вычисляем RMSE на валидационном наборе
                        val_forecast = forecast.iloc[-len(df_val):].copy()
                        rmse = np.sqrt(mean_squared_error(df_val['y'].values, val_forecast['yhat'].values))

                        print(f"Параметры: {params}, RMSE: {rmse:.4f}")

                        # Обновляем лучшие параметры, если текущие лучше
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = params

                    except Exception as e:
                        print(f"Ошибка для параметров {params}: {e}")
                        continue

        print(f"\nЛучшие параметры: {best_params}, RMSE: {best_rmse:.4f}")
        self.params = best_params
        return best_params

    def predict_next(self, periods=1, frequency='D', include_components=False):
        """
        Прогнозирует цену на заданное количество периодов вперед

        Parameters:
        -----------
        periods : int
            Количество периодов для прогноза
        frequency : str
            Частота прогноза ('D' - дни, 'W' - недели и т.д.)
        include_components : bool
            Включать ли компоненты прогноза (тренд, сезонность и т.д.)

        Returns:
        --------
        dict
            Результаты прогноза
        """
        if self.model is None:
            print("ОШИБКА: Модель не обучена. Запустите train() сначала.")
            return None

        try:
            # Создаем DataFrame для прогноза
            future = self.model.make_future_dataframe(periods=periods, freq=frequency)

            # Добавляем значения регрессоров, если они есть
            if self.exogenous_features:
                for feature in self.exogenous_features:
                    if self.forecast is not None and feature in self.forecast.columns:
                        # Используем последнее известное значение
                        future[feature] = self.forecast[feature].iloc[-1]
                    else:
                        future[feature] = 0.0

            # Делаем прогноз
            forecast = self.model.predict(future)
            next_date = forecast['ds'].iloc[-1]
            next_price = forecast['yhat'].iloc[-1]

            # Получаем верхнюю и нижнюю границы прогноза (доверительный интервал)
            next_price_lower = forecast['yhat_lower'].iloc[-1]
            next_price_upper = forecast['yhat_upper'].iloc[-1]

            # Рассчитываем изменение от последней известной цены
            price_change = next_price - self.last_price
            price_change_pct = (next_price / self.last_price - 1) * 100

            # Определяем торговый сигнал
            if price_change_pct > 1.0:  # Сильный рост
                signal = "BUY"
            elif price_change_pct < -1.0:  # Сильное падение
                signal = "SELL"
            else:  # Небольшое изменение
                signal = "HOLD"

            # Оцениваем уверенность в прогнозе
            # Чем шире доверительный интервал относительно цены, тем ниже уверенность
            confidence_range = (next_price_upper - next_price_lower) / next_price
            if confidence_range < 0.05:  # Узкий интервал: высокая уверенность
                confidence = 0.9
            elif confidence_range < 0.10:
                confidence = 0.7
            elif confidence_range < 0.20:
                confidence = 0.5
            else:  # Широкий интервал: низкая уверенность
                confidence = 0.3

            # Формируем результат
            result = {
                'current_price': float(self.last_price),
                'current_date': self.train_end_date.strftime('%Y-%m-%d') if self.train_end_date else "N/A",
                'forecast_date': next_date.strftime('%Y-%m-%d'),
                'predicted_price': float(next_price),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'signal': signal,
                'confidence': float(confidence),
                'lower_bound': float(next_price_lower),
                'upper_bound': float(next_price_upper)
            }

            # Добавляем компоненты, если запрошено
            if include_components and 'trend' in forecast.columns:
                components = {
                    'trend': float(forecast['trend'].iloc[-1]),
                    'yearly': float(forecast['yearly'].iloc[-1]) if 'yearly' in forecast.columns else 0.0,
                    'weekly': float(forecast['weekly'].iloc[-1]) if 'weekly' in forecast.columns else 0.0,
                    'daily': float(forecast['daily'].iloc[-1]) if 'daily' in forecast.columns else 0.0
                }
                result['components'] = components

            return result

        except Exception as e:
            print(f"Ошибка при прогнозировании: {e}")
            # Возвращаем базовый прогноз в случае ошибки
            return {
                'current_price': float(self.last_price) if self.last_price is not None else 0.0,
                'predicted_price': float(self.last_price) if self.last_price is not None else 0.0,
                'price_change_pct': 0.0,
                'signal': 'NEUTRAL',
                'confidence': 0.0
            }

    def get_model_components(self):
        """
        Возвращает информацию о компонентах модели Prophet

        Returns:
        --------
        dict
            Информация о компонентах модели
        """
        if self.model is None:
            print("ОШИБКА: Модель не обучена. Запустите train() сначала.")
            return None

        components = {
            'seasonalities': list(self.model.seasonalities.keys()),
            'changepoints': self.model.changepoints.tolist() if hasattr(self.model, 'changepoints') else [],
            'n_changepoints': self.model.n_changepoints if hasattr(self.model, 'n_changepoints') else 0,
            'seasonality_mode': self.model.seasonality_mode if hasattr(self.model, 'seasonality_mode') else 'additive',
            'growth': self.model.growth if hasattr(self.model, 'growth') else 'linear'
        }

        return components


# Основная функция
def main(db_path):
    # Загружаем данные
    print("Загрузка данных...")

    # table_name = os.path.basename(db_path)[:-3]  # Используем имя файла без .db
    # df = load_data(db_path, table_name)
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

    # Создаем и обучаем модель Prophet
    print("\nОбучение модели Prophet...")
    model = ProphetTradeModel()

    # Подготавливаем данные для Prophet
    prophet_train, prophet_test, orig_train, orig_test = model.prepare_data(
        df_features,
        split_ratio=0.8,
        add_regressor_columns=['volume_log', 'rsi_14', 'volatility_10']  # Добавляем дополнительные регрессоры
    )

    # Обучаем модель
    forecast, test_metrics = model.train(prophet_train, prophet_test)

    if forecast is None:
        print("ОШИБКА: Не удалось обучить модель Prophet.")
        return None, None

    # Выводим результаты кросс-валидации
    print("\nВыполняем кросс-валидацию модели...")
    model.cross_validate_model(prophet_train, initial='90 days', period='30 days', horizon='14 days')

    # Выводим последние 5 фактических и прогнозных значений
    print("\nПоследние 5 прогнозов:")
    if prophet_test is not None and len(prophet_test) > 0:
        last_idx = min(5, len(prophet_test))
        for i in range(last_idx):
            idx = len(prophet_test) - last_idx + i
            actual_date = prophet_test['ds'].iloc[idx]
            actual_price = prophet_test['y'].iloc[idx]

            # Находим соответствующий прогноз
            forecast_idx = forecast[forecast['ds'] == actual_date].index
            if len(forecast_idx) > 0:
                pred_price = forecast.loc[forecast_idx[0], 'yhat']
                error_pct = (pred_price - actual_price) / actual_price * 100
                print(f"Дата: {actual_date.strftime('%Y-%m-%d')}, Реальная цена: {actual_price:.4f}, "
                      f"Прогноз: {pred_price:.4f}, Ошибка: {error_pct:.2f}%")

    # Прогноз на следующий временной интервал
    print("\nПрогноз на следующий временной интервал:")
    next_prediction = model.predict_next(periods=1, include_components=True)

    # print(f"Текущая цена ({next_prediction['current_date']}): {next_prediction['current_price']:.4f}")
    print(f"Текущая цена: {next_prediction['current_price']:.4f}")
    print(f"Прогнозируемая цена: {next_prediction['predicted_price']:.4f}")
    print(f"Ожидаемое изменение: {next_prediction['price_change_pct']:.2f}%")
    print(f"Доверительный интервал: [{next_prediction['lower_bound']:.4f}, {next_prediction['upper_bound']:.4f}]")
    # print(f"Торговый сигнал: {next_prediction['signal']} (уверенность: {next_prediction['confidence']:.2f})")
    print(f"Торговый сигнал: {next_prediction['signal']}")

    if 'components' in next_prediction:
        print("\nКомпоненты прогноза:")
        for comp_name, comp_value in next_prediction['components'].items():
            print(f"{comp_name}: {comp_value:.4f}")

    # Ретроспективная оценка торговых сигналов на тестовой выборке
    if prophet_test is not None and len(prophet_test) > 0:
        print("\nРетроспективная оценка торговых сигналов:")

        # Объединяем прогноз с фактическими данными
        eval_df = pd.merge(
            forecast[['ds', 'yhat']],
            prophet_test[['ds', 'y']],
            on='ds',
            how='inner'
        )

        eval_df['y_shifted'] = eval_df['y'].shift(1).fillna(eval_df['y'].iloc[0])
        eval_df['signal'] = np.sign(eval_df['yhat'] - eval_df['y_shifted'])
        eval_df['actual_return'] = eval_df['y'].pct_change().fillna(0)
        eval_df['strategy_return'] = eval_df['signal'].shift(1) * eval_df['actual_return']
        eval_df['cumulative_return'] = (1 + eval_df['strategy_return']).cumprod() - 1

        total_trades = np.sum(np.abs(np.diff(eval_df['signal'])) > 0) + 1
        profitable_trades = np.sum(eval_df['strategy_return'] > 0)
        profit_sum = np.sum(eval_df['strategy_return'][eval_df['strategy_return'] > 0])
        loss_sum = abs(np.sum(eval_df['strategy_return'][eval_df['strategy_return'] < 0]))
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

        print(f"Всего сделок: {total_trades}")
        print(f"Прибыльных сделок: {profitable_trades} ({profitable_trades / total_trades * 100:.2f}% от общего числа)")
        print(f"Общая доходность: {eval_df['cumulative_return'].iloc[-1] * 100:.2f}%")
        print(f"Коэффициент прибыли (Profit Factor): {profit_factor:.2f}")

    return model, df_features


if __name__ == "__main__":
    main("BBG000F6YPH8")
