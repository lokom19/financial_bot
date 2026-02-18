import os
import re

from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import requests
import logging
from sqlalchemy import text
from datetime import datetime, timedelta
from typing import Optional

load_dotenv()

# Pattern for valid table names (alphanumeric, underscores, and some special chars)
VALID_TABLE_NAME_PATTERN = re.compile(r'^[A-Za-z0-9_\-\.]+$')


def validate_table_name(table_name: str) -> bool:
    """
    Validate table name to prevent SQL injection.

    Args:
        table_name: Name to validate

    Returns:
        True if valid, False otherwise
    """
    if not table_name:
        return False
    if len(table_name) > 255:
        return False
    return bool(VALID_TABLE_NAME_PATTERN.match(table_name))

DB_HOST = os.getenv("DB_HOST") # "localhost"  # Change this to your database host
DB_PORT = os.getenv("DB_PORT")  # "5432"  # Default PostgreSQL port
DB_NAME = os.getenv("DB_NAME")    # "postgres"  # Change to your database name
DB_USER = os.getenv("DB_USER")      # "postgres"  # Change to your username
DB_PASSWORD = os.getenv("DB_PASSWORD")    # "mysecretpassword"  # Change to your password
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

class FearGreedDataEnhancer:
    """Класс для добавления Fear and Greed Index к финансовым данным"""

    def __init__(self):
        self.api_url = "https://api.alternative.me/fng/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = {}  # Кэш для избежания повторных запросов

    def get_fear_greed_data(self, days_limit: int = 0) -> Optional[dict]:
        """
        Получает данные Fear and Greed Index

        Args:
            days_limit: Количество дней (0 = все доступные данные)

        Returns:
            Словарь с данными или None при ошибке
        """
        try:
            url = f"{self.api_url}?limit={days_limit}"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            else:
                logging.warning(f"Fear and Greed API returned status {response.status_code}")
                return None

        except Exception as e:
            logging.error(f"Ошибка при получении Fear and Greed данных: {e}")
            return None

    def create_fear_greed_dataframe(self, limit_days: int = 365) -> pd.DataFrame:
        """
        Создает DataFrame с Fear and Greed данными

        Args:
            limit_days: Ограничение по дням

        Returns:
            DataFrame с колонками: date, fear_greed_index, fg_classification
        """
        fg_data = self.get_fear_greed_data(limit_days)

        if not fg_data:
            logging.warning("Не удалось получить Fear and Greed данные, создаем пустой DataFrame")
            return pd.DataFrame(columns=['date', 'fear_greed_index', 'fg_classification'])

        processed_data = []
        for item in fg_data:
            try:
                timestamp = int(item['timestamp'])
                date = datetime.fromtimestamp(timestamp).date()

                processed_data.append({
                    'date': date,
                    'fear_greed_index': int(item['value']),
                    'fg_classification': item['value_classification']
                })
            except Exception as e:
                logging.warning(f"Ошибка обработки Fear and Greed записи: {e}")
                continue

        df_fg = pd.DataFrame(processed_data)
        df_fg['date'] = pd.to_datetime(df_fg['date'])

        logging.info(f"Создан Fear and Greed DataFrame с {len(df_fg)} записями")
        return df_fg

    def generate_synthetic_fear_greed(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Генерирует синтетические данные Fear and Greed для недостающих дат

        Args:
            start_date: Начальная дата
            end_date: Конечная дата

        Returns:
            DataFrame с синтетическими данными
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Простой алгоритм генерации на основе случайных блужданий
        np.random.seed(42)
        base_value = 50
        synthetic_data = []

        for date in date_range:
            # Добавляем немного случайности и тренда
            base_value += np.random.normal(0, 5)
            base_value = max(0, min(100, base_value))  # Ограничиваем 0-100

            # Классификация
            if base_value <= 25:
                classification = "Extreme Fear"
            elif base_value <= 45:
                classification = "Fear"
            elif base_value <= 55:
                classification = "Neutral"
            elif base_value <= 75:
                classification = "Greed"
            else:
                classification = "Extreme Greed"

            synthetic_data.append({
                'date': date,
                'fear_greed_index': int(base_value),
                'fg_classification': classification
            })

        return pd.DataFrame(synthetic_data)


# Глобальный экземпляр для переиспользования
fear_greed_enhancer = FearGreedDataEnhancer()


def load_data(ticker_name: str, add_fear_greed: bool = True, engine=engine) -> pd.DataFrame:
    """
    Загружает данные тикера и опционально добавляет Fear and Greed Index

    Args:
        ticker_name: Название тикера
        add_fear_greed: Добавлять ли Fear and Greed Index
        engine: SQLAlchemy engine для подключения к БД

    Returns:
        DataFrame с данными тикера и Fear and Greed Index

    Raises:
        ValueError: If ticker_name fails validation
    """
    # Validate ticker name to prevent SQL injection
    if not validate_table_name(ticker_name):
        raise ValueError(
            f"Invalid ticker name: {ticker_name}. "
            "Only alphanumeric characters, underscores, hyphens, and dots allowed."
        )

    try:
        # Escape double quotes in table name for safety
        safe_ticker = ticker_name.replace('"', '""')

        # Form the query with validated and escaped table name
        query = text(f"""
            SELECT * FROM all_dfs."{safe_ticker}"
            ORDER BY timestamp
        """)

        # Выполнение запроса с помощью pandas
        df = pd.read_sql_query(query, engine)

        # Преобразование timestamp в datetime, если столбец существует
        if 'timestamp' in df.columns and not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        logging.info(f"Успешно загружены данные для {ticker_name}, {len(df)} строк")

        # Добавляем Fear and Greed Index, если запрошено
        if add_fear_greed and not df.empty:
            df = add_fear_greed_index(df)

        return df

    except Exception as e:
        logging.error(f"Ошибка при загрузке данных для {ticker_name}: {str(e)}")
        return pd.DataFrame()


def add_fear_greed_index(df: pd.DataFrame, date_column: str = 'timestamp') -> pd.DataFrame:
    """
    Добавляет Fear and Greed Index к существующему DataFrame

    Args:
        df: Исходный DataFrame с финансовыми данными
        date_column: Название колонки с датами

    Returns:
        DataFrame с добавленными колонками Fear and Greed
    """
    try:
        if df.empty:
            return df

        # Создаем копию для безопасности
        result_df = df.copy()

        # Убеждаемся, что date_column это datetime
        if date_column in result_df.columns:
            result_df[date_column] = pd.to_datetime(result_df[date_column])

            # Создаем колонку date для мержа (только дата без времени)
            result_df['merge_date'] = result_df[date_column].dt.date
            result_df['merge_date'] = pd.to_datetime(result_df['merge_date'])
        else:
            logging.error(f"Колонка {date_column} не найдена в DataFrame")
            return result_df

        # Получаем диапазон дат из данных
        min_date = result_df['merge_date'].min()
        max_date = result_df['merge_date'].max()

        logging.info(f"Добавляем Fear and Greed для периода {min_date.date()} - {max_date.date()}")

        # Пытаемся получить реальные данные Fear and Greed
        days_range = (max_date - min_date).days + 30  # +30 дней запаса
        fg_df = fear_greed_enhancer.create_fear_greed_dataframe(limit_days=days_range)

        # Если реальных данных недостаточно, дополняем синтетическими
        if fg_df.empty or fg_df['date'].min() > min_date:
            logging.info("Дополняем недостающие Fear and Greed данные синтетическими")

            # Определяем диапазон для синтетических данных
            synthetic_start = min_date
            synthetic_end = fg_df['date'].min() - timedelta(days=1) if not fg_df.empty else max_date

            synthetic_fg = fear_greed_enhancer.generate_synthetic_fear_greed(
                synthetic_start, synthetic_end
            )

            # Объединяем реальные и синтетические данные
            fg_df = pd.concat([synthetic_fg, fg_df], ignore_index=True)
            fg_df = fg_df.drop_duplicates(subset=['date']).sort_values('date')

        # Мерджим данные по дате
        result_df = result_df.merge(
            fg_df,
            left_on='merge_date',
            right_on='date',
            how='left'
        )

        # Заполняем пропуски методом forward fill
        result_df['fear_greed_index'] = result_df['fear_greed_index'].ffill()
        result_df['fg_classification'] = result_df['fg_classification'].ffill()

        # Если всё ещё есть пропуски в начале, заполняем нейтральным значением
        result_df['fear_greed_index'] = result_df['fear_greed_index'].fillna(50)
        result_df['fg_classification'] = result_df['fg_classification'].fillna('Neutral')

        classification_map = {
            'Extreme Fear': 1,
            'Fear': 2,
            'Neutral': 3,
            'Greed': 4,
            'Extreme Greed': 5
        }
        result_df['fg_classification'] = result_df['fg_classification'].map(classification_map).fillna(3)

        # Удаляем вспомогательные колонки
        result_df = result_df.drop(['merge_date', 'date'], axis=1, errors='ignore')

        # Добавляем дополнительные индикаторы Fear and Greed
        result_df = add_fear_greed_indicators(result_df)

        added_cols = ['fear_greed_index', 'fg_classification', 'fg_is_extreme_fear', 'fg_is_extreme_greed']
        logging.info(f"Успешно добавлены колонки Fear and Greed: {added_cols}")

        return result_df

    except Exception as e:
        logging.error(f"Ошибка при добавлении Fear and Greed Index: {e}")
        return df  # Возвращаем исходный DataFrame при ошибке


def add_fear_greed_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет дополнительные индикаторы на основе Fear and Greed Index

    Args:
        df: DataFrame с fear_greed_index колонкой

    Returns:
        DataFrame с дополнительными индикаторами
    """
    if 'fear_greed_index' not in df.columns:
        return df

    # Бинарные индикаторы экстремальных значений
    df['fg_is_extreme_fear'] = df['fear_greed_index'] <= 25
    df['fg_is_extreme_greed'] = df['fear_greed_index'] >= 75
    # После создания булевых колонок добавьте:
    df['fg_is_extreme_fear'] = df['fg_is_extreme_fear'].astype(int)
    df['fg_is_extreme_greed'] = df['fg_is_extreme_greed'].astype(int)

    # Скользящие средние Fear and Greed (если достаточно данных)
    if len(df) >= 7:
        df['fg_ma_7'] = df['fear_greed_index'].rolling(window=7).mean()
    if len(df) >= 30:
        df['fg_ma_30'] = df['fear_greed_index'].rolling(window=30).mean()

    # Изменения Fear and Greed
    df['fg_change_1d'] = df['fear_greed_index'].diff(1)

    # Числовые коды для классификации (для ML моделей)
    classification_map = {
        'Extreme Fear': 1,
        'Fear': 2,
        'Neutral': 3,
        'Greed': 4,
        'Extreme Greed': 5
    }
    df['fg_classification_code'] = df['fg_classification'].map(classification_map).fillna(3)

    return df


# ========================================
# АЛЬТЕРНАТИВНАЯ УПРОЩЕННАЯ ВЕРСИЯ
# ========================================

def load_data_simple(ticker_name: str, engine=engine) -> pd.DataFrame:
    """
    Упрощенная версия с базовым добавлением Fear and Greed
    """
    try:
        # Основной запрос данных
        query = text(f"""
            SELECT * FROM all_dfs."{ticker_name}" 
            ORDER BY timestamp
        """)
        df = pd.read_sql_query(query, engine)

        if df.empty:
            return df

        # Преобразование timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Простое добавление Fear and Greed
        df = add_simple_fear_greed(df)

        logging.info(f"Загружены данные для {ticker_name}: {len(df)} строк с Fear and Greed")
        return df

    except Exception as e:
        logging.error(f"Ошибка при загрузке данных для {ticker_name}: {str(e)}")
        return pd.DataFrame()


def add_simple_fear_greed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Простое добавление текущего Fear and Greed Index
    """
    try:
        # Получаем только текущее значение
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)

        if response.status_code == 200:
            data = response.json()
            current_fg = int(data['data'][0]['value'])
            current_classification = data['data'][0]['value_classification']
        else:
            # Fallback значения
            current_fg = 50
            current_classification = 'Neutral'

        # Добавляем ко всем строкам (для простоты)
        df['fear_greed_index'] = current_fg
        df['fg_classification'] = current_classification
        df['fg_is_extreme'] = (current_fg <= 25) | (current_fg >= 75)

        return df

    except Exception as e:
        logging.warning(f"Ошибка получения Fear and Greed, используем значения по умолчанию: {e}")
        df['fear_greed_index'] = 50
        df['fg_classification'] = 'Neutral'
        df['fg_is_extreme'] = False
        return df


# ========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ========================================
# Без Fear and Greed:
# df = load_data("BBG00A1034X1", add_fear_greed=True, engine=engine)
