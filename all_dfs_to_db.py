import asyncio
import logging
import os
import random
import time
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
from grpc import StatusCode
from grpc.aio._call import AioRpcError
from sqlalchemy import create_engine, text
from tinkoff.invest import AsyncClient, CandleInterval, RequestError, AioRequestError
from tinkoff.invest.schemas import CandleSource
from tinkoff.invest.utils import now

from get_all_tinkoff_figi import get_tickers_from_bd

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

TOKEN = os.environ.get("INVEST_TOKEN")
if not TOKEN:
    logger.error("INVEST_TOKEN не найден в переменных окружения!")

# Конфигурация повторных попыток
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # секунды
MAX_BACKOFF = 60  # максимальная задержка в секундах

# Конфигурация базы данных PostgreSQL
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "mysecretpassword")

# Строка подключения к PostgreSQL
PG_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# Функция для создания схемы и таблицы, если они не существуют
def setup_database():
    """Создание схемы и таблицы в PostgreSQL, если они не существуют"""
    try:
        engine = create_engine(PG_CONNECTION_STRING)
        logger.info(PG_CONNECTION_STRING)

        with engine.connect() as connection:
            # Создаем схему all_dfs, если не существует
            connection.execute(text("CREATE SCHEMA IF NOT EXISTS all_dfs;"))

            # Убедимся, что изменения применяются
            connection.commit()

        logger.info("Схема all_dfs создана или уже существует")
        return True
    except Exception as e:
        logger.error(f"Ошибка при настройке базы данных: {e}")
        return False


# Функция для преобразования Quotation в float
def quotation_to_float(quotation):
    return quotation.units + quotation.nano / 1_000_000_000


async def get_df(figi, ticker_info):
    """Получение данных свечей с механизмом повторных попыток"""
    retry_count = 0
    ticker_name = ticker_info[1] if len(ticker_info) > 1 else "Unknown"

    while retry_count < MAX_RETRIES:
        try:
            logger.info(f"Попытка {retry_count + 1} получения данных для {figi} ({ticker_name})")

            async with AsyncClient(TOKEN) as client:
                # Устанавливаем таймаут для соединения
                candles = client.get_all_candles(
                    instrument_id=figi,
                    from_=now() - timedelta(days=1000),
                    interval=CandleInterval.CANDLE_INTERVAL_DAY,
                    candle_source_type=CandleSource.CANDLE_SOURCE_EXCHANGE,
                )

                data = []
                async for candle in candles:
                    data.append({
                        "timestamp": candle.time,
                        "open": quotation_to_float(candle.open),
                        "high": quotation_to_float(candle.high),
                        "low": quotation_to_float(candle.low),
                        "close": quotation_to_float(candle.close),
                        "volume": candle.volume,
                        "figi": figi,  # Добавляем FIGI для идентификации инструмента
                    })

                df = pd.DataFrame(data)
                # Добавляем маппинг Fear & Greed Index по дате
                # if not df.empty and fng_df is not None and not fng_df.empty:
                #     # Преобразуем timestamp в дату для join'а
                #     df['date'] = df['timestamp'].dt.date
                #
                #     # Объединяем данные по дате
                #     df = df.merge(
                #         fng_df[['date', 'fear_greed_value', 'fear_greed_classification',
                #                 'fear_greed_classification_numeric']],
                #         on='date',
                #         how='left'
                #     )
                #
                #     # Заполняем пропуски (для выходных дней) методом forward fill
                #     df = df.sort_values('timestamp')  # Убеждаемся, что данные отсортированы по времени
                #     df['fear_greed_value'] = df['fear_greed_value'].fillna(method='ffill')
                #     df['fear_greed_classification'] = df['fear_greed_classification'].fillna(method='ffill')
                #     df['fear_greed_classification_numeric'] = df['fear_greed_classification_numeric'].fillna(
                #         method='ffill')
                #
                #     # Если первые записи остались с NaN (данных FNG еще не было), заполняем нейтральными значениями
                #     df['fear_greed_value'] = df['fear_greed_value'].fillna(50)  # Нейтральное значение
                #     df['fear_greed_classification'] = df['fear_greed_classification'].fillna('Neutral')
                #     df['fear_greed_classification_numeric'] = df['fear_greed_classification_numeric'].fillna(3)
                #
                #     # Удаляем вспомогательную колонку date
                #     df = df.drop('date', axis=1)
                #
                #     logger.info(f"Данные для {figi} обогащены Fear & Greed Index")
                logger.info(f'{figi} ({ticker_name}) ++++++++++++++++++ Успешно получено {len(df)} свечей')

                if not df.empty:
                    df.timestamp = df.timestamp.dt.tz_localize(None)  # Убираем часовой пояс

                    # Сохраняем данные в PostgreSQL схему all_dfs
                    try:
                        engine = create_engine(PG_CONNECTION_STRING)

                        # Имя таблицы будет равно FIGI
                        table_name = figi

                        # Сохраняем в схему all_dfs
                        df.to_sql(
                            name=table_name,
                            con=engine,
                            schema='all_dfs',
                            if_exists='replace',  # Заменяем существующую таблицу
                            index=False
                        )

                        logger.info(f"Данные для {figi} ({ticker_name}) успешно сохранены в PostgreSQL схему all_dfs")
                    except Exception as db_error:
                        logger.error(f"Ошибка сохранения в PostgreSQL для {figi} ({ticker_name}): {str(db_error)}")

                return df

        except (AioRpcError, RequestError, AioRequestError) as e:
            retry_count += 1

            # Анализируем ошибку
            if hasattr(e, 'code'):
                if callable(e.code):
                    # Если это метод, вызываем его
                    error_code = e.code()
                else:
                    # Если это атрибут, используем напрямую
                    error_code = e.code

                if error_code == StatusCode.UNAVAILABLE:
                    error_message = f"Сетевая ошибка при получении данных для {figi} ({ticker_name}): {str(e)}"
                else:
                    error_message = f"Ошибка API при получении данных для {figi} ({ticker_name}): {str(e)}"
            else:
                error_message = f"Ошибка API при получении данных для {figi} ({ticker_name}): {str(e)}"

            logger.error(error_message)

            if retry_count >= MAX_RETRIES:
                logger.error(f"Достигнуто максимальное число попыток ({MAX_RETRIES}) для {figi} ({ticker_name})")
                # Возвращаем пустой DataFrame вместо ошибки
                return pd.DataFrame()

            # Расчет времени задержки с экспоненциальным увеличением и случайным фактором (jitter)
            backoff = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** (retry_count - 1)))
            jitter = random.uniform(0, 0.3 * backoff)  # Добавляем до 30% случайности
            wait_time = backoff + jitter

            logger.info(f"Повторная попытка через {wait_time:.2f} секунд...")
            await asyncio.sleep(wait_time)

        except Exception as e:
            retry_count += 1
            logger.error(f"Неожиданная ошибка при получении данных для {figi} ({ticker_name}): {str(e)}")

            if retry_count >= MAX_RETRIES:
                logger.error(f"Достигнуто максимальное число попыток ({MAX_RETRIES}) для {figi} ({ticker_name})")
                return pd.DataFrame()

            # Такой же механизм повторных попыток для других ошибок
            backoff = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** (retry_count - 1)))
            jitter = random.uniform(0, 0.3 * backoff)
            wait_time = backoff + jitter

            logger.info(f"Повторная попытка через {wait_time:.2f} секунд...")
            await asyncio.sleep(wait_time)


async def process_tickers(tickers):
    """Обработка списка тикеров с ограничением параллельности"""
    # Ограничиваем количество одновременных запросов
    semaphore = asyncio.Semaphore(1)  # не более 1 одновременных запросов

    async def process_ticker(ticker_info):
        figi = ticker_info[0]
        async with semaphore:
            res = await get_df(figi, ticker_info)
            # await asyncio.sleep(1)
            return res

    # Запускаем задачи для всех тикеров
    tasks = [process_ticker(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Проверяем результаты
    success_count = 0
    error_count = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            ticker_name = tickers[i][1] if len(tickers[i]) > 1 else "Unknown"
            logger.error(f"Ошибка при обработке {tickers[i][0]} ({ticker_name}): {str(result)}")
            error_count += 1
        elif isinstance(result, pd.DataFrame) and not result.empty:
            success_count += 1

    logger.info(f"Обработка завершена. Успешно: {success_count}, с ошибками: {error_count}")
    return success_count, error_count


async def main_async():
    """Асинхронная основная функция"""
    start_time = time.time()

    try:
        # Создаем схему, если она не существует
        setup_database()

        tickers = get_tickers_from_bd()
        logger.info(f"Получено {len(tickers)} тикеров для обработки")

        success, errors = await process_tickers(tickers)

        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Время выполнения: {execution_time:.2f} секунд")
        logger.info(f"Успешно обработано: {success}/{len(tickers)} тикеров")

        # Возвращаем результат для использования в других модулях
        return {
            "success": success,
            "errors": errors,
            "total": len(tickers),
            "execution_time": execution_time
        }

    except Exception as e:
        logger.error(f"Ошибка в основной функции: {str(e)}")
        return {
            "success": 0,
            "errors": 0,
            "total": 0,
            "execution_time": time.time() - start_time,
            "error": str(e)
        }


def main():
    """Точка входа для запуска из других модулей или напрямую"""
    return asyncio.run(main_async())


if __name__ == '__main__':
    main()