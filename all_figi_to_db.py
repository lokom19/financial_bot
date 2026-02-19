import logging
import os
import time
from pandas import DataFrame
from dotenv import load_dotenv
from t_tech.invest import Client, SecurityTradingStatus, RequestError
from t_tech.invest.services import InstrumentsService
from t_tech.invest.utils import quotation_to_decimal
from sqlalchemy import create_engine

load_dotenv()

TOKEN = os.environ.get("INVEST_TOKEN")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Тестовые тикеры для быстрой проверки
TEST_TICKERS = ["SBER", "YNDX", "VTBR", "TCSG", "OZON"]


def fetch_instruments_with_retry(instruments, method, max_retries=3):
    """Получить инструменты с retry при ошибках."""
    for attempt in range(max_retries):
        try:
            result = getattr(instruments, method)().instruments
            logger.info(f"✓ {method}: получено {len(result)} инструментов")
            return result
        except RequestError as e:
            logger.warning(f"Ошибка при получении {method} (попытка {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"✗ Не удалось получить {method} после {max_retries} попыток")
                return []
    return []


def main(test_mode=False):
    """Получаем список инструментов из Tinkoff API.

    Args:
        test_mode: Если True, сохраняем только тестовые тикеры (SBER, YNDX, VTBR, TCSG, OZON)
    """
    if not TOKEN:
        logger.error("INVEST_TOKEN не установлен!")
        return False

    with Client(TOKEN) as client:
        instruments: InstrumentsService = client.instruments
        tickers = []

        # В тестовом режиме берём только акции
        methods = ["shares"] if test_mode else ["shares", "bonds", "etfs", "currencies", "futures"]

        for method in methods:
            items = fetch_instruments_with_retry(instruments, method)
            for item in items:
                tickers.append(
                    {
                        "name": item.name,
                        "ticker": item.ticker,
                        "class_code": item.class_code,
                        "figi": item.figi,
                        "uid": item.uid,
                        "type": method,
                        "min_price_increment": float(quotation_to_decimal(item.min_price_increment)),
                        "scale": 9 - len(str(item.min_price_increment.nano)) + 1,
                        "lot": item.lot,
                        "trading_status": str(SecurityTradingStatus(item.trading_status).name),
                        "api_trade_available_flag": item.api_trade_available_flag,
                        "currency": item.currency,
                        "exchange": item.exchange,
                        "buy_available_flag": item.buy_available_flag,
                        "sell_available_flag": item.sell_available_flag,
                        "short_enabled_flag": item.short_enabled_flag,
                        "klong": float(quotation_to_decimal(item.klong)),
                        "kshort": float(quotation_to_decimal(item.kshort)),
                    }
                )

        tickers_df = DataFrame(tickers)

        # Фильтруем только тестовые тикеры если нужно
        if test_mode:
            tickers_df = tickers_df[tickers_df['ticker'].isin(TEST_TICKERS)]
            logger.info(f"Тестовый режим: отфильтровано {len(tickers_df)} тикеров: {TEST_TICKERS}")

        logger.info(f"Получено {len(tickers_df)} тикеров")
        if not tickers_df.empty:
            logger.info(f"Тикеры:\n{tickers_df[['ticker', 'name', 'figi']].to_string()}")

        # Write to database
        tickers_df.to_sql(name='tickers', con=engine, if_exists='replace', index=False, schema="public")
        logger.info("✓ Данные записаны в public.tickers")

        return True

if __name__ == "__main__":
    import sys
    # Тестовый режим: python all_figi_to_db.py --test
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)