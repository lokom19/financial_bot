import os
import psycopg2
import requests
from psycopg2.extras import execute_values, Json, RealDictCursor
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# === КОНФИГ ПОДКЛЮЧЕНИЯ ===
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")

LOG_FILE = os.getenv("LOG_FILE", "log.txt")
STREAMLIT_URL = os.getenv("STREAMLIT_URL", "http://127.0.0.1:8008/log")

# === API Binance ===
BINANCE_SPOT_API = "https://api.binance.com/api/v3"
BINANCE_FUTURES_API = "https://fapi.binance.com/fapi/v1"

# === ЛОГИРОВАНИЕ ===
# logging.basicConfig(
#     filename=LOG_FILE,
#     filemode='a',
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(message)s"
# )
logger = logging.getLogger(__name__)


def send_status(event: str, data: dict):
    pass
    # payload = {"event": event, "timestamp": datetime.utcnow().isoformat(), "data": data}
    # try:
    #     requests.post(STREAMLIT_URL, json=payload, timeout=5)
    # except Exception as ex:
    #     logger.error("Failed to send status to Streamlit: %s", ex)


# === FASTAPI ===
app = FastAPI(
    title="Binance Trades Importer",
    description="Импортирует последние 10000 сделок по BTC/USDT в Postgres",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"status": "Binance Trades Importer is running"}


def check_database():
    """Проверка доступности БД и существования таблиц"""
    logger.info("Проверка доступности БД...")
    send_status("db_check_start", {})

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        logger.info(cur)

        # Проверка существования таблиц через information_schema
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'model_results'
            )
        """)
        model_results_exists = cur.fetchone()[0]

        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'tickers'
            )
        """)
        tickers_exists = cur.fetchone()[0]

        # Проверка схемы all_dfs
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.schemata
                WHERE schema_name = 'all_dfs'
            )
        """)
        all_dfs_schema_exists = cur.fetchone()[0]

        # Проверка схемы crypto
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.schemata
                WHERE schema_name = 'crypto'
            )
        """)
        crypto_schema_exists = cur.fetchone()[0]

        # Проверка таблиц в схеме crypto
        crypto_tables_exist = False
        if crypto_schema_exists:
            cur.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'crypto'
            """)
            crypto_tables_count = cur.fetchone()[0]
            crypto_tables_exist = crypto_tables_count > 0

        conn.close()

        logger.info("Результаты проверки БД:")
        logger.info(f"  - model_results: {model_results_exists}")
        logger.info(f"  - tickers: {tickers_exists}")
        logger.info(f"  - схема all_dfs: {all_dfs_schema_exists}")
        logger.info(f"  - схема crypto: {crypto_schema_exists}")
        logger.info(f"  - таблицы в crypto: {crypto_tables_exist}")

        send_status("db_check_result", {
            "model_results": model_results_exists,
            "tickers": tickers_exists,
            "all_dfs_schema": all_dfs_schema_exists,
            "crypto_schema": crypto_schema_exists,
            "crypto_tables": crypto_tables_exist
        })

        # Проверяем основные таблицы
        if not model_results_exists:
            raise Exception("Таблица model_results не существует в схеме public")
        if not tickers_exists:
            raise Exception("Таблица tickers не существует в схеме public")

        return True

    except psycopg2.OperationalError as e:
        logger.error(f"Ошибка подключения к БД: {e}")
        send_status("db_check_failed", {"error": f"Ошибка подключения: {str(e)}"})
        raise
    except Exception as e:
        logger.error(f"Ошибка проверки БД: {e}")
        send_status("db_check_failed", {"error": str(e)})
        raise


def get_db_connection():
    """Установка соединения с БД"""
    logger.info("Установка соединения с БД...")
    send_status("db_connect", {"host": DB_HOST, "port": DB_PORT, "dbname": DB_NAME})

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        logger.info("Соединение с БД установлено")
        send_status("db_connected", {})
        return conn
    except Exception as e:
        logger.error("Ошибка подключения к БД: %s", e)
        send_status("db_error", {"error": str(e)})
        raise


def insert_trades(trades: list, table_name: str, conn):
    """Вставка данных в БД с проверкой"""
    if not trades:
        logger.info("Нет данных для вставки в %s", table_name)
        send_status("insert_skipped", {"table": table_name})
        return 0

    cols = [
        "trade_id", "timestamp_ms", "datetime", "symbol",
        "order_id", "order_type", "side", "price", "amount",
        "cost", "fee_cost", "fee_currency", "fee_rate", "info"
    ]

    values = []
    for t in trades:
        fee = t.get('fee') or {}
        dt = datetime.fromtimestamp(t['timestamp'] / 1000.0)
        values.append((
            t.get('id'),
            t.get('timestamp'),
            dt,
            t.get('symbol'),
            t.get('order'),
            t.get('type'),
            t.get('side'),
            t.get('price'),
            t.get('amount'),
            t.get('cost'),
            fee.get('cost'),
            fee.get('currency'),
            fee.get('rate'),
            Json(t.get('info', {}))
        ))

    try:
        with conn.cursor() as cur:
            cols_sql = ", ".join(cols)
            query = f"""
                INSERT INTO trading.{table_name} ({cols_sql})
                VALUES %s
                ON CONFLICT (trade_id, timestamp_ms, symbol) DO NOTHING
            """
            execute_values(cur, query, values)
            conn.commit()

        logger.info("Успешно вставлено %d записей в %s", len(values), table_name)
        send_status("insert_success", {"table": table_name, "count": len(values)})
        return len(values)

    except Exception as e:
        logger.error("Ошибка вставки в %s: %s", table_name, e)
        send_status("insert_failed", {"table": table_name, "error": str(e)})
        conn.rollback()
        raise


def fetch_binance_trades(api_url: str, symbol: str, limit: int, start_time: Optional[int] = None):
    """Запрос данных с Binance API"""
    params = {
        'symbol': symbol.replace('/', ''),
        'limit': limit
    }
    if start_time:
        params['startTime'] = start_time

    # try:
    logger.info("Запрос к %s с параметрами: %s", api_url, params)
    response = requests.get(f"{api_url}/trades", params=params, timeout=10)
    response.raise_for_status()
    return response.json()
    # except requests.exceptions.RequestException as e:
    #     logger.error("Ошибка API запроса: %s", e)
    #     raise HTTPException(status_code=500, detail=f"Binance API error: {str(e)}")


@app.get("/api/import_trades", summary="Импорт 10000 сделок спот и фьючерс")
async def import_trades():
    logger.info("Запуск импорта данных")
    send_status("import_start", {})
    conn = None

    try:
        # Проверка БД перед началом работы
        check_database()

        symbol = 'BTCUSDT'
        limit = 1000
        all_spot = []
        all_fut = []
        start_time = None

        # Получение спотовых сделок
        for _ in range(10):
            spot_trades = fetch_binance_trades(BINANCE_SPOT_API, symbol, limit, start_time)
            if not spot_trades:
                break
            all_spot.extend(spot_trades)
            start_time = spot_trades[-1]['time'] + 1
            time.sleep(0.2)

        # Получение фьючерсных сделок
        start_time = None
        for _ in range(10):
            fut_trades = fetch_binance_trades(BINANCE_FUTURES_API, symbol, limit, start_time)
            if not fut_trades:
                break
            all_fut.extend(fut_trades)
            start_time = fut_trades[-1]['time'] + 1
            time.sleep(0.2)

        # Конвертация данных
        def convert_trade(trade):
            return {
                'id': str(trade['id']),
                'timestamp': trade['time'],
                'symbol': 'BTC/USDT',
                'side': 'buy' if trade['isBuyerMaker'] else 'sell',
                'price': float(trade['price']),
                'amount': float(trade['qty']),
                'cost': float(trade['quoteQty']),
                'info': trade
            }

        all_spot = [convert_trade(t) for t in all_spot]
        all_fut = [convert_trade(t) for t in all_fut]

        # Вставка данных
        conn = get_db_connection()
        cnt_spot = insert_trades(all_spot, "binance_trades_spot", conn)
        cnt_fut = insert_trades(all_fut, "binance_trades_futures", conn)

        return {
            "success": True,
            "inserted_spot": cnt_spot,
            "inserted_futures": cnt_fut,
        }

    except Exception as e:
        logger.error("Критическая ошибка импорта: %s", e)
        send_status("import_failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
            logger.info("Соединение с БД закрыто")
            send_status("db_closed", {})


if __name__ == "__main__":
    import uvicorn

    logger.info("Запуск приложения")
    send_status("app_start", {})
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")