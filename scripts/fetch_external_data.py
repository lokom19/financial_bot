#!/usr/bin/env python3
"""
Загрузка внешних временных рядов для cross-asset features.

Источники:
- MOEX ISS API — USD/RUB, Brent (BR фьючерс), IMOEX
- CBR ежедневный XML — ключевая ставка

Данные сохраняются в public.external_series (series_name, date, value).
Скрипт идемпотентен — использует UPSERT.

Запуск:
    python scripts/fetch_external_data.py
    python scripts/fetch_external_data.py --series brent
    python scripts/fetch_external_data.py --days 365
"""
import argparse
import logging
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("fetch_external")


def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', '')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    return create_engine(url)


def ensure_table(engine):
    """Создаёт таблицу external_series если её нет."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.external_series (
                series_name VARCHAR(50) NOT NULL,
                date DATE NOT NULL,
                value DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (series_name, date)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_external_series_name_date
            ON public.external_series (series_name, date DESC)
        """))
    logger.info("Таблица external_series готова")


def upsert_series(engine, series_name: str, points: list):
    """points: list of (date, value)"""
    if not points:
        return 0
    with engine.begin() as conn:
        for d, v in points:
            conn.execute(text("""
                INSERT INTO public.external_series (series_name, date, value)
                VALUES (:n, :d, :v)
                ON CONFLICT (series_name, date) DO UPDATE SET value = EXCLUDED.value
            """), {"n": series_name, "d": d, "v": v})
    return len(points)


# ============================================================
# MOEX ISS API — https://iss.moex.com/iss/reference/
# ============================================================

def _fetch_moex_candles(engine_url: str, market: str, board: str,
                        security: str, start: date, end: date) -> list:
    """
    Универсальная загрузка свечей через MOEX ISS.
    Возвращает список (date, close).
    """
    base_url = (
        f"https://iss.moex.com/iss/engines/{engine_url}/markets/{market}"
        f"/boards/{board}/securities/{security}/candles.json"
    )
    interval = 24  # дневные свечи
    all_points = []
    from_str = start.isoformat()
    till_str = end.isoformat()
    while True:
        params = {
            "from": from_str, "till": till_str,
            "interval": interval, "start": len(all_points),
        }
        try:
            resp = requests.get(base_url, params=params, timeout=20)
            if resp.status_code != 200:
                logger.warning("MOEX %s HTTP %d", security, resp.status_code)
                break
            data = resp.json()
            candles = data.get("candles", {})
            cols = candles.get("columns", [])
            rows = candles.get("data", [])
            if not rows:
                break
            close_idx = cols.index("close")
            begin_idx = cols.index("begin")
            for row in rows:
                begin_dt = datetime.fromisoformat(row[begin_idx])
                close_v = row[close_idx]
                if close_v is not None:
                    all_points.append((begin_dt.date(), float(close_v)))
            if len(rows) < 500:
                break  # последняя страница
        except Exception as e:
            logger.warning("MOEX fetch error for %s: %s", security, e)
            break
    return all_points


def fetch_usd_rub(engine, start: date, end: date) -> int:
    """USD/RUB через MOEX CETS — code USD000UTSTOM (расчёты TOM)."""
    points = _fetch_moex_candles(
        engine_url="currency", market="selt", board="CETS",
        security="USD000UTSTOM", start=start, end=end,
    )
    n = upsert_series(engine, "usd_rub", points)
    logger.info("USD/RUB: сохранено %d свечей", n)
    return n


def fetch_imoex(engine, start: date, end: date) -> int:
    """Индекс МосБиржи."""
    points = _fetch_moex_candles(
        engine_url="stock", market="index", board="SNDX",
        security="IMOEX", start=start, end=end,
    )
    n = upsert_series(engine, "imoex", points)
    logger.info("IMOEX: сохранено %d свечей", n)
    return n


def fetch_brent(engine, start: date, end: date) -> int:
    """
    Brent через Yahoo Finance (BZ=F — фьючерс на Brent Crude).
    Yahoo отдаёт CSV/JSON без авторизации через query1.finance.yahoo.com.
    """
    # Также сохраняем старый proxy для backward compat, но приоритет — real Brent
    _ = _fetch_moex_candles(
        engine_url="stock", market="index", board="SNDX",
        security="RTSOG", start=start, end=end,
    )
    if _:
        upsert_series(engine, "brent_proxy", _)

    # Real Brent через Yahoo
    import time as _time
    start_ts = int(_time.mktime(start.timetuple()))
    end_ts = int(_time.mktime(end.timetuple())) + 86400
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/BZ=F"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    }
    points = []
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            logger.warning("Yahoo Brent HTTP %d (fallback на proxy)", resp.status_code)
            return len(_) if _ else 0
        # CSV: Date,Open,High,Low,Close,Adj Close,Volume
        for line in resp.text.strip().split("\n")[1:]:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                d = datetime.fromisoformat(parts[0]).date()
                close_v = float(parts[4])
                points.append((d, close_v))
            except (ValueError, IndexError):
                continue
        n = upsert_series(engine, "brent", points)
        logger.info("Brent (Yahoo BZ=F): сохранено %d свечей", n)
        return n
    except Exception as e:
        logger.warning("Yahoo Brent fetch failed: %s (fallback на proxy)", e)
        return len(_) if _ else 0


def fetch_cbr_rate(engine, start: date, end: date) -> int:
    """
    Ключевая ставка ЦБ через открытое API.
    https://www.cbr-xml-daily.ru/ — неофициальный агрегатор, только текущая.
    Для истории используем cbr.ru SOAP или dumps.
    Пока пропустим — ключевая ставка меняется редко (2-3 раза в год),
    её можно захардкодить или взять с cbr.ru scripts.
    """
    logger.info("CBR key rate: пока не реализовано (меняется редко)")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365 * 3,
                        help="Загружать данные за последние N дней (default 3 года)")
    parser.add_argument("--series", choices=["all", "usd_rub", "imoex", "brent", "cbr"],
                        default="all")
    args = parser.parse_args()

    engine = get_engine()
    ensure_table(engine)

    end = date.today()
    start = end - timedelta(days=args.days)
    logger.info("Загружаем внешние ряды %s → %s", start, end)

    if args.series in ("all", "usd_rub"):
        fetch_usd_rub(engine, start, end)
    if args.series in ("all", "imoex"):
        fetch_imoex(engine, start, end)
    if args.series in ("all", "brent"):
        fetch_brent(engine, start, end)
    if args.series in ("all", "cbr"):
        fetch_cbr_rate(engine, start, end)

    logger.info("Готово")


if __name__ == "__main__":
    main()
