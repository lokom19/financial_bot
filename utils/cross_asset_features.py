"""
Обогащение датафрейма тикера cross-asset фичами.

Для каждого тикера из TICKER_TO_EXTERNAL добавляем соответствующие
внешние ряды (USD/RUB, Brent proxy, IMOEX) и derived-фичи:
- {series}_close — абсолютное значение
- {series}_return_1d, _5d, _20d — доходности
- {series}_ma_ratio_20 — отношение close/MA20 (импульс)

Данные читаются из public.external_series (заполняется скриптом
scripts/fetch_external_data.py).
"""
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Какие внешние ряды добавляем к каждому тикеру.
# Мэппинг подтверждён A/B тестом на catboost:
# ✓ SBER/VTBR (+3-4% dir): usd_rub — валютные позиции банков
# ✓ AFLT (+4.5% dir): usd_rub+brent_proxy — топливные costs
# × YDEX/OZON: IMOEX вредит (-1..-5% dir), нужен NASDAQ вместо
# ~ GAZP: brent_proxy слабо помогает (+0.5%), нужен настоящий Brent
TICKER_TO_EXTERNAL = {
    # Нефтегаз — Brent proxy (пока) + валюта. IMOEX убрали (циркулярность).
    "GAZP": ["brent", "usd_rub"],
    "LKOH": ["brent", "usd_rub"],
    "ROSN": ["brent", "usd_rub"],
    # Банки — USD/RUB + IMOEX (положительный эффект подтверждён)
    "SBER": ["usd_rub", "imoex"],
    "VTBR": ["usd_rub", "imoex"],
    "TCSG": ["usd_rub", "imoex"],
    # IT / ритейл — только USD/RUB, IMOEX ухудшает
    # TODO: добавить NASDAQ proxy когда появится fetcher
    "OZON": ["usd_rub"],
    "YDEX": ["usd_rub"],
    # Телеком — умеренная зависимость от MOEX
    "MTSS": ["imoex"],
    "HEAD": ["imoex"],
    # Транспорт — валюта + топливо (Brent), эффект +4.5%
    "AFLT": ["brent", "usd_rub"],
}


def _load_external_series(engine, series_name: str,
                          start_date, end_date) -> pd.DataFrame:
    """
    Читает временной ряд из external_series.
    Возвращает DataFrame с колонками [date, value].
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT date, value FROM public.external_series
                WHERE series_name = :n AND date BETWEEN :s AND :e
                ORDER BY date
            """), conn, params={"n": series_name, "s": start_date, "e": end_date})
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.warning("Не удалось загрузить %s: %s", series_name, e)
        return pd.DataFrame(columns=["date", "value"])


def _add_derived(df: pd.DataFrame, series: str) -> pd.DataFrame:
    """Считает return_1d/5d/20d и ma_ratio_20 для колонки {series}_close."""
    close_col = f"{series}_close"
    if close_col not in df.columns:
        return df

    # Доходности
    df[f"{series}_return_1d"] = df[close_col].pct_change(1) * 100
    df[f"{series}_return_5d"] = df[close_col].pct_change(5) * 100
    df[f"{series}_return_20d"] = df[close_col].pct_change(20) * 100

    # Импульс: цена относительно скользящей средней
    ma20 = df[close_col].rolling(20, min_periods=5).mean()
    df[f"{series}_ma_ratio_20"] = df[close_col] / ma20 - 1

    return df


def add_cross_asset_features(df: pd.DataFrame, ticker: str,
                              engine) -> pd.DataFrame:
    """
    Обогащает DataFrame тикера cross-asset фичами.

    Args:
        df: DataFrame с колонкой 'timestamp' (datetime)
        ticker: код тикера (SBER, OZON и т.д.) — не FIGI
        engine: SQLAlchemy engine для чтения external_series

    Returns:
        DataFrame с добавленными фичами (либо исходный если ряды не найдены).
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return df

    external_names = TICKER_TO_EXTERNAL.get(ticker.upper())
    if not external_names:
        logger.debug("Cross-asset: нет маппинга для тикера %s", ticker)
        return df

    result = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result["timestamp"]):
        result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["_join_date"] = result["timestamp"].dt.date
    result["_join_date"] = pd.to_datetime(result["_join_date"])

    start = result["_join_date"].min().date()
    end = result["_join_date"].max().date()

    added_any = False
    for series in external_names:
        ext = _load_external_series(engine, series, start, end)
        if ext.empty:
            logger.debug("Cross-asset: %s пустой для %s", series, ticker)
            continue

        ext = ext.rename(columns={"value": f"{series}_close", "date": "_join_date"})
        result = result.merge(ext, on="_join_date", how="left")

        # forward-fill пропуски (выходные, праздники не совпадают между рынками)
        result[f"{series}_close"] = result[f"{series}_close"].ffill().bfill()

        # derived
        result = _add_derived(result, series)
        added_any = True

    result = result.drop(columns=["_join_date"], errors="ignore")

    if added_any:
        n_new = sum(1 for c in result.columns if any(c.startswith(f"{s}_") for s in external_names))
        logger.info("Cross-asset для %s: добавлено %d фич из рядов %s",
                    ticker, n_new, external_names)
    return result


def resolve_ticker_from_figi(engine, figi: str) -> Optional[str]:
    """FIGI → ticker через таблицу public.tickers."""
    try:
        with engine.connect() as conn:
            r = conn.execute(text(
                "SELECT ticker FROM public.tickers WHERE figi = :f LIMIT 1"
            ), {"f": figi}).fetchone()
            return r[0] if r else None
    except Exception:
        return None
