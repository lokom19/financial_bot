"""
Расчёт технических индикаторов по последним свечам тикера.
"""
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Московский TZ + час, когда биржа гарантированно закрылась (вечерка ~23:50)
_MSK = timezone(timedelta(hours=3))
_MARKET_CLOSE_HOUR_MSK = 20  # после 20:00 МСК дневная свеча считается закрытой


def _is_today_candle_complete() -> bool:
    """Закрылись ли уже основные торги по Москве."""
    now = datetime.now(_MSK)
    # Сб(5) и Вс(6) — биржа не работает, считаем что предыдущий день закрыт
    if now.weekday() >= 5:
        return True
    return now.hour >= _MARKET_CLOSE_HOUR_MSK


def compute_ta_indicators(engine, figi: str, n_candles: int = 100) -> dict:
    """
    Считает RSI, MACD, Stochastic, SMA по последним свечам тикера
    из таблицы all_dfs.<figi>. Возвращает значения на последней дате.

    Важно: если самая последняя свеча в БД — за СЕГОДНЯ, а рынок ещё
    открыт (до 20:00 МСК) — эта свеча НЕПОЛНАЯ и её close не финальный.
    Отбрасываем её и работаем с предыдущим полным днём.
    """
    try:
        q = text(
            f'SELECT timestamp, open, high, low, close, volume '
            f'FROM all_dfs."{figi}" ORDER BY timestamp DESC LIMIT :n'
        )
        with engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"n": n_candles})
        if df.empty or len(df) < 30:
            return {}
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Защита от "недоторгованной" сегодняшней свечи
        today_msk = datetime.now(_MSK).date()
        if not _is_today_candle_complete():
            last_date = pd.to_datetime(df['timestamp'].iloc[-1]).date()
            if last_date >= today_msk:
                logger.info(
                    "Отбрасываю свечу %s — день не закрыт (текущее время МСК %s)",
                    last_date, datetime.now(_MSK).strftime("%H:%M"),
                )
                df = df[pd.to_datetime(df['timestamp']).dt.date < today_msk].reset_index(drop=True)
                if df.empty or len(df) < 30:
                    return {}

        # RSI(14)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = np.where(loss != 0, gain / loss, 100)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        # Stochastic K(14)
        low_min = df["low"].rolling(14).min()
        high_max = df["high"].rolling(14).max()
        stoch_k = 100 * ((df["close"] - low_min) / (high_max - low_min))

        # SMA
        sma_5 = df["close"].rolling(5).mean()
        sma_20 = df["close"].rolling(20).mean()

        # Bollinger position
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_pos = (df["close"] - bb_lower) / (bb_upper - bb_lower)

        def last(v):
            try:
                x = float(v.iloc[-1])
                return None if (x != x) else x
            except Exception:
                return None

        # Дата последней свечи в датасете
        last_ts = df["timestamp"].iloc[-1]
        try:
            last_date_iso = (
                last_ts.date().isoformat() if hasattr(last_ts, "date")
                else pd.to_datetime(last_ts).date().isoformat()
            )
        except Exception:
            last_date_iso = None

        return {
            "rsi": last(pd.Series(rsi)),
            "macd": last(macd),
            "macd_signal": last(macd_signal),
            "stoch_k": last(stoch_k),
            "sma_5": last(sma_5),
            "sma_20": last(sma_20),
            "bb_position": last(bb_pos),
            "last_price": last(df["close"]),
            "last_high": last(df["high"]),
            "last_low": last(df["low"]),
            "last_candle_date": last_date_iso,
        }
    except Exception as e:
        logger.warning("Не удалось посчитать TA для %s: %s", figi, e)
        return {}
