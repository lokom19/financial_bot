"""
PRO-симулятор портфеля: стратегия Position Continuation с ATR trailing stop.

Отличия от базового portfolio_simulator:
- Позиция может держаться до N дней (не закрывается каждый день)
- Initial stop = 1.0 × ATR14 от entry
- Trailing stop = extreme_price ∓ 1.0 × ATR14 (для BUY/SELL)
- Выход по одному из:
    * trailing stop пробит внутри дня
    * verdict сменился на противоположный (confidence ≥ средняя)
    * достигнут лимит держания (по умолчанию 5 дней)
- Фильтр входа: только confidence ≥ "средняя"

Формат возврата совместим с базовым симулятором + новые поля:
- days_held в trades
- exit_reason: trailing / signal_flip / time_limit
"""
import logging
from collections import defaultdict
from datetime import date as _date, timedelta as _td
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

DEFAULT_COMMISSION_PCT = 0.05
CONFIDENCE_LEVEL = {"низкая": 1, "средняя": 2, "высокая": 3}


def _compute_atr(engine, figi: str, period: int = 14, lookback_days: int = 60) -> dict:
    """
    Возвращает {date: atr_value} для последних lookback_days дней.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = скользящее среднее TR за period дней.
    """
    q = text(
        f'SELECT timestamp::date AS d, high, low, close '
        f'FROM all_dfs."{figi}" '
        f'ORDER BY timestamp DESC LIMIT :n'
    )
    try:
        with engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"n": lookback_days + period})
    except Exception as e:
        logger.warning("ATR fetch failed for %s: %s", figi, e)
        return {}
    if df.empty or len(df) < period + 1:
        return {}
    df = df.sort_values("d").reset_index(drop=True)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return {d: float(v) for d, v in zip(df["d"], atr) if pd.notna(v)}


def _fetch_ohlc_range(engine, figi: str, start: _date, end: _date) -> dict:
    """{date: (open, high, low, close)} — свечи для симуляции."""
    q = text(
        f'SELECT timestamp::date AS d, open, high, low, close '
        f'FROM all_dfs."{figi}" '
        f'WHERE timestamp::date BETWEEN :s AND :e '
        f'ORDER BY timestamp'
    )
    try:
        with engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"s": start, "e": end})
    except Exception as e:
        logger.warning("OHLC fetch failed for %s: %s", figi, e)
        return {}
    return {row.d: (float(row.open), float(row.high),
                    float(row.low), float(row.close))
            for row in df.itertuples()}


def simulate_pro(engine, initial_capital: float = 100_000.0,
                 commission_pct: float = DEFAULT_COMMISSION_PCT,
                 min_confidence: str = "средняя",
                 max_hold_days: int = 5,
                 atr_mult_stop: float = 1.5,
                 atr_mult_trail: float = 2.0,
                 exclude_tickers: Optional[set] = None) -> dict:
    """
    Симуляция PRO-стратегии по всем ticker_reports.

    Args:
        initial_capital: стартовый депозит
        commission_pct: комиссия в одну сторону (%)
        min_confidence: минимум "низкая" / "средняя" / "высокая"
        max_hold_days: максимум дней держания позиции
        atr_mult_stop: множитель ATR для initial stop (default 1.5)
        atr_mult_trail: множитель ATR для trailing stop (default 2.0)
        exclude_tickers: множество тикеров для исключения из торговли
    """
    exclude_tickers = exclude_tickers or set()
    exclude_tickers = {t.upper() for t in exclude_tickers}

    # 1. Все вердикты по хронологии
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT prediction_date, ticker, figi, verdict, confidence,
                   current_price, entry_price
            FROM (
                SELECT DISTINCT ON (figi, prediction_date) *
                FROM public.ticker_reports
                WHERE prediction_date IS NOT NULL
                ORDER BY figi, prediction_date, timestamp DESC
            ) latest
            WHERE actual_close IS NOT NULL
            ORDER BY prediction_date ASC, ticker ASC
        """)).fetchall()

    # Список всех известных тикеров (для UI — до фильтрации)
    all_tickers = sorted({r[1] for r in rows}) if rows else []

    # Применяем фильтр по тикерам
    if exclude_tickers:
        rows = [r for r in rows if r[1].upper() not in exclude_tickers]

    if not rows:
        empty = _empty_result(initial_capital, commission_pct, min_confidence)
        empty["available_tickers"] = all_tickers
        empty["excluded_tickers"] = sorted(exclude_tickers)
        return empty

    min_level = CONFIDENCE_LEVEL.get(min_confidence, 0)

    # 2. Собираем FIGI -> ticker mapping и все даты для загрузки OHLC/ATR
    figi_by_ticker = {r[1]: r[2] for r in rows}
    all_dates = sorted({r[0] for r in rows})
    date_from = all_dates[0]
    date_to = all_dates[-1] + _td(days=max_hold_days + 5)

    # Загружаем OHLC + ATR для каждого тикера
    ohlc = {t: _fetch_ohlc_range(engine, f, date_from, date_to)
            for t, f in figi_by_ticker.items()}
    atr_map = {t: _compute_atr(engine, f, period=14,
                                lookback_days=(date_to - date_from).days + 30)
               for t, f in figi_by_ticker.items()}

    # 3. Верdicts by (date, ticker)
    verdicts = {}
    for r in rows:
        verdicts[(r[0], r[1])] = {
            "verdict": r[3],
            "confidence": (r[4] or "").lower(),
            "current_price": float(r[5]) if r[5] else None,
            "entry_price": float(r[6]) if r[6] else None,
        }

    tickers = sorted(figi_by_ticker.keys())

    # 4. Симуляция день за днём
    equity = initial_capital
    equity_curve = [{
        "date": (date_from - _td(days=1)).isoformat(),
        "equity": round(equity, 2),
        "ret_pct": 0.0,
    }]
    trades = []
    by_ticker_stats = defaultdict(lambda: {
        "trades": 0, "wins": 0, "correct_dir": 0,
        "pnl_total": 0.0, "return_pct_total": 0.0,
        "days_held_total": 0,
    })
    open_positions = {}  # ticker -> {entry, direction, stop, extreme, capital, opened_date, days_held}

    # Все календарные даты от первой до последней
    trading_dates = sorted({d for t in tickers for d in ohlc[t].keys()})
    trading_dates = [d for d in trading_dates if date_from <= d <= date_to]

    for day in trading_dates:
        day_pnl = 0.0

        # === фаза 1: обработка открытых позиций (updates + exits) ===
        to_close = []
        for ticker, pos in open_positions.items():
            candle = ohlc[ticker].get(day)
            if not candle:
                continue
            _, high, low, close = candle
            atr = atr_map[ticker].get(day) or atr_map[ticker].get(
                max((d for d in atr_map[ticker] if d < day), default=None)
            )
            if atr is None:
                atr = pos["entry"] * 0.015  # 1.5% фолбэк если нет ATR

            # обновляем экстремум
            if pos["direction"] == "BUY":
                pos["extreme"] = max(pos["extreme"], high)
                new_stop = pos["extreme"] - atr_mult_trail * atr
                pos["stop"] = max(pos["stop"], new_stop)
            else:
                pos["extreme"] = min(pos["extreme"], low)
                new_stop = pos["extreme"] + atr_mult_trail * atr
                pos["stop"] = min(pos["stop"], new_stop)

            pos["days_held"] += 1

            # проверка выходов в приоритетном порядке
            exit_price = None
            exit_reason = None

            # 1) trailing stop
            if pos["direction"] == "BUY" and low <= pos["stop"]:
                exit_price, exit_reason = pos["stop"], "trailing"
            elif pos["direction"] == "SELL" and high >= pos["stop"]:
                exit_price, exit_reason = pos["stop"], "trailing"

            # 2) signal flip (только если сегодня есть новый вердикт)
            if exit_price is None:
                today_v = verdicts.get((day, ticker))
                if today_v and today_v["verdict"] in ("BUY", "SELL"):
                    if today_v["verdict"] != pos["direction"] and \
                            CONFIDENCE_LEVEL.get(today_v["confidence"], 0) >= 2:
                        exit_price, exit_reason = close, "signal_flip"

            # 3) time limit
            if exit_price is None and pos["days_held"] >= max_hold_days:
                exit_price, exit_reason = close, "time_limit"

            if exit_price is not None:
                if pos["direction"] == "BUY":
                    gross_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
                else:
                    gross_pct = (pos["entry"] - exit_price) / pos["entry"] * 100
                net_pct = gross_pct - 2 * commission_pct
                pnl = pos["capital"] * net_pct / 100.0
                day_pnl += pnl

                actual_dir_correct = (
                    (pos["direction"] == "BUY" and close >= pos["entry"]) or
                    (pos["direction"] == "SELL" and close <= pos["entry"])
                )
                trades.append({
                    "date": day.isoformat(),
                    "opened_date": pos["opened_date"].isoformat(),
                    "ticker": ticker,
                    "verdict": pos["direction"],
                    "confidence": pos["confidence"],
                    "entry_price": round(pos["entry"], 2),
                    "exit_price": round(exit_price, 2),
                    "exit_reason": exit_reason,
                    "days_held": pos["days_held"],
                    "gross_return_pct": round(gross_pct, 2),
                    "net_return_pct": round(net_pct, 2),
                    "pnl_rub": round(pnl, 2),
                    "capital_at_trade": round(pos["capital"], 2),
                    "correct_direction": actual_dir_correct,
                })
                st = by_ticker_stats[ticker]
                st["trades"] += 1
                if net_pct > 0:
                    st["wins"] += 1
                if actual_dir_correct:
                    st["correct_dir"] += 1
                st["pnl_total"] += pnl
                st["return_pct_total"] += net_pct
                st["days_held_total"] += pos["days_held"]
                to_close.append(ticker)

        for t in to_close:
            del open_positions[t]

        # обновляем equity после закрытий (влияет на размер новых позиций)
        equity += day_pnl

        # === фаза 2: открываем новые позиции по свежим сигналам ===
        day_signals = [(t, verdicts.get((day, t))) for t in tickers
                       if verdicts.get((day, t))]
        # только те, что прошли фильтры и по которым нет открытой позиции
        new_entries = []
        for ticker, v in day_signals:
            if ticker in open_positions:
                continue
            if v["verdict"] not in ("BUY", "SELL"):
                continue
            if CONFIDENCE_LEVEL.get(v["confidence"], 0) < min_level:
                continue
            candle = ohlc[ticker].get(day)
            if not candle:
                continue
            entry_p = v["entry_price"] or v["current_price"]
            if not entry_p or entry_p <= 0:
                continue
            # fill-check по high/low
            _, high, low, close = candle
            filled = (v["verdict"] == "BUY" and low <= entry_p) or \
                     (v["verdict"] == "SELL" and high >= entry_p)
            if not filled:
                continue
            new_entries.append((ticker, v, entry_p, candle))

        if new_entries:
            # равномерное распределение свободного капитала
            free_slots = len(new_entries)
            # доступный капитал = equity минус то что уже в открытых позициях
            capital_in_positions = sum(p["capital"] for p in open_positions.values())
            free_capital = max(0.0, equity - capital_in_positions)
            per_slot = free_capital / free_slots if free_slots else 0

            for ticker, v, entry_p, candle in new_entries:
                if per_slot <= 0:
                    continue
                atr = atr_map[ticker].get(day) or atr_map[ticker].get(
                    max((d for d in atr_map[ticker] if d < day), default=None)
                )
                if atr is None:
                    atr = entry_p * 0.015

                if v["verdict"] == "BUY":
                    initial_stop = entry_p - atr_mult_stop * atr
                    extreme = entry_p
                else:
                    initial_stop = entry_p + atr_mult_stop * atr
                    extreme = entry_p

                open_positions[ticker] = {
                    "entry": entry_p,
                    "direction": v["verdict"],
                    "confidence": v["confidence"],
                    "stop": initial_stop,
                    "extreme": extreme,
                    "capital": per_slot,
                    "opened_date": day,
                    "days_held": 0,
                }

        equity_curve.append({
            "date": day.isoformat(),
            "equity": round(equity, 2),
            "ret_pct": round((equity / initial_capital - 1) * 100, 2),
        })

    # закрываем "висящие" позиции на последнюю известную цену
    if open_positions:
        last_day = trading_dates[-1] if trading_dates else date_from
        for ticker, pos in list(open_positions.items()):
            candle = ohlc[ticker].get(last_day)
            if not candle:
                continue
            _, _, _, close = candle
            if pos["direction"] == "BUY":
                gross_pct = (close - pos["entry"]) / pos["entry"] * 100
            else:
                gross_pct = (pos["entry"] - close) / pos["entry"] * 100
            net_pct = gross_pct - 2 * commission_pct
            pnl = pos["capital"] * net_pct / 100.0
            equity += pnl
            trades.append({
                "date": last_day.isoformat(),
                "opened_date": pos["opened_date"].isoformat(),
                "ticker": ticker,
                "verdict": pos["direction"],
                "confidence": pos["confidence"],
                "entry_price": round(pos["entry"], 2),
                "exit_price": round(close, 2),
                "exit_reason": "unresolved",
                "days_held": pos["days_held"],
                "gross_return_pct": round(gross_pct, 2),
                "net_return_pct": round(net_pct, 2),
                "pnl_rub": round(pnl, 2),
                "capital_at_trade": round(pos["capital"], 2),
                "correct_direction": None,
            })

    # === Сводка ===
    n_trades = len(trades)
    winning = sum(1 for t in trades if t["net_return_pct"] > 0)
    correct_dir_total = sum(1 for t in trades if t["correct_direction"])
    best = max((t["net_return_pct"] for t in trades), default=None)
    worst = min((t["net_return_pct"] for t in trades), default=None)
    avg_hold = (sum(t["days_held"] for t in trades) / n_trades) if n_trades else None

    exits_count = defaultdict(int)
    for t in trades:
        exits_count[t["exit_reason"]] += 1

    summary = {
        "initial_capital": initial_capital,
        "final_equity": round(equity, 2),
        "total_return_pct": round((equity / initial_capital - 1) * 100, 2),
        "trades": n_trades,
        "winning_trades": winning,
        "win_rate_pct": round(winning / n_trades * 100, 1) if n_trades else None,
        "correct_direction_trades": correct_dir_total,
        "direction_accuracy_pct": round(correct_dir_total / n_trades * 100, 1) if n_trades else None,
        "best_trade_pct": best,
        "worst_trade_pct": worst,
        "commission_pct": commission_pct,
        "min_confidence": min_confidence,
        "max_hold_days": max_hold_days,
        "atr_mult_stop": atr_mult_stop,
        "atr_mult_trail": atr_mult_trail,
        "avg_holding_days": round(avg_hold, 1) if avg_hold else None,
        "days_simulated": len(equity_curve),
        "exits": {
            "trailing": exits_count["trailing"],
            "signal_flip": exits_count["signal_flip"],
            "time_limit": exits_count["time_limit"],
            "unresolved": exits_count["unresolved"],
        },
    }

    by_ticker = {}
    for ticker, st in by_ticker_stats.items():
        by_ticker[ticker] = {
            "trades": st["trades"],
            "wins": st["wins"],
            "win_rate_pct": round(st["wins"] / st["trades"] * 100, 1) if st["trades"] else None,
            "correct_dir": st["correct_dir"],
            "direction_accuracy_pct": round(st["correct_dir"] / st["trades"] * 100, 1) if st["trades"] else None,
            "pnl_rub": round(st["pnl_total"], 2),
            "avg_return_pct": round(st["return_pct_total"] / st["trades"], 2) if st["trades"] else None,
            "avg_holding_days": round(st["days_held_total"] / st["trades"], 1) if st["trades"] else None,
        }

    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "trades": trades,
        "by_ticker": by_ticker,
        "available_tickers": all_tickers,
        "excluded_tickers": sorted(exclude_tickers),
    }


def _empty_result(initial_capital, commission_pct, min_confidence):
    return {
        "summary": {
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "total_return_pct": 0.0,
            "trades": 0, "winning_trades": 0,
            "win_rate_pct": None,
            "best_trade_pct": None, "worst_trade_pct": None,
            "commission_pct": commission_pct,
            "min_confidence": min_confidence,
            "avg_holding_days": None,
            "days_simulated": 0,
            "exits": {"trailing": 0, "signal_flip": 0, "time_limit": 0, "unresolved": 0},
        },
        "equity_curve": [],
        "trades": [],
        "by_ticker": {},
    }
