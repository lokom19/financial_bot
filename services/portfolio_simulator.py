"""
Симулятор портфеля — "что было бы с деньгами, если бы следовать AI-вердиктам".

Логика:
- Депозит делится поровну между активными тикерами (3 → ~33% каждому)
- Каждый день для каждого тикера смотрим вердикт LLM (BUY/SELL/HOLD)
- BUY: купить по entry_price на закрытии вчерашнего → продать на закрытии prediction_date
       P&L = (actual_close - entry) / entry × capital_per_ticker
- SELL: short — заработок если цена пошла вниз
       P&L = (entry - actual_close) / entry × capital_per_ticker
- HOLD/NEUTRAL: остаёмся в кэше, P&L = 0
- Учитываем target_hit / stop_hit: если цель достигнута — фиксация на target_price;
  если стоп пробит — закрытие на stop_loss
- Все вердикты с одинаковым prediction_date агрегируются как "торговый день"

Возвращает equity curve + список сделок + сводка.
"""
import logging
from collections import defaultdict
from datetime import date as _date

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Комиссия Мосбиржи + брокера в одну сторону (round trip x2)
DEFAULT_COMMISSION_PCT = 0.05  # 0.05% за сделку (комиссия Tinkoff "Инвестор")


def simulate(engine, initial_capital: float = 100_000.0,
             commission_pct: float = DEFAULT_COMMISSION_PCT,
             only_closed: bool = True) -> dict:
    """
    Симулирует торговлю по всем AI-вердиктам из ticker_reports.

    Args:
        initial_capital: стартовый депозит (RUB)
        commission_pct: комиссия за сделку (в %, в одну сторону)
        only_closed: учитывать только закрытые позиции (actual_close IS NOT NULL)

    Returns:
        {
            "summary": { ... },
            "equity_curve": [ {"date": "...", "equity": 110_500.0, "ret_pct": 10.5}, ... ],
            "trades": [ {"date": "...", "ticker": "...", "verdict": "...", "pnl": ..., ...}, ... ],
            "by_ticker": { "SBER": {...}, "OZON": {...}, "VTBR": {...} },
        }
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT prediction_date, ticker, verdict, confidence,
                   current_price, entry_price, target_price, stop_loss,
                   actual_close, actual_high, actual_low,
                   correct_direction, target_hit, stop_hit
            FROM (
                -- Один вердикт на (ticker, prediction_date) — последний по timestamp
                SELECT DISTINCT ON (figi, prediction_date) *
                FROM public.ticker_reports
                WHERE prediction_date IS NOT NULL
                ORDER BY figi, prediction_date, timestamp DESC
            ) latest
            WHERE (NOT :only_closed OR actual_close IS NOT NULL)
            ORDER BY prediction_date ASC, ticker ASC
        """), {"only_closed": only_closed}).fetchall()

    if not rows:
        return {
            "summary": {
                "initial_capital": initial_capital,
                "final_equity": initial_capital,
                "total_return_pct": 0.0,
                "trades": 0,
                "winning_trades": 0,
                "win_rate_pct": None,
                "best_trade_pct": None,
                "worst_trade_pct": None,
                "commission_pct": commission_pct,
            },
            "equity_curve": [],
            "trades": [],
            "by_ticker": {},
        }

    # Группируем по prediction_date (общая дата торгов)
    by_day = defaultdict(list)
    for r in rows:
        by_day[r[0]].append(r)

    equity = initial_capital
    equity_curve = []
    trades = []

    # Стартовая точка кривой — день ДО первой сделки с equity = initial_capital.
    # Без неё, когда все сделки в один день, curve состоит из 1 точки и canvas
    # не рисует линию (только заливка тянется к углам и выглядит как падение).
    from datetime import timedelta as _td
    first_day = min(by_day.keys())
    equity_curve.append({
        "date": (first_day - _td(days=1)).isoformat(),
        "equity": round(equity, 2),
        "ret_pct": 0.0,
    })
    by_ticker_stats = defaultdict(lambda: {
        "trades": 0, "wins": 0,
        "pnl_total": 0.0, "return_pct_total": 0.0,
    })

    # Идём по торговым дням в хронологическом порядке
    for trade_date in sorted(by_day.keys()):
        day_rows = by_day[trade_date]

        # На каждый день: вычисляем сколько тикеров с активным сигналом BUY/SELL
        active = [r for r in day_rows if r[2] in ("BUY", "SELL")]
        n_active = len(active)
        if n_active == 0:
            # Все HOLD/NEUTRAL — деньги в кэше, equity не меняется
            equity_curve.append({
                "date": trade_date.isoformat(),
                "equity": round(equity, 2),
                "ret_pct": round((equity / initial_capital - 1) * 100, 2),
            })
            continue

        # Капитал на каждую активную сделку = equity / n_active
        per_trade_capital = equity / n_active
        day_pnl = 0.0

        for r in day_rows:
            (pdate, ticker, verdict, confidence, cur_price, entry, target, stop,
             actual_close, actual_high, actual_low,
             correct_dir, target_hit_flag, stop_hit_flag) = r

            if verdict not in ("BUY", "SELL"):
                continue

            # Цена входа: entry_price если задан LLM-ом, иначе current_price
            entry_p = float(entry) if entry is not None else float(cur_price)
            if entry_p <= 0:
                continue

            # Определяем цену выхода с учётом стопа/цели:
            # - Для BUY: long — если high >= target → закрываемся по target
            #            если low <= stop → закрываемся по stop
            #            иначе — по actual_close
            # - Для SELL: short — наоборот
            close_price = None
            exit_reason = "close"

            if actual_close is None:
                continue  # ещё не закрылось, нечего считать

            high = float(actual_high) if actual_high is not None else float(actual_close)
            low = float(actual_low) if actual_low is not None else float(actual_close)

            # Валидные значения target/stop относительно entry_p.
            # LLM иногда путает: указывает stop ниже entry для SELL или
            # выше entry для BUY — такой "стоп" физически не защитный.
            # Игнорируем такие числа, чтобы не засчитывать фиктивные выходы.
            def _v(x):
                return float(x) if x is not None and float(x) > 0 else None
            target_v = _v(target)
            stop_v = _v(stop)

            if verdict == "BUY":
                # BUY: цель ВЫШЕ входа, стоп НИЖЕ
                if target_v is not None and target_v <= entry_p:
                    target_v = None
                if stop_v is not None and stop_v >= entry_p:
                    stop_v = None
                if stop_v is not None and low <= stop_v:
                    close_price = stop_v
                    exit_reason = "stop"
                elif target_v is not None and high >= target_v:
                    close_price = target_v
                    exit_reason = "target"
                else:
                    close_price = float(actual_close)
                gross_return_pct = (close_price - entry_p) / entry_p * 100
            else:  # SELL
                # SELL: цель НИЖЕ входа, стоп ВЫШЕ
                if target_v is not None and target_v >= entry_p:
                    target_v = None
                if stop_v is not None and stop_v <= entry_p:
                    stop_v = None
                if stop_v is not None and high >= stop_v:
                    close_price = stop_v
                    exit_reason = "stop"
                elif target_v is not None and low <= target_v:
                    close_price = target_v
                    exit_reason = "target"
                else:
                    close_price = float(actual_close)
                gross_return_pct = (entry_p - close_price) / entry_p * 100

            # Round-trip комиссия (вход + выход)
            net_return_pct = gross_return_pct - 2 * commission_pct
            pnl_rub = per_trade_capital * net_return_pct / 100.0
            day_pnl += pnl_rub

            trades.append({
                "date": trade_date.isoformat(),
                "ticker": ticker,
                "verdict": verdict,
                "confidence": confidence,
                "entry_price": entry_p,
                "exit_price": close_price,
                "exit_reason": exit_reason,
                "gross_return_pct": round(gross_return_pct, 2),
                "net_return_pct": round(net_return_pct, 2),
                "pnl_rub": round(pnl_rub, 2),
                "capital_at_trade": round(per_trade_capital, 2),
            })

            # Статистика по тикеру
            st = by_ticker_stats[ticker]
            st["trades"] += 1
            if net_return_pct > 0:
                st["wins"] += 1
            st["pnl_total"] += pnl_rub
            st["return_pct_total"] += net_return_pct

        equity += day_pnl
        equity_curve.append({
            "date": trade_date.isoformat(),
            "equity": round(equity, 2),
            "ret_pct": round((equity / initial_capital - 1) * 100, 2),
        })

    # Сводка
    n_trades = len(trades)
    winning = sum(1 for t in trades if t["net_return_pct"] > 0)
    best = max((t["net_return_pct"] for t in trades), default=None)
    worst = min((t["net_return_pct"] for t in trades), default=None)

    summary = {
        "initial_capital": initial_capital,
        "final_equity": round(equity, 2),
        "total_return_pct": round((equity / initial_capital - 1) * 100, 2),
        "trades": n_trades,
        "winning_trades": winning,
        "win_rate_pct": round(winning / n_trades * 100, 1) if n_trades else None,
        "best_trade_pct": best,
        "worst_trade_pct": worst,
        "commission_pct": commission_pct,
        "days_simulated": len(equity_curve),
    }

    # Финальная статистика по тикерам
    by_ticker = {}
    for ticker, st in by_ticker_stats.items():
        by_ticker[ticker] = {
            "trades": st["trades"],
            "wins": st["wins"],
            "win_rate_pct": round(st["wins"] / st["trades"] * 100, 1) if st["trades"] else None,
            "pnl_rub": round(st["pnl_total"], 2),
            "avg_return_pct": round(st["return_pct_total"] / st["trades"], 2) if st["trades"] else None,
        }

    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "trades": trades,
        "by_ticker": by_ticker,
    }
