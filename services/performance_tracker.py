"""
Подсчёт "живой" точности моделей за последние N дней.

В отличие от R² на старом тестовом сете (которая фиксируется в момент
обучения и протухает), эта метрика смотрит на реальные прогнозы из
public.model_results и сверяет с фактической ценой на следующий день.

Используется в generate_ticker_report чтобы LLM получала актуальные
показатели качества, а не оценки трёхмесячной давности.
"""
from typing import Optional, Dict, List

from sqlalchemy import text


def compute_recent_hit_rates(engine, figi: str,
                              windows: tuple = (5, 30)) -> Dict[str, Dict[int, dict]]:
    """
    Возвращает hit rate каждой модели за последние N дней (N из `windows`).

    Returns:
        {
          "ridge": {
            5:  {"hit_rate": 60.0, "correct": 3, "total": 5},
            30: {"hit_rate": 54.5, "correct": 12, "total": 22},
          },
          ...
        }
    """
    # Получаем все исторические свечи для тикера (даты + close)
    candles = _load_candle_dates_to_close(engine, figi)
    if not candles:
        return {}
    dates_sorted = sorted(candles.keys())

    def next_close_after(date_iso: str) -> Optional[float]:
        """Возвращает close ПЕРВОГО торгового дня строго после date_iso."""
        for d in dates_sorted:
            if d > date_iso:
                return candles[d]
        return None

    max_window = max(windows)

    # Берём все прогнозы по тикеру за период покрытия (с запасом)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT model_name, timestamp::date AS d,
                   current_price, predicted_price
            FROM public.model_results
            WHERE db_name = :figi
              AND predicted_price IS NOT NULL
              AND current_price IS NOT NULL
              AND timestamp >= NOW() - (:days || ' days')::interval
            ORDER BY timestamp DESC
        """), {"figi": figi, "days": max_window * 2}).fetchall()

    # Группируем: для каждой модели — список (date, correct?)
    by_model: Dict[str, List[tuple]] = {}
    for model_name, d, cur, pred in rows:
        d_iso = d.isoformat()
        actual_next = next_close_after(d_iso)
        if actual_next is None:
            continue
        cur_f = float(cur)
        pred_f = float(pred)
        # дедуп: только последний прогноз за день (rows уже DESC, первый — самый свежий)
        pred_up = pred_f > cur_f
        actual_up = actual_next > cur_f
        correct = (pred_up == actual_up)
        by_model.setdefault(model_name, []).append((d_iso, correct))

    # Дедупликация: один прогноз на (model, date) — берём первый встретившийся
    # (т.к. сортировка DESC, это и есть самый свежий за день).
    deduped: Dict[str, Dict[str, bool]] = {}
    for model, items in by_model.items():
        for d_iso, correct in items:
            deduped.setdefault(model, {}).setdefault(d_iso, correct)

    # Считаем hit rate в каждом окне
    result: Dict[str, Dict[int, dict]] = {}
    for model, dates_map in deduped.items():
        # Сортируем по дате DESC и берём последние N
        items = sorted(dates_map.items(), key=lambda x: x[0], reverse=True)
        result[model] = {}
        for w in windows:
            window_items = items[:w]
            n = len(window_items)
            if n == 0:
                result[model][w] = {"hit_rate": None, "correct": 0, "total": 0}
                continue
            correct = sum(1 for _, c in window_items if c)
            result[model][w] = {
                "hit_rate": round(correct / n * 100, 1),
                "correct": correct,
                "total": n,
            }
    return result


def _load_candle_dates_to_close(engine, figi: str, days: int = 90) -> Dict[str, float]:
    """{ '2026-06-27': 297.61, ... } за последние N дней."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                f'SELECT timestamp::date AS d, close FROM all_dfs."{figi}" '
                f"WHERE timestamp >= NOW() - (:days || ' days')::interval "
                f"ORDER BY timestamp"
            ), {"days": days}).fetchall()
        return {r[0].isoformat(): float(r[1]) for r in rows}
    except Exception:
        return {}
