"""
Определение следующего торгового дня для конкретного тикера.

Раньше делали по календарной формуле (пятница → +3 → понедельник), но
Мосбиржа с 2024 года торгует и по выходным по ряду инструментов, а по
праздникам — не всегда. Календарь неточен → prediction_date уезжает,
резолвинг оценки направления бьёт мимо реальной свечи.

Правильный подход — искать первую свечу с datой строго после data_date
прямо в all_dfs.<figi>. Если такой ещё нет — используем календарный
fallback (+1 рабочий день), потому что nightly ещё не тянул свежие данные.
"""
from datetime import date as _date, timedelta as _td
from typing import Optional

from sqlalchemy import text


def next_trading_day(engine, figi: str, data_date) -> Optional[_date]:
    """
    Возвращает дату следующей торговой сессии после ``data_date`` для
    инструмента ``figi``. Определяется по фактическому наличию свечи
    в all_dfs.<figi>.

    Если ни одной свечи после data_date ещё нет — возвращаем
    "следующий будний день" как безопасный дефолт (для nightly, который
    формирует прогноз ещё до появления реальной свечи).
    """
    if data_date is None:
        return None
    if isinstance(data_date, str):
        from datetime import datetime as _dt
        try:
            data_date = _dt.fromisoformat(data_date).date()
        except Exception:
            return None

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(f'''
                    SELECT MIN(timestamp::date)
                    FROM all_dfs."{figi}"
                    WHERE timestamp::date > :d
                '''),
                {"d": data_date},
            ).scalar()
            if row is not None:
                return row
    except Exception:
        pass  # таблицы нет / нет прав / кривой FIGI — идём в fallback

    # Fallback: календарный next business day (Mon–Fri).
    # Прогон в 2:00 понедельника после пятницы будет иметь data_date=Fri,
    # а свеча за Mon ещё не сложилась — берём формально понедельник.
    nxt = data_date + _td(days=1)
    while nxt.weekday() >= 5:  # 5=Sat, 6=Sun
        nxt += _td(days=1)
    return nxt
