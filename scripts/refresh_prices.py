"""
Обновляет current_price и predicted_price в model_results / ticker_reports
из АКТУАЛЬНЫХ свечей в all_dfs.<figi>. Не трогает модели и не переобучает.

Применение: когда модели были обучены на partial candle (intraday-цена),
Tinkoff потом обновил эту свечу, и в "Было" висит устаревшее значение.

Логика:
  current_price ← close из all_dfs за data_end_date (последний полный день)
  predicted_price ← current_price * (1 + expected_change / 100)
  (expected_change — это "сырое" значение прогноза модели в %, оно НЕ меняется)
"""
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', '')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    return create_engine(url)


def _effective_data_date(engine, figi: str, training_date):
    """
    Берём последний полный день в all_dfs для FIGI, ≤ training_date.
    Если рынок открыт сейчас И training_date == сегодня — отбрасываем
    сегодняшнюю запись.
    """
    MSK = timezone(timedelta(hours=3))
    now_msk = datetime.now(MSK)
    today_msk = now_msk.date()
    market_open = now_msk.hour < 20 and now_msk.weekday() < 5

    with engine.connect() as conn:
        # Если рынок открыт — пропускаем сегодняшнюю свечу
        if market_open:
            row = conn.execute(text(
                f'SELECT timestamp::date, close FROM all_dfs."{figi}" '
                f"WHERE timestamp::date < :today "
                f"ORDER BY timestamp DESC LIMIT 1"
            ), {"today": today_msk}).fetchone()
        else:
            row = conn.execute(text(
                f'SELECT timestamp::date, close FROM all_dfs."{figi}" '
                f"ORDER BY timestamp DESC LIMIT 1"
            )).fetchone()
        if row:
            return row[0], float(row[1])
    return None, None


def refresh_model_results(engine):
    """Обновляет current_price + predicted_price для всех model_results."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, db_name, current_price, predicted_price, expected_change
            FROM public.model_results
            WHERE current_price IS NOT NULL
              AND expected_change IS NOT NULL
        """)).fetchall()

    log.info(f"Обновляю {len(rows)} записей model_results")

    updated = 0
    skipped = 0
    cache: dict = {}  # figi -> (date, close)

    with engine.begin() as conn:
        for row in rows:
            row_id, figi, old_cur, old_pred, exp_change = row

            if figi not in cache:
                cache[figi] = _effective_data_date(engine, figi, None)
            new_date, new_close = cache[figi]

            if new_close is None:
                skipped += 1
                continue

            # Пересчитываем predicted_price из expected_change (сырой прогноз)
            new_predicted = new_close * (1 + float(exp_change) / 100.0)

            # Только если значения реально поменялись (на >0.01)
            if abs(float(old_cur) - new_close) < 0.01:
                continue

            conn.execute(text("""
                UPDATE public.model_results
                SET current_price = :cp,
                    predicted_price = :pp,
                    data_end_date = :de
                WHERE id = :id
            """), {
                "cp": new_close,
                "pp": new_predicted,
                "de": new_date,
                "id": row_id,
            })
            updated += 1

    log.info(f"  Обновлено: {updated}, пропущено (нет свечей): {skipped}")


def refresh_ticker_reports(engine):
    """Обновляет current_price + entry/target/stop для ticker_reports."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, figi, current_price, entry_price, target_price, stop_loss
            FROM public.ticker_reports
            WHERE current_price IS NOT NULL
        """)).fetchall()

    log.info(f"Обновляю {len(rows)} записей ticker_reports")

    updated = 0
    cache: dict = {}

    with engine.begin() as conn:
        for row in rows:
            rep_id, figi, old_cur, entry, target, stop = row

            if figi not in cache:
                cache[figi] = _effective_data_date(engine, figi, None)
            new_date, new_close = cache[figi]
            if new_close is None:
                continue

            if abs(float(old_cur) - new_close) < 0.01:
                continue

            # Пропорционально пересчитываем целевые уровни от старого current_price
            ratio = new_close / float(old_cur) if float(old_cur) else 1.0
            new_entry = float(entry) * ratio if entry is not None else None
            new_target = float(target) * ratio if target is not None else None
            new_stop = float(stop) * ratio if stop is not None else None

            conn.execute(text("""
                UPDATE public.ticker_reports
                SET current_price = :cp,
                    entry_price = :ep,
                    target_price = :tp,
                    stop_loss = :sl,
                    data_date = :dd
                WHERE id = :id
            """), {
                "cp": new_close, "ep": new_entry, "tp": new_target, "sl": new_stop,
                "dd": new_date, "id": rep_id,
            })
            updated += 1

    log.info(f"  Обновлено: {updated}")


def main():
    engine = get_engine()
    log.info("=" * 60)
    log.info("Refresh prices — обновление БЕЗ переобучения")
    log.info("=" * 60)
    refresh_model_results(engine)
    refresh_ticker_reports(engine)
    log.info("Готово")


if __name__ == "__main__":
    main()
