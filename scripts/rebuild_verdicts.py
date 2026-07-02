"""
Пересчёт correct_direction для всех уже закрытых ticker_reports.

Раньше резолвер сравнивал actual_close с current_price (цена на момент
отчёта). Теперь — с entry_price (цена входа, названная LLM), с фолбэком
на current_price. Из-за этого старые записи могли получить неправильную
метку "корректно/некорректно".

Скрипт идемпотентный — можно запускать сколько угодно раз.
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def main():
    url = (
        f"postgresql://{os.getenv('DB_USER','postgres')}:"
        f"{os.getenv('DB_PASSWORD','postgres')}@"
        f"{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}/"
        f"{os.getenv('DB_NAME','postgres')}"
    )
    engine = create_engine(url)
    with engine.begin() as conn:
        # Fill-check: сделка открылась, только если цена реально достигла
        # entry_price в течение дня (для BUY: low<=entry, для SELL: high>=entry).
        # Если нет — correct_direction=NULL (UI: "цена входа не достигнута").
        result = conn.execute(text("""
            UPDATE public.ticker_reports
            SET correct_direction = CASE
                -- BUY: fill требует low <= entry
                WHEN verdict = 'BUY' AND actual_low IS NOT NULL
                     AND actual_low > COALESCE(entry_price, current_price) THEN NULL
                -- SELL: fill требует high >= entry
                WHEN verdict = 'SELL' AND actual_high IS NOT NULL
                     AND actual_high < COALESCE(entry_price, current_price) THEN NULL
                WHEN verdict = 'BUY'  AND actual_close > COALESCE(entry_price, current_price) THEN true
                WHEN verdict = 'SELL' AND actual_close < COALESCE(entry_price, current_price) THEN true
                WHEN verdict IN ('BUY','SELL') THEN false
                WHEN verdict = 'HOLD' AND COALESCE(entry_price, current_price) IS NOT NULL
                     AND abs(actual_close - COALESCE(entry_price, current_price))
                         / COALESCE(entry_price, current_price) <= 0.005 THEN true
                WHEN verdict = 'HOLD' THEN false
                ELSE correct_direction
            END
            WHERE actual_close IS NOT NULL
              AND COALESCE(entry_price, current_price) IS NOT NULL
        """))
        print(f"Обновлено записей: {result.rowcount}")


if __name__ == "__main__":
    main()
