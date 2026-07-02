#!/usr/bin/env python3
"""
Патч последней записи OZON в ticker_reports:
- entry_price → 4421
- correct_direction пересчитывается относительно новой цены входа
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()
import psycopg2

NEW_ENTRY = 4421.0

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME", "postgres"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
)
cur = conn.cursor()

cur.execute("""
    SELECT id, prediction_date, verdict, current_price, entry_price,
           target_price, stop_loss, actual_close, actual_high, actual_low
    FROM public.ticker_reports
    WHERE ticker = 'OZON'
    ORDER BY timestamp DESC
    LIMIT 1
""")
row = cur.fetchone()
if not row:
    print("Записей OZON не найдено")
    conn.close()
    sys.exit(1)

(rid, rdate, verdict, cur_price, entry, target, stop,
 actual_close, actual_high, actual_low) = row

print("=== Текущая запись ===")
print(f"  id              : {rid}")
print(f"  prediction_date : {rdate}")
print(f"  verdict       : {verdict}")
print(f"  current_price : {cur_price}")
print(f"  entry_price   : {entry}")
print(f"  target_price  : {target}")
print(f"  stop_loss     : {stop}")
print(f"  actual_close  : {actual_close}")
print(f"  actual_high   : {actual_high}")
print(f"  actual_low    : {actual_low}")
print()

# Проверяем fill: цена должна была дойти до NEW_ENTRY
if actual_low is None or actual_high is None:
    print("WARN: нет intraday данных (high/low), fill-check пропускается")
    filled = True
elif verdict == "BUY":
    filled = float(actual_low) <= NEW_ENTRY
elif verdict == "SELL":
    filled = float(actual_high) >= NEW_ENTRY
else:
    filled = False

print(f"Fill-check (вошли ли по {NEW_ENTRY}): {'ДА' if filled else 'НЕТ — цена не дошла!'}")
if not filled:
    print("Обновление не выполнено.")
    conn.close()
    sys.exit(1)

# Пересчитываем correct_direction относительно новой цены входа
correct_dir = None
if actual_close is not None:
    ac = float(actual_close)
    if verdict == "BUY":
        correct_dir = ac >= NEW_ENTRY
    elif verdict == "SELL":
        correct_dir = ac <= NEW_ENTRY

print(f"correct_direction → {correct_dir}")
print()

cur.execute("""
    UPDATE public.ticker_reports
    SET entry_price = %s,
        correct_direction = %s
    WHERE id = %s
""", (NEW_ENTRY, correct_dir, rid))

conn.commit()
print(f"✓ Запись {rid} обновлена: entry_price={NEW_ENTRY}, correct_direction={correct_dir}")
conn.close()
