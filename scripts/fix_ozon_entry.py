#!/usr/bin/env python3
"""
Патч записи OZON в ticker_reports.

Использование:
  python scripts/fix_ozon_entry.py            # показать последние 5 записей OZON
  python scripts/fix_ozon_entry.py --id 27    # обновить запись с id=27
"""
import os, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()
import psycopg2

NEW_ENTRY = 4421.0

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=None, help="id записи для обновления")
args = parser.parse_args()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME", "postgres"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
)
cur = conn.cursor()

# Если id не указан — просто показываем последние 5 записей
if args.id is None:
    cur.execute("""
        SELECT id, prediction_date, verdict, current_price, entry_price,
               target_price, stop_loss, actual_close, actual_high, actual_low, correct_direction
        FROM public.ticker_reports
        WHERE ticker = 'OZON'
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    rows = cur.fetchall()
    print(f"{'id':>5}  {'date':<12}  {'verdict':<6}  {'cur_price':>10}  {'entry':>10}  {'target':>10}  {'stop':>10}  {'close':>8}  {'correct'}")
    for r in rows:
        print(f"{r[0]:>5}  {str(r[1]):<12}  {str(r[2]):<6}  {str(r[3] or ''):>10}  {str(r[4] or ''):>10}  {str(r[5] or ''):>10}  {str(r[6] or ''):>10}  {str(r[7] or ''):>8}  {r[10]}")
    print("\nЗапусти с --id <нужный id> чтобы обновить запись.")
    conn.close()
    sys.exit(0)

# Обновляем конкретную запись
cur.execute("""
    SELECT id, prediction_date, verdict, current_price, entry_price,
           target_price, stop_loss, actual_close, actual_high, actual_low
    FROM public.ticker_reports
    WHERE id = %s AND ticker = 'OZON'
""", (args.id,))
row = cur.fetchone()
if not row:
    print(f"Запись OZON с id={args.id} не найдена")
    conn.close()
    sys.exit(1)

(rid, rdate, verdict, cur_price, entry, target, stop,
 actual_close, actual_high, actual_low) = row

print("=== Текущая запись ===")
print(f"  id              : {rid}")
print(f"  prediction_date : {rdate}")
print(f"  verdict         : {verdict}")
print(f"  current_price   : {cur_price}")
print(f"  entry_price     : {entry}")
print(f"  target_price    : {target}")
print(f"  stop_loss       : {stop}")
print(f"  actual_close    : {actual_close}")
print(f"  actual_high     : {actual_high}")
print(f"  actual_low      : {actual_low}")
print()

# Fill-check
if actual_low is None or actual_high is None:
    print("WARN: нет intraday данных, fill-check пропускается")
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

# correct_direction относительно новой цены входа
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
