#!/usr/bin/env python3
"""
Ночной пайплайн обучения + LLM-валидации.

Шаги:
  1) Находит FIGI для SBER, OZON, VTBR в public.tickers.
  2) Запускает train_models.py для этих трёх FIGI (все модели).
  3) Для каждой новой записи без llm_signal — запрашивает LLM
     (services.llm_service.analyze_signal) и сохраняет ответ в БД.

Запуск вручную:
    python scripts/nightly_pipeline.py
    python scripts/nightly_pipeline.py --skip-training   # только LLM по существующим
    python scripts/nightly_pipeline.py --walk-forward    # с walk-forward валидацией
"""
import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import json

from services.llm_service import analyze_signal, generate_ticker_report
from services.ta_indicators import compute_ta_indicators as _compute_ta
from services.performance_tracker import compute_recent_hit_rates
from services.news_service import fetch_news_for_ticker

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("nightly")

TOP_TICKERS = ["SBER", "OZON", "VTBR"]
TRAINING_TIMEOUT = int(os.getenv("NIGHTLY_TRAIN_TIMEOUT", "10800"))  # 3 hours
LLM_LOOKBACK_HOURS = int(os.getenv("NIGHTLY_LLM_LOOKBACK_HOURS", "12"))
# Сколько последних дней дотягивать свечами перед обучением
FETCH_DAYS = int(os.getenv("NIGHTLY_FETCH_DAYS", "7"))
FETCH_TIMEOUT = int(os.getenv("NIGHTLY_FETCH_TIMEOUT", "1800"))  # 30 минут

# Какие модели НЕ обучать ночью. По умолчанию выключаем тяжёлые DL.
# Можно переопределить через env: NIGHTLY_EXCLUDE_MODELS="lstm,tcn,rdpg_lstm"
# Или передать --include-dl чтобы обучать всё.
EXCLUDED_MODELS = [
    m.strip() for m in os.getenv(
        "NIGHTLY_EXCLUDE_MODELS",
        "lstm,tcn,rdpg_lstm"
    ).split(",") if m.strip()
]
# Все доступные модели (синхронизировано с scripts/train_models.py)
ALL_MODELS = [
    "ridge", "huber", "xgboost", "lightgbm", "catboost",
    "random_forest", "extra_trees",
    "arima", "arma_garch", "rf_classifier",
    "prophet", "lstm", "tcn", "rdpg_lstm",
]


def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', '')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    return create_engine(url)


def compute_ta_indicators(engine, figi: str) -> dict:
    """Тонкая обёртка над services.ta_indicators.compute_ta_indicators."""
    return _compute_ta(engine, figi)


def fetch_fresh_candles(days: int = FETCH_DAYS) -> bool:
    """
    Дотягиваем свежие свечи из Tinkoff API в all_dfs.
    Это нужно чтобы прогноз не делался на устаревших данных.
    Вывод стримим построчно (как и в обучении).
    """
    cmd = [sys.executable, "-u", "all_dfs_to_db.py", "--days", str(days), "--interval", "day"]
    logger.info(f"Команда: {' '.join(cmd)}")
    start = datetime.utcnow()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    logger.info(f"[fetch] {line}")
                if (datetime.utcnow() - start).total_seconds() > FETCH_TIMEOUT:
                    proc.kill()
                    logger.error(f"✗ Загрузка превысила таймаут {FETCH_TIMEOUT}s")
                    return False
        finally:
            proc.stdout.close()

        rc = proc.wait()
        elapsed = (datetime.utcnow() - start).total_seconds()
        if rc == 0:
            logger.info(f"✓ Загрузка завершена за {elapsed:.0f}s")
            return True
        logger.warning(f"✗ Загрузка завершилась с кодом {rc} (продолжим всё равно)")
        return False
    except Exception as e:
        logger.error(f"✗ Ошибка fetch: {e}")
        return False


def resolve_ticker_reports(engine):
    """
    Закрывает (resolves) AI-отчёты, для которых prediction_date <= последняя свеча.
    Заполняет actual_close/high/low + correct_direction + target_hit + stop_hit.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    resolved = 0
    try:
        # Берём все отчёты без actual_close
        rows = session.execute(text("""
            SELECT id, figi, prediction_date, current_price, verdict,
                   target_price, stop_loss, entry_price
            FROM public.ticker_reports
            WHERE actual_close IS NULL
              AND prediction_date IS NOT NULL
        """)).fetchall()
        for r in rows:
            rep_id, figi, pred_date, cur_price, verdict, tgt, stop, entry_p = r
            # Ищем свечу за prediction_date (или ближайшую следующую)
            bar = session.execute(text(f"""
                SELECT timestamp::date, close, high, low
                FROM all_dfs."{figi}"
                WHERE timestamp::date >= :pd
                ORDER BY timestamp ASC
                LIMIT 1
            """), {"pd": pred_date}).fetchone()
            if not bar:
                continue
            bar_date, close_p, high_p, low_p = bar
            close_p = float(close_p)
            high_p = float(high_p) if high_p is not None else None
            low_p = float(low_p) if low_p is not None else None
            # Опорная цена для оценки направления — entry_price (та, что LLM
            # назвала "ценой входа"). Если не задана — фолбэк на current_price
            # (цена на момент отчёта). Иначе бывают ситуации типа
            # "BUY вход 3400, close 3421 (рост) → correct=False", потому что
            # на момент отчёта current_price было выше entry.
            entry_p = float(entry_p) if entry_p else None
            cur = float(cur_price) if cur_price else None
            anchor = entry_p or cur

            # Fill-check: сделка открывается только если цена в течение
            # дня реально дошла до entry_price. Если нет — метка направления
            # остаётся NULL (UI покажет "цена входа не достигнута").
            entry_reached = None
            if anchor and verdict in ("BUY", "SELL") and high_p is not None and low_p is not None:
                if verdict == "BUY":
                    entry_reached = low_p <= anchor
                else:  # SELL
                    entry_reached = high_p >= anchor

            # Корректность направления
            correct = None
            if anchor and verdict in ("BUY", "SELL"):
                if entry_reached is False:
                    # цена входа не достигнута — сделка не открылась
                    correct = None
                else:
                    actual_up = close_p > anchor
                    correct = (verdict == "BUY" and actual_up) or \
                              (verdict == "SELL" and not actual_up)
            elif verdict == "HOLD" and anchor and high_p and low_p:
                # HOLD: считаем "корректно", если цена осталась в узком диапазоне ±0.5%
                drift = abs(close_p - anchor) / anchor if anchor else 1.0
                correct = drift <= 0.005

            # target_hit / stop_hit (только следующий день, как договорились).
            # Если сделка не открылась (entry_reached=False) — уровни не оцениваем.
            target_hit = None
            stop_hit = None
            if entry_reached is not False:
                if tgt and high_p is not None and low_p is not None:
                    if verdict == "BUY":
                        target_hit = high_p >= float(tgt)
                    elif verdict == "SELL":
                        target_hit = low_p <= float(tgt)
                if stop and high_p is not None and low_p is not None:
                    if verdict == "BUY":
                        stop_hit = low_p <= float(stop)
                    elif verdict == "SELL":
                        stop_hit = high_p >= float(stop)

            session.execute(text("""
                UPDATE public.ticker_reports
                SET actual_close = :cp,
                    actual_high = :hp,
                    actual_low = :lp,
                    actual_resolved_at = :now,
                    correct_direction = :cd,
                    target_hit = :th,
                    stop_hit = :sh
                WHERE id = :id
            """), {
                "cp": close_p, "hp": high_p, "lp": low_p,
                "now": datetime.utcnow(),
                "cd": correct, "th": target_hit, "sh": stop_hit,
                "id": rep_id,
            })
            resolved += 1
        session.commit()
        logger.info(f"  Закрыто (resolved) AI-отчётов: {resolved}")
    finally:
        session.close()


def generate_and_save_ticker_reports(engine, pairs):
    """
    Для каждого тикера из топ-3 собирает консолидированный AI-отчёт
    (через services.llm_service.generate_ticker_report) и сохраняет
    в public.ticker_reports. Одна запись на тикер на data_date.
    """
    from sqlalchemy import and_
    from datetime import timedelta as _td
    Session = sessionmaker(bind=engine)
    session = Session()
    saved = 0

    try:
        for ticker_name, figi in pairs:
            # Подтянем последние ML-результаты по каждой модели для тикера
            sub = session.execute(text("""
                SELECT model_name, MAX(timestamp) AS max_ts
                FROM public.model_results
                WHERE db_name = :figi
                GROUP BY model_name
            """), {"figi": figi}).fetchall()
            model_keys = [(r[0], r[1]) for r in sub]
            if not model_keys:
                logger.warning(f"  {ticker_name}: нет ML-результатов, пропускаю")
                continue

            # Загружаем сами строки
            models_data = []
            r2_vals, dir_vals = [], []
            current_price = None
            last_ts = None
            for model_name, max_ts in model_keys:
                row = session.execute(text("""
                    SELECT trading_signal, test_r2, test_direction_accuracy,
                           expected_change, predicted_price, current_price, timestamp
                    FROM public.model_results
                    WHERE db_name = :figi AND model_name = :m AND timestamp = :ts
                    LIMIT 1
                """), {"figi": figi, "m": model_name, "ts": max_ts}).fetchone()
                if not row:
                    continue
                if row[5] and current_price is None:
                    current_price = float(row[5])
                if row[6] and (last_ts is None or row[6] > last_ts):
                    last_ts = row[6]
                models_data.append({
                    "model_name": model_name,
                    "signal": row[0],
                    "r2": float(row[1]) if row[1] is not None else None,
                    "direction_accuracy": float(row[2]) if row[2] is not None else None,
                    "expected_change": float(row[3]) if row[3] is not None else None,
                    "predicted_price": float(row[4]) if row[4] is not None else None,
                })
                if row[1] is not None:
                    r2_vals.append(float(row[1]))
                if row[2] is not None:
                    dir_vals.append(float(row[2]))

            # TA
            ta = _compute_ta(engine, figi)
            if current_price is None and ta.get("last_price"):
                current_price = ta["last_price"]

            # Даты
            data_date_iso = ta.get("last_candle_date")
            try:
                from datetime import datetime as _dt
                d = _dt.fromisoformat(data_date_iso).date() if data_date_iso else None
                if d:
                    delta = 3 if d.weekday() == 4 else (2 if d.weekday() == 5 else 1)
                    prediction_d = d + _td(days=delta)
                else:
                    prediction_d = None
            except Exception:
                d = None
                prediction_d = None

            # Подтягиваем новости (smart-lab) — без них LLM в отчёте
            # пишет "новостных данных нет", даже когда на странице виджет
            # их показывает (виджет дёргает API отдельно на лету).
            try:
                news_items = fetch_news_for_ticker(ticker_name, max_items=5)
                if news_items:
                    print(f"   📰 Подтянуто {len(news_items)} новостей по {ticker_name}")
            except Exception as e:
                print(f"   ⚠ Не удалось подтянуть новости по {ticker_name}: {e}")
                news_items = None

            # Запрашиваем LLM на каждом прогоне (свежий вердикт)
            report = generate_ticker_report(
                ticker=ticker_name,
                current_price=current_price or 0.0,
                models_data=models_data,
                ta_indicators=ta,
                news_items=news_items,
                data_date=data_date_iso,
                prediction_date=prediction_d.isoformat() if prediction_d else None,
            )

            params = {
                "figi": figi, "ticker": ticker_name,
                "ts": datetime.utcnow(),
                "dd": d, "pd": prediction_d,
                "cp": current_price,
                "v": report.get("verdict") or "UNKNOWN",
                "conf": report.get("confidence") or "—",
                "ep": report.get("entry_price"),
                "tp": report.get("target_price"),
                "sl": report.get("stop_loss"),
                "r": report.get("report") or report.get("reasoning") or "",
                "sj": json.dumps(report.get("sections") or {}, ensure_ascii=False),
                "mj": json.dumps(models_data, ensure_ascii=False),
                "tj": json.dumps(ta, ensure_ascii=False, default=str),
            }

            # UPSERT: одна запись на (figi, data_date), но всегда свежая.
            # Если за этот data_date уже есть отчёт — обновляем его (актуальный
            # timestamp + актуальные данные/вердикт). Это даёт пользователю
            # видеть "последний прогон ночью" даже если data_date не сменился.
            existing = session.execute(text("""
                SELECT id FROM public.ticker_reports
                WHERE figi = :figi AND data_date = :dd
                LIMIT 1
            """), {"figi": figi, "dd": d}).fetchone()

            if existing:
                params["id"] = existing[0]
                session.execute(text("""
                    UPDATE public.ticker_reports
                    SET timestamp = :ts,
                        prediction_date = :pd,
                        current_price = :cp,
                        verdict = :v, confidence = :conf,
                        entry_price = :ep, target_price = :tp, stop_loss = :sl,
                        reasoning = :r,
                        sections_json = :sj,
                        models_snapshot_json = :mj,
                        ta_snapshot_json = :tj,
                        -- сбрасываем resolved-поля, т.к. вердикт новый
                        actual_close = NULL,
                        actual_high = NULL,
                        actual_low = NULL,
                        actual_resolved_at = NULL,
                        correct_direction = NULL,
                        target_hit = NULL,
                        stop_hit = NULL
                    WHERE id = :id
                """), params)
                logger.info(
                    f"  {ticker_name} (UPDATE id={existing[0]}) → {report.get('verdict')} "
                    f"(conf={report.get('confidence')}, target={report.get('target_price')})"
                )
            else:
                session.execute(text("""
                    INSERT INTO public.ticker_reports (
                        figi, ticker, timestamp, data_date, prediction_date,
                        current_price, verdict, confidence,
                        entry_price, target_price, stop_loss, reasoning,
                        sections_json, models_snapshot_json, ta_snapshot_json
                    ) VALUES (
                        :figi, :ticker, :ts, :dd, :pd,
                        :cp, :v, :conf,
                        :ep, :tp, :sl, :r,
                        :sj, :mj, :tj
                    )
                """), params)
                logger.info(
                    f"  {ticker_name} (INSERT) → {report.get('verdict')} "
                    f"(conf={report.get('confidence')}, target={report.get('target_price')})"
                )
            session.commit()
            saved += 1
        logger.info(f"  Сохранено AI-отчётов: {saved}")
    finally:
        session.close()


def report_latest_candle_dates(engine, figi_list: list):
    """Залогируем самые свежие даты свечей по топ-тикерам — для контроля."""
    with engine.connect() as conn:
        for figi in figi_list:
            try:
                row = conn.execute(
                    text(
                        f'SELECT timestamp::date AS d, close '
                        f'FROM all_dfs."{figi}" ORDER BY timestamp DESC LIMIT 1'
                    )
                ).fetchone()
                if row:
                    logger.info(f"  {figi}: последняя свеча {row[0]} close={row[1]}")
                else:
                    logger.warning(f"  {figi}: свечей в БД нет!")
            except Exception as e:
                logger.warning(f"  {figi}: не удалось прочитать ({e})")


def resolve_figis(engine, tickers: List[str]) -> List[tuple]:
    """Найти FIGI для тикеров. Возвращает [(ticker, figi), ...]."""
    found = []
    with engine.connect() as conn:
        for t in tickers:
            row = conn.execute(
                text("SELECT figi FROM public.tickers WHERE ticker = :t LIMIT 1"),
                {"t": t},
            ).fetchone()
            if row:
                found.append((t, row[0]))
                logger.info(f"  {t} → {row[0]}")
            else:
                logger.warning(f"  {t}: FIGI не найден в public.tickers")
    return found


def run_training(figi_list: List[str], walk_forward: bool = False,
                 include_dl: bool = False) -> bool:
    """
    Запустить train_models.py для указанных FIGI.
    Вывод тренировки стримим построчно — чтобы было видно прогресс.
    Если include_dl=False — обучает только модели НЕ из EXCLUDED_MODELS.
    """
    cmd = [sys.executable, "-u", "scripts/train_models.py"]
    for figi in figi_list:
        cmd += ["--ticker", figi]

    if not include_dl and EXCLUDED_MODELS:
        # train_models.py принимает --model несколько раз: передаём whitelist
        active_models = [m for m in ALL_MODELS if m not in EXCLUDED_MODELS]
        logger.info(f"Исключены модели: {EXCLUDED_MODELS}")
        logger.info(f"Будут обучены: {active_models}")
        for m in active_models:
            cmd += ["--model", m]
    else:
        logger.info("Обучаем все модели (включая DL)")

    if walk_forward:
        cmd.append("--walk-forward")

    logger.info(f"Команда: {' '.join(cmd)}")
    start = datetime.utcnow()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Стримим stdout построчно в наш логгер
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    logger.info(f"[train] {line}")
                # таймаут общий по wall-clock
                if (datetime.utcnow() - start).total_seconds() > TRAINING_TIMEOUT:
                    proc.kill()
                    logger.error(f"✗ Обучение превысило таймаут {TRAINING_TIMEOUT}s")
                    return False
        finally:
            proc.stdout.close()

        rc = proc.wait()
        if rc == 0:
            elapsed = (datetime.utcnow() - start).total_seconds()
            logger.info(f"✓ Обучение завершено за {elapsed:.0f}s")
            return True
        logger.error(f"✗ Обучение завершилось с кодом {rc}")
        return False
    except Exception as e:
        logger.error(f"✗ Ошибка запуска обучения: {e}")
        return False


def process_llm_for_new_results(engine, figi_list: List[str]) -> dict:
    """
    Пройти по всем строкам model_results за последние N часов,
    у которых llm_signal IS NULL, и запросить LLM по каждой.
    Возвращает статистику.
    """
    since = datetime.utcnow() - timedelta(hours=LLM_LOOKBACK_HOURS)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Подгружаем live hit rates по тикерам — реальная свежая точность моделей
    live_perf_cache = {}
    for figi in figi_list:
        try:
            live_perf_cache[figi] = compute_recent_hit_rates(engine, figi)
        except Exception:
            live_perf_cache[figi] = {}

    try:
        rows = session.execute(
            text("""
                SELECT id, db_name, ticker_name, model_name,
                       current_price, trading_signal,
                       test_r2, test_direction_accuracy,
                       win_rate, profit_factor, cumulative_return, test_samples
                FROM public.model_results
                WHERE llm_signal IS NULL
                  AND timestamp >= :since
                  AND db_name = ANY(:figis)
                  AND trading_signal IS NOT NULL
                  AND current_price IS NOT NULL
                ORDER BY db_name, timestamp DESC
            """),
            {"since": since, "figis": figi_list},
        ).fetchall()

        logger.info(f"Найдено {len(rows)} записей без LLM-валидации")

        stats = {"agree": 0, "disagree": 0, "unavailable": 0, "errors": 0}

        for row in rows:
            row_id = row[0]
            db_name = row[1]
            ticker = row[2] or db_name
            model_name = row[3]

            # Live точность этой модели по данному тикеру
            live = live_perf_cache.get(db_name, {}).get(model_name, {})
            live30 = live.get(30, {})

            try:
                # Per-row LLM анализ: ТОЛЬКО метрики обучения этой модели
                # + live точность. Без TA / новостей / рынка.
                result = analyze_signal(
                    ticker=ticker,
                    current_price=float(row[4] or 0),
                    trading_signal=row[5] or "HOLD",
                    models_data=[{
                        "model_name": model_name,
                        "signal": row[5],
                        "r2": float(row[6]) if row[6] is not None else 0,
                        "direction_accuracy": float(row[7]) if row[7] is not None else 50,
                        "win_rate": float(row[8]) if row[8] is not None else None,
                        "profit_factor": float(row[9]) if row[9] is not None else None,
                        "cumulative_return": float(row[10]) if row[10] is not None else None,
                        "test_samples": int(row[11]) if row[11] is not None else None,
                        # LIVE (свежая реальная точность)
                        "recent_hit_rate_30d": live30.get("hit_rate"),
                        "recent_samples_30d": live30.get("total", 0),
                    }],
                    r2_avg=float(row[6]) if row[6] is not None else 0,
                    direction_accuracy_avg=float(row[7]) if row[7] is not None else 50,
                )
                answer = result["answer"]
                reasoning = result.get("reasoning", "") or result.get("explanation", "")
                stats[answer.lower()] = stats.get(answer.lower(), 0) + 1

                session.execute(
                    text("""
                        UPDATE public.model_results
                        SET llm_signal = :llm_signal,
                            llm_reasoning = :llm_reasoning,
                            llm_processed_at = :now
                        WHERE id = :id
                    """),
                    {
                        "llm_signal": answer,
                        "llm_reasoning": reasoning,
                        "now": datetime.utcnow(),
                        "id": row_id,
                    },
                )
                session.commit()
                short = (reasoning[:80] + "...") if len(reasoning) > 80 else reasoning
                logger.info(f"  #{row_id} {ticker}/{row[3]} {row[5]} → {answer} | {short}")
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"  #{row_id} LLM ошибка: {e}")

        return stats
    finally:
        session.close()


def run_nightly_pipeline(skip_training: bool = False, walk_forward: bool = False,
                         include_dl: bool = False, skip_llm: bool = False,
                         skip_fetch: bool = False):
    start = datetime.now()
    logger.info("=" * 60)
    logger.info(f"Ночной пайплайн стартовал в {start}")
    logger.info(f"Тикеры: {TOP_TICKERS}")
    logger.info("=" * 60)

    engine = get_engine()

    logger.info("\n[1/4] Поиск FIGI для топ-тикеров...")
    pairs = resolve_figis(engine, TOP_TICKERS)
    if not pairs:
        logger.error("Ни один тикер не найден в public.tickers. Прерываю.")
        return
    figi_list = [figi for _, figi in pairs]

    if not skip_fetch:
        logger.info(f"\n[2/4] Подтягиваем свежие свечи (последние {FETCH_DAYS} дней)...")
        logger.info("Свечи ДО загрузки:")
        report_latest_candle_dates(engine, figi_list)
        ok = fetch_fresh_candles()
        if not ok:
            logger.warning("Загрузка свечей не завершилась успехом — продолжаю на старых данных")
        logger.info("Свечи ПОСЛЕ загрузки:")
        report_latest_candle_dates(engine, figi_list)
    else:
        logger.info("\n[2/4] Загрузка свечей пропущена (--skip-fetch)")

    if not skip_training:
        logger.info("\n[3/4] Обучение моделей...")
        ok = run_training(figi_list, walk_forward=walk_forward, include_dl=include_dl)
        if not ok:
            logger.warning("Обучение завершилось с ошибками — всё равно запускаю LLM-обработку.")
    else:
        logger.info("\n[3/4] Обучение пропущено (--skip-training)")

    if skip_llm:
        logger.info("\n[4/5] LLM-валидация пропущена (--skip-llm)")
        stats = "skipped"
    else:
        logger.info("\n[4/5] LLM-валидация per-row...")
        stats = process_llm_for_new_results(engine, figi_list)

    if skip_llm:
        logger.info("\n[5/5] Консолидированные AI-отчёты пропущены (--skip-llm)")
    else:
        logger.info("\n[5/5] Резолв старых AI-отчётов + генерация новых...")
        resolve_ticker_reports(engine)
        generate_and_save_ticker_reports(engine, pairs)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=" * 60)
    logger.info(f"Готово за {elapsed:.0f}s")
    logger.info(f"LLM-статистика: {stats}")
    logger.info("=" * 60)

    engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Ночной пайплайн: train + LLM")
    parser.add_argument("--skip-training", action="store_true",
                        help="Пропустить обучение, только LLM по существующим записям")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Использовать walk-forward валидацию")
    parser.add_argument("--include-dl", action="store_true",
                        help="Включить DL модели (LSTM/TCN). По умолчанию выключены.")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Пропустить LLM-валидацию (только обучение)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Пропустить загрузку свежих свечей из Tinkoff API")
    args = parser.parse_args()
    run_nightly_pipeline(
        skip_training=args.skip_training,
        walk_forward=args.walk_forward,
        include_dl=args.include_dl,
        skip_llm=args.skip_llm,
        skip_fetch=args.skip_fetch,
    )


if __name__ == "__main__":
    main()
