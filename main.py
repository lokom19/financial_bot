import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, List

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session

from utils.calculate_weight import calculate_model_score
from pydantic_models.model_result import ModelResult
from auth.models import User, Base as AuthBase
from auth.security import decode_token
from auth.security_middleware import limiter
from auth.router import router as auth_router, COOKIE_NAME

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from services.llm_service import (
    analyze_signal,
    generate_ticker_report,
    explain_ta_indicators,
    summarize_ta_indicators,
)
from services.ta_indicators import compute_ta_indicators
from services.news_service import fetch_news_for_ticker
from services.performance_tracker import compute_recent_hit_rates

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DB_HOST = os.getenv("DB_HOST") # "localhost"  # Change this to your database host
DB_PORT = os.getenv("DB_PORT")  # "5432"  # Default PostgreSQL port
DB_NAME = os.getenv("DB_NAME")    # "postgres"  # Change to your database name
DB_USER = os.getenv("DB_USER")      # "postgres"  # Change to your username
DB_PASSWORD = os.getenv("DB_PASSWORD")    # "mysecretpassword"  # Change to your password
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

# Sync session factory (for auth)
SyncSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db_session() -> Session:
    return SyncSessionLocal()


def get_current_user_from_request(request: Request) -> Optional[User]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    username = payload.get("sub")
    if not username:
        return None
    db = SyncSessionLocal()
    try:
        return db.query(User).filter(User.username == username, User.is_active == True).first()
    finally:
        db.close()


def _require_login(request: Request):
    """
    Возвращает (user, None) если пользователь залогинен,
    или (None, redirect_response) если нет.

    Использование:
        user, redirect = _require_login(request)
        if redirect:
            return redirect
    """
    user = get_current_user_from_request(request)
    if user:
        return user, None
    # Для API эндпоинтов отдаём 401, для страниц — редирект на логин
    if request.url.path.startswith("/api/"):
        return None, JSONResponse(
            status_code=401,
            content={"error": "Требуется авторизация"},
        )
    return None, RedirectResponse(url="/auth/login", status_code=302)

ASYNC_DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Создаем асинхронный движок
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,  # Установите True для отладки SQL запросов
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)


# Создаем асинхронную сессию
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

app = FastAPI(
    title="Trading Signals API",
    description="""
API для доступа к результатам обученных моделей прогнозирования финансовых инструментов.

## Возможности

* Просмотр результатов всех моделей
* Фильтрация по торговым сигналам (BUY/SELL/HOLD/NEUTRAL)
* Получение лучших предсказаний
* Health check для мониторинга

## Модели

Доступные модели: Ridge, XGBoost, LightGBM, CatBoost, LSTM, Prophet, ARIMA и другие.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настраиваем директорию для шаблонов
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Настройка статических файлов (CSS, JS)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Директория с результатами
OUTPUT_DIR = BASE_DIR / "output"

# Streamlit dashboard URL (shown in top-bar nav)
STREAMLIT_URL = os.getenv("STREAMLIT_URL", "http://localhost:8501")

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include auth router
app.include_router(auth_router)


@app.on_event("startup")
async def startup_event():
    """Create users table on startup if it doesn't exist."""
    try:
        AuthBase.metadata.create_all(bind=engine, tables=[User.__table__])
        logger.info("Users table ready")
    except Exception as e:
        logger.warning(f"Could not create users table: {e}")


# ============================================================
# LLM Analysis Endpoint
# ============================================================

class LLMAnalyzeRequest(BaseModel):
    ticker: str
    current_price: float
    trading_signal: str
    r2_avg: float = 0.0
    direction_accuracy_avg: float = 50.0
    models_data: List[dict] = []


@app.post("/api/llm/analyze", tags=["LLM"])
@limiter.limit("30/hour")
async def llm_analyze(request: Request, payload: LLMAnalyzeRequest):
    """Query LLM to analyze a trading signal."""
    result = analyze_signal(
        ticker=payload.ticker,
        current_price=payload.current_price,
        trading_signal=payload.trading_signal,
        models_data=payload.models_data,
        r2_avg=payload.r2_avg,
        direction_accuracy_avg=payload.direction_accuracy_avg,
    )
    return result


# ============================================================
# Ticker-level: страница + развёрнутый отчёт от LLM
# ============================================================

def _load_ticker_overview(db: Session, ticker_or_figi: str) -> dict:
    """
    Загружает последние записи по каждой модели для одного тикера.
    Принимает либо ticker (SBER), либо figi (BBG004730N88).
    """
    from sqlalchemy import and_, or_

    # Найдём FIGI и название
    figi = None
    ticker_name = None
    row = db.execute(
        text("SELECT ticker, figi FROM public.tickers WHERE ticker = :v OR figi = :v"),
        {"v": ticker_or_figi},
    ).fetchone()
    if row:
        ticker_name, figi = row[0], row[1]
    else:
        figi = ticker_or_figi
        ticker_name = ticker_or_figi

    # Последняя запись по каждой модели для этого FIGI
    subq = (
        db.query(
            ModelResult.model_name,
            func.max(ModelResult.timestamp).label("max_ts"),
        )
        .filter(ModelResult.db_name == figi)
        .group_by(ModelResult.model_name)
        .subquery()
    )
    rows = (
        db.query(ModelResult)
        .join(
            subq,
            and_(
                ModelResult.model_name == subq.c.model_name,
                ModelResult.timestamp == subq.c.max_ts,
            ),
        )
        .filter(ModelResult.db_name == figi)
        .all()
    )

    # Recent hit rates по каждой модели (live точность за 5/30 дней) —
    # критически важная метрика для LLM. R² из training-сета мог устареть,
    # эти числа показывают реальное качество прогнозов модели сейчас.
    recent_perf = compute_recent_hit_rates(engine, figi) if figi else {}

    # Пороги для auto-disable:
    # Если за последние 30 дней с n>=10 прогнозов модель угадывала < 40% —
    # она хуже монетки и временно выключается из консенсуса/LLM-payload.
    # Можно настроить через env, дефолт консервативный.
    import os as _os
    DISABLE_MIN_SAMPLES = int(_os.getenv("MODEL_DISABLE_MIN_SAMPLES", "10"))
    DISABLE_HIT_RATE_THRESHOLD = float(_os.getenv("MODEL_DISABLE_HIT_RATE", "40"))

    models = []
    r2_vals, dir_vals = [], []
    current_price = None
    last_training_dt = None
    for r in rows:
        if r.current_price and current_price is None:
            current_price = float(r.current_price)
        if r.timestamp and (last_training_dt is None or r.timestamp > last_training_dt):
            last_training_dt = r.timestamp
        # Recent hit rate (live) — самая важная метрика "доверия"
        perf = recent_perf.get(r.model_name, {})
        recent5 = perf.get(5, {})
        recent30 = perf.get(30, {})

        # Auto-disable: модель плохо работает live → выключаем из консенсуса
        hit30 = recent30.get("hit_rate")
        n30 = recent30.get("total", 0)
        is_disabled = (
            hit30 is not None
            and n30 >= DISABLE_MIN_SAMPLES
            and hit30 < DISABLE_HIT_RATE_THRESHOLD
        )

        models.append({
            "model_name": r.model_name,
            "signal": r.trading_signal,
            "r2": float(r.test_r2) if r.test_r2 is not None else None,
            "direction_accuracy": float(r.test_direction_accuracy)
                if r.test_direction_accuracy is not None else None,
            "expected_change": float(r.expected_change)
                if r.expected_change is not None else None,
            "win_rate": float(r.win_rate) if r.win_rate is not None else None,
            "predicted_price": float(r.predicted_price)
                if r.predicted_price is not None else None,
            "trained_at": r.timestamp.isoformat() if r.timestamp else None,
            # Новые поля: live точность за последние 5/30 дней
            "recent_hit_rate_5d": recent5.get("hit_rate"),
            "recent_hit_rate_30d": recent30.get("hit_rate"),
            "recent_samples_5d": recent5.get("total", 0),
            "recent_samples_30d": recent30.get("total", 0),
            "is_disabled": is_disabled,
            # Период обучения
            "train_start": r.data_start_date.isoformat() if r.data_start_date else None,
            "train_end": r.data_end_date.isoformat() if r.data_end_date else None,
            "train_samples": r.train_samples,
            "test_samples": r.test_samples,
        })
        if r.test_r2 is not None:
            r2_vals.append(float(r.test_r2))
        if r.test_direction_accuracy is not None:
            dir_vals.append(float(r.test_direction_accuracy))

    # Подтянем TA по тикеру
    ta = compute_ta_indicators(engine, figi) if figi else {}
    if current_price is None and ta.get("last_price"):
        current_price = ta["last_price"]

    # Даты для отображения и для LLM
    from datetime import timedelta as _td
    data_date = ta.get("last_candle_date")  # дата последней свечи
    # Прогнозная дата — следующий торговый день (приближаем как +1, +3 для пятницы)
    prediction_date = None
    if data_date:
        try:
            from datetime import datetime as _dt
            d = _dt.fromisoformat(data_date).date()
            # пятница (4) → пн (+3), суббота (5) → пн (+2), иначе +1
            delta = 3 if d.weekday() == 4 else (2 if d.weekday() == 5 else 1)
            prediction_date = (d + _td(days=delta)).isoformat()
        except Exception:
            pass

    # Подсчёт сигналов — ТОЛЬКО среди активных моделей
    # (отключённые исключены, они не должны влиять на консенсус)
    signals = {"BUY": 0, "SELL": 0, "HOLD": 0, "NEUTRAL": 0}
    for m in models:
        if m.get("is_disabled"):
            continue
        s = (m["signal"] or "").upper()
        if s in signals:
            signals[s] += 1

    disabled_count = sum(1 for m in models if m.get("is_disabled"))
    active_count = len(models) - disabled_count

    return {
        "ticker": ticker_name,
        "figi": figi,
        "current_price": current_price,
        "models": models,
        "models_count": len(models),
        "active_count": active_count,
        "disabled_count": disabled_count,
        "signals": signals,
        "r2_avg": sum(r2_vals) / len(r2_vals) if r2_vals else None,
        "direction_avg": sum(dir_vals) / len(dir_vals) if dir_vals else None,
        "ta": ta,
        "data_date": data_date,
        "prediction_date": prediction_date,
        "last_training_at": last_training_dt.isoformat() if last_training_dt else None,
        # Текущие пороги торговых сигналов (в %)
        "signal_thresholds": {
            "buy": float(_os.getenv("SIGNAL_BUY_THRESHOLD", "0.2")),
            "sell": float(_os.getenv("SIGNAL_SELL_THRESHOLD", "-0.2")),
            "neutral": float(_os.getenv("SIGNAL_NEUTRAL_THRESHOLD", "0.05")),
        },
    }


@app.get("/ticker/{ticker}", response_class=HTMLResponse, tags=["Pages"])
async def ticker_page(request: Request, ticker: str):
    """Страница с полной сводкой по тикеру + кнопка LLM-отчёта. Требует авторизации."""
    current_user, redirect = _require_login(request)
    if redirect:
        return redirect
    db = SyncSessionLocal()
    try:
        overview = _load_ticker_overview(db, ticker)
    finally:
        db.close()

    return templates.TemplateResponse(
        request=request,
        name="ticker_report.html",
        context={
            "current_user": current_user,
            "streamlit_url": STREAMLIT_URL,
            "overview": overview,
        },
    )


@app.get("/api/ticker/{ticker}/predictions-history", tags=["Stats"])
async def predictions_history(ticker: str, days: int = 90):
    """
    Возвращает данные для таблицы "реальная цена vs прогнозы моделей":
      - actual_prices: реальная история close-цен за N дней
      - predictions: точечные прогнозы из model_results
        (с пометкой корректно/некорректно предсказано направление)
      - stats_per_model: процент корректно предсказанных направлений по моделям
    """
    db = SyncSessionLocal()
    try:
        # FIGI и ticker_name
        row = db.execute(
            text("SELECT ticker, figi FROM public.tickers WHERE ticker = :v OR figi = :v"),
            {"v": ticker},
        ).fetchone()
        if row:
            ticker_name, figi = row[0], row[1]
        else:
            return JSONResponse(status_code=404, content={"error": f"Тикер {ticker} не найден"})

        # Историческая цена + high/low за день
        actual_rows = db.execute(
            text(
                f'SELECT timestamp::date AS d, close, high, low '
                f'FROM all_dfs."{figi}" '
                f"ORDER BY timestamp DESC LIMIT :n"
            ),
            {"n": days},
        ).fetchall()
        actual_prices = [
            {
                "date": r[0].isoformat(),
                "close": float(r[1]),
                "high": float(r[2]) if r[2] is not None else None,
                "low": float(r[3]) if r[3] is not None else None,
            }
            for r in reversed(actual_rows)
        ]

        # Все предсказания за тот же период
        if actual_prices:
            since_date = actual_prices[0]["date"]
        else:
            since_date = "1970-01-01"

        # DISTINCT ON (date, model) — если за день было несколько прогонов
        # (например утренний + ночной), берём ПОСЛЕДНИЙ по timestamp.
        pred_rows = db.execute(
            text("""
                SELECT DISTINCT ON (timestamp::date, model_name)
                    timestamp, model_name, current_price, predicted_price,
                    trading_signal, expected_change, llm_signal
                FROM public.model_results
                WHERE db_name = :figi
                  AND timestamp::date >= :since
                  AND predicted_price IS NOT NULL
                  AND current_price IS NOT NULL
                ORDER BY timestamp::date, model_name, timestamp DESC
            """),
            {"figi": figi, "since": since_date},
        ).fetchall()
        # Сортировка по дате/времени для отображения
        pred_rows = sorted(pred_rows, key=lambda r: r[0])

        # Карты для проверки направления и для отображения high/low
        date_to_close = {p["date"]: p["close"] for p in actual_prices}
        date_to_high = {p["date"]: p["high"] for p in actual_prices}
        date_to_low = {p["date"]: p["low"] for p in actual_prices}
        dates_sorted = [p["date"] for p in actual_prices]

        def next_bar_after(d_str: str):
            """
            Возвращает (date, close, high, low) для ПЕРВОГО торгового дня
            строго после d_str. Корректно обрабатывает выходные/праздники.
            """
            for date_iso in dates_sorted:
                if date_iso > d_str:
                    return (
                        date_iso,
                        date_to_close[date_iso],
                        date_to_high.get(date_iso),
                        date_to_low.get(date_iso),
                    )
            return (None, None, None, None)

        predictions = []
        per_model_stats = {}
        for r in pred_rows:
            d_iso = r[0].date().isoformat()
            cur = float(r[2])
            pred = float(r[3])
            signal = r[4]
            model = r[1]

            # Проверка корректности предсказанного направления + high/low следующего дня
            next_date, actual_next, next_high, next_low = next_bar_after(d_iso)
            correct = None
            if actual_next is not None:
                pred_up = pred > cur
                actual_up = actual_next > cur
                correct = bool(pred_up == actual_up)

            predictions.append({
                "date": d_iso,
                "model": model,
                "current_price": cur,
                "predicted_price": pred,
                "signal": signal,
                "expected_change": float(r[5]) if r[5] is not None else None,
                "llm_signal": r[6],
                "correct_direction": correct,
                "actual_next_close": actual_next,
                "actual_next_high": next_high,
                "actual_next_low": next_low,
                "actual_next_date": next_date,
            })

            if correct is not None:
                s = per_model_stats.setdefault(model, {"total": 0, "hits": 0})
                s["total"] += 1
                if correct:
                    s["hits"] += 1

        stats_per_model = [
            {
                "model": m,
                "total": s["total"],
                "hits": s["hits"],
                "hit_rate": round(s["hits"] / s["total"] * 100, 1) if s["total"] else 0,
            }
            for m, s in sorted(per_model_stats.items())
        ]

        return {
            "ticker": ticker_name,
            "figi": figi,
            "actual_prices": actual_prices,
            "predictions": predictions,
            "stats_per_model": stats_per_model,
            "total_predictions": len(predictions),
        }
    finally:
        db.close()


@app.get("/api/ticker/{ticker}/ai-reports-history", tags=["Stats"])
async def ai_reports_history(ticker: str, days: int = 60):
    """
    История консолидированных AI-отчётов по тикеру:
    каждая запись = один LLM-вердикт за день + факт следующего дня.
    """
    db = SyncSessionLocal()
    try:
        row = db.execute(
            text("SELECT ticker, figi FROM public.tickers WHERE ticker = :v OR figi = :v"),
            {"v": ticker},
        ).fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"error": f"Тикер {ticker} не найден"})
        ticker_name, figi = row[0], row[1]

        # DISTINCT ON (prediction_date) — для каждой даты-прогноза показываем
        # ТОЛЬКО самый свежий вердикт.
        # Зачем: несколько ночных прогонов (например пт/сб/вс) могут целиться
        # в один и тот же понедельник — пользователю нужен последний по
        # timestamp вердикт, а не дублирование.
        reports = db.execute(text("""
            SELECT id, timestamp, data_date, prediction_date, current_price,
                   verdict, confidence, entry_price, target_price, stop_loss,
                   reasoning,
                   actual_close, actual_high, actual_low,
                   correct_direction, target_hit, stop_hit
            FROM (
                SELECT DISTINCT ON (prediction_date) *
                FROM public.ticker_reports
                WHERE figi = :figi
                  AND timestamp >= NOW() - (:days || ' days')::interval
                  AND prediction_date IS NOT NULL
                ORDER BY prediction_date DESC NULLS LAST, timestamp DESC
            ) latest
            ORDER BY prediction_date DESC NULLS LAST, timestamp DESC
        """), {"figi": figi, "days": days}).fetchall()

        items = []
        for r in reports:
            items.append({
                "id": r[0],
                "timestamp": r[1].isoformat() if r[1] else None,
                "data_date": r[2].isoformat() if r[2] else None,
                "prediction_date": r[3].isoformat() if r[3] else None,
                "current_price": float(r[4]) if r[4] is not None else None,
                "verdict": r[5],
                "confidence": r[6],
                "entry_price": float(r[7]) if r[7] is not None else None,
                "target_price": float(r[8]) if r[8] is not None else None,
                "stop_loss": float(r[9]) if r[9] is not None else None,
                "reasoning": r[10],
                "actual_close": float(r[11]) if r[11] is not None else None,
                "actual_high": float(r[12]) if r[12] is not None else None,
                "actual_low": float(r[13]) if r[13] is not None else None,
                "correct_direction": r[14],
                "target_hit": r[15],
                "stop_hit": r[16],
            })

        # Статистика по корректности
        resolved = [it for it in items if it["correct_direction"] is not None]
        correct = [it for it in resolved if it["correct_direction"]]
        target_hits = [it for it in resolved if it["target_hit"]]
        return {
            "ticker": ticker_name,
            "figi": figi,
            "items": items,
            "total": len(items),
            "resolved": len(resolved),
            "correct": len(correct),
            "accuracy": round(len(correct) / len(resolved) * 100, 1) if resolved else None,
            "target_hits": len(target_hits),
        }
    finally:
        db.close()


# Простой in-memory кеш для TA-объяснений и резюме (на день)
# Ключ: (ticker, data_date) → {explanation, generated_at}
_TA_EXPLAIN_CACHE: dict = {}
_TA_SUMMARY_CACHE: dict = {}


@app.get("/api/ticker/{ticker}/explain-ta", tags=["LLM"])
@limiter.limit("30/hour")
async def explain_ta(request: Request, ticker: str, force: bool = False):
    """
    AI-объяснение текущих значений технических индикаторов простым языком.
    Кешируется на день (по data_date). ?force=1 — перегенерировать.
    """
    db = SyncSessionLocal()
    try:
        overview = _load_ticker_overview(db, ticker)
    finally:
        db.close()

    if not overview["ta"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Нет данных технического анализа для этого тикера"},
        )

    cache_key = (overview["ticker"], overview.get("data_date"))
    if not force and cache_key in _TA_EXPLAIN_CACHE:
        cached = _TA_EXPLAIN_CACHE[cache_key]
        return {
            "ticker": overview["ticker"],
            "data_date": overview.get("data_date"),
            "explanation": cached["explanation"],
            "cached": True,
            "generated_at": cached["generated_at"],
        }

    result = explain_ta_indicators(overview["ticker"], overview["ta"])
    if result.get("error"):
        return JSONResponse(status_code=502, content={"error": result["error"]})

    from datetime import datetime as _dt
    generated_at = _dt.utcnow().isoformat()
    _TA_EXPLAIN_CACHE[cache_key] = {
        "explanation": result["explanation"],
        "generated_at": generated_at,
    }
    return {
        "ticker": overview["ticker"],
        "data_date": overview.get("data_date"),
        "explanation": result["explanation"],
        "cached": False,
        "generated_at": generated_at,
    }


@app.get("/api/ticker/{ticker}/news", tags=["News"])
@limiter.limit("60/hour")
async def ticker_news(request: Request, ticker: str, limit: int = 5):
    """Новостной фид по тикеру (smart-lab.ru, кеш 30 мин)."""
    try:
        items = fetch_news_for_ticker(ticker.upper(), max_items=limit)
        return {"ticker": ticker.upper(), "items": items, "count": len(items)}
    except Exception as e:
        logger.error(f"news endpoint error: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": "Не удалось получить новости", "items": []},
        )


@app.get("/api/ticker/{ticker}/ta-summary", tags=["LLM"])
@limiter.limit("30/hour")
async def ta_summary(request: Request, ticker: str, force: bool = False):
    """
    Краткий вывод по техническим индикаторам:
    БЫЧИЙ / МЕДВЕЖИЙ / СМЕШАННЫЙ / НЕЙТРАЛЬНЫЙ + одно предложение пояснения.
    Кеш на день.
    """
    db = SyncSessionLocal()
    try:
        overview = _load_ticker_overview(db, ticker)
    finally:
        db.close()

    if not overview["ta"]:
        return JSONResponse(status_code=400, content={"error": "Нет данных TA"})

    cache_key = (overview["ticker"], overview.get("data_date"))
    if not force and cache_key in _TA_SUMMARY_CACHE:
        return {**_TA_SUMMARY_CACHE[cache_key], "cached": True}

    result = summarize_ta_indicators(overview["ticker"], overview["ta"])
    if result.get("error"):
        return JSONResponse(status_code=502, content={"error": result["error"]})

    from datetime import datetime as _dt
    payload = {
        "ticker": overview["ticker"],
        "data_date": overview.get("data_date"),
        "verdict": result["verdict"],
        "strength": result["strength"],
        "summary": result["summary"],
        "key_indicators": result["key_indicators"],
        "generated_at": _dt.utcnow().isoformat(),
    }
    _TA_SUMMARY_CACHE[cache_key] = payload
    return {**payload, "cached": False}


class TickerReportRequest(BaseModel):
    ticker: str  # либо тикер, либо FIGI


@app.post("/api/llm/ticker-report", tags=["LLM"])
@limiter.limit("20/hour")
async def llm_ticker_report(request: Request, payload: TickerReportRequest):
    """
    Развёрнутый отчёт от LLM по тикеру:
    использует ВСЕ доступные модели + TA-индикаторы.
    В будущем подключится новостной фон.
    """
    db = SyncSessionLocal()
    try:
        overview = _load_ticker_overview(db, payload.ticker)
    finally:
        db.close()

    if not overview["models"]:
        return JSONResponse(
            status_code=404,
            content={"error": f"Нет данных моделей для тикера {payload.ticker}"},
        )

    # Подтягиваем новости (с кешем 30 минут)
    try:
        news_items = fetch_news_for_ticker(overview["ticker"], max_items=5)
    except Exception as e:
        logger.warning(f"news fetch failed: {e}")
        news_items = None

    # Не передаём отключённые модели — они исторически плохо угадывают,
    # их прогнозы могут только запутать LLM.
    active_models = [m for m in overview["models"] if not m.get("is_disabled")]

    report = generate_ticker_report(
        ticker=overview["ticker"],
        current_price=overview["current_price"] or 0.0,
        models_data=active_models,
        ta_indicators=overview["ta"],
        news_items=news_items,
        data_date=overview.get("data_date"),
        prediction_date=overview.get("prediction_date"),
    )
    return report


# ============================================================
# Health Check Endpoints
# ============================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Basic health check endpoint.

    Returns:
        JSON with status "healthy"
    """
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check including database connectivity.

    Returns:
        JSON with status and database connection info
    """
    try:
        # Test database connection
        Session = sessionmaker(bind=engine)
        session = Session()
        session.execute("SELECT 1")
        session.close()
        return {
            "status": "ready",
            "database": "connected",
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "not_ready",
            "database": str(e),
            "version": "2.0.0"
        }


# ============================================================
# Main Application Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["Pages"], description="Стартовая страница")
async def read_root(request: Request):
    """
    Главная: для НЕ залогиненных — лендинг с описанием сервиса.
    Для залогиненных — список моделей и виджет AI-отчётов.
    """
    current_user = get_current_user_from_request(request)
    if not current_user:
        return templates.TemplateResponse(
            request=request,
            name="landing.html",
            context={},
        )

    try:
        db = SyncSessionLocal()

        models = db.query(ModelResult.model_name).distinct().all()
        logger.info(f"Найдено {len(models)} уникальных моделей в базе данных")

        models_info = []
        from sqlalchemy import and_

        for model in models:
            model_name = model[0]

            if model_name != "all_models":
                # Берём ПОСЛЕДНЮЮ запись по КАЖДОМУ тикеру (db_name),
                # а не одну глобально-последнюю — тикеры обучаются
                # последовательно, у каждого свой timestamp.
                subq = (
                    db.query(
                        ModelResult.db_name,
                        func.max(ModelResult.timestamp).label("max_ts"),
                    )
                    .filter(ModelResult.model_name == model_name)
                    .group_by(ModelResult.db_name)
                    .subquery()
                )
                latest_results = (
                    db.query(ModelResult)
                    .join(
                        subq,
                        and_(
                            ModelResult.db_name == subq.c.db_name,
                            ModelResult.timestamp == subq.c.max_ts,
                        ),
                    )
                    .filter(ModelResult.model_name == model_name)
                    .all()
                )

                if latest_results:
                    latest_timestamp = max(r.timestamp for r in latest_results)
                    total_files = len(latest_results)
                    signals_count = {"BUY": 0, "SELL": 0, "HOLD": 0, "NEUTRAL": 0}

                    r2_vals, dir_vals = [], []
                    for result in latest_results:
                        # Use structured DB fields first, fall back to text parsing
                        sig = result.trading_signal
                        if not sig:
                            content = result.text or ""
                            if "Торговый сигнал: BUY" in content:
                                sig = "BUY"
                            elif "Торговый сигнал: SELL" in content:
                                sig = "SELL"
                            elif "Торговый сигнал: HOLD" in content:
                                sig = "HOLD"
                            elif "Торговый сигнал: NEUTRAL" in content:
                                sig = "NEUTRAL"

                        if sig in signals_count:
                            signals_count[sig] += 1

                        if result.test_r2 is not None:
                            r2_vals.append(result.test_r2)
                        if result.test_direction_accuracy is not None:
                            dir_vals.append(result.test_direction_accuracy)

                    avg_r2 = sum(r2_vals) / len(r2_vals) if r2_vals else None
                    avg_dir = sum(dir_vals) / len(dir_vals) if dir_vals else None

                    models_info.append({
                        "name": model_name,
                        "total_files": total_files,
                        "signals": signals_count,
                        "latest_date": latest_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "avg_r2": avg_r2,
                        "avg_direction": avg_dir,
                    })

        models_info.sort(key=lambda x: x["total_files"], reverse=True)
        db.close()

    except Exception as e:
        logger.error(f"Ошибка при получении данных из базы данных: {e}")
        models_info = []

    # Список уникальных тикеров с актуальными данными — для быстрого перехода
    # на страницу AI-отчёта
    try:
        db2 = SyncSessionLocal()
        ticker_rows = db2.execute(text("""
            SELECT DISTINCT
                COALESCE(t.ticker, mr.ticker_name, mr.db_name) AS ticker,
                mr.db_name AS figi
            FROM public.model_results mr
            LEFT JOIN public.tickers t ON mr.db_name = t.figi
            WHERE mr.trading_signal IS NOT NULL
            ORDER BY 1
        """)).fetchall()
        tickers_list = [{"ticker": r[0], "figi": r[1]} for r in ticker_rows]
        db2.close()
    except Exception as e:
        logger.warning(f"Не удалось получить список тикеров: {e}")
        tickers_list = []

    return templates.TemplateResponse(
        request=request,
        name="models_list.html",
        context={
            "models_info": models_info,
            "current_user": current_user,
            "streamlit_url": STREAMLIT_URL,
            "tickers_list": tickers_list,
        },
    )


@app.get("/model/{model_name}", response_class=HTMLResponse, tags=["Информация о обученных моделях"],
         description="Результаты обучения конкретной модели")
async def view_model(request: Request,
                     model_name: str,
                     signal: Optional[str] = None,
                     sort: str = "accuracy",
                     order: str = "desc"):
    """Страница с результатами конкретной модели — требует авторизации"""
    logger.info(f"Запрошена модель: {model_name}")
    current_user, redirect = _require_login(request)
    if redirect:
        return redirect

    try:
        db = SyncSessionLocal()

        # Берём только ПОСЛЕДНЮЮ запись по каждому тикеру (db_name)
        # для этой модели — чтобы не показывать дубли от walk-forward
        # фолдов и предыдущих запусков обучения.
        from sqlalchemy import and_
        subq = (
            db.query(
                ModelResult.db_name,
                func.max(ModelResult.timestamp).label("max_ts"),
            )
            .filter(ModelResult.model_name == model_name)
            .group_by(ModelResult.db_name)
            .subquery()
        )
        model_results = (
            db.query(ModelResult)
            .join(
                subq,
                and_(
                    ModelResult.db_name == subq.c.db_name,
                    ModelResult.timestamp == subq.c.max_ts,
                ),
            )
            .filter(ModelResult.model_name == model_name)
            .all()
        )
        logger.info(f"Найдено {len(model_results)} уникальных тикеров для модели {model_name}")

        if not model_results:
            db.close()
            return RedirectResponse(url="/")

        files_data = []

        for result in model_results:
            try:
                content = result.text or ""

                # Use structured DB fields (primary) with text fallback
                file_signal = result.trading_signal
                if not file_signal:
                    if "Торговый сигнал: BUY" in content:
                        file_signal = "BUY"
                    elif "Торговый сигнал: SELL" in content:
                        file_signal = "SELL"
                    elif "Торговый сигнал: HOLD" in content:
                        file_signal = "HOLD"
                    elif "Торговый сигнал: NEUTRAL" in content:
                        file_signal = "NEUTRAL"

                if signal and signal.upper() != file_signal:
                    continue

                # Structured metrics from DB
                accuracy = result.test_direction_accuracy
                r_squared = result.test_r2
                mape = result.test_mape
                expected_change = result.expected_change
                current_price = result.current_price
                win_rate = result.win_rate
                profit_factor = result.profit_factor
                cumulative_return = result.cumulative_return
                total_trades = result.total_trades

                # Extract sharpe_ratio from text (not in DB schema yet)
                sharpe_ratio = None
                sr_match = re.search(r'Sharpe Ratio: ([-+]?\d+\.\d+)', content)
                if sr_match:
                    sharpe_ratio = float(sr_match.group(1))

                max_drawdown = None
                md_match = re.search(r'Максимальная просадка: ([\d.]+)%', content)
                if md_match:
                    max_drawdown = float(md_match.group(1))

                ticker = result.ticker_name or result.db_name or "Unknown"
                date = result.timestamp.strftime("%Y-%m-%d") if result.timestamp else "Unknown"

                files_data.append({
                    'name': result.db_name,
                    'content': content,
                    'signal': file_signal,
                    'accuracy': accuracy,
                    'r_squared': r_squared,
                    'mape': mape,
                    'ticker': ticker,
                    'algorithm': model_name,
                    'date': date,
                    'expected_change': expected_change,
                    'current_price': current_price,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'cumulative_return': cumulative_return,
                    'total_trades': total_trades,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'llm_signal': result.llm_signal,
                    'llm_reasoning': result.llm_reasoning,
                    'llm_processed_at': result.llm_processed_at.strftime('%Y-%m-%d %H:%M') if result.llm_processed_at else None,
                })
            except Exception as e:
                logger.error(f"Ошибка при обработке записи {result.id}: {e}")

        db.close()

        signal_stats = {
            "BUY": sum(1 for f in files_data if f['signal'] == "BUY"),
            "SELL": sum(1 for f in files_data if f['signal'] == "SELL"),
            "HOLD": sum(1 for f in files_data if f['signal'] == "HOLD"),
            "NEUTRAL": sum(1 for f in files_data if f['signal'] == "NEUTRAL"),
        }

        sort_key_map = {
            "accuracy": "accuracy",
            "r_squared": "r_squared",
            "expected_change": "expected_change",
            "win_rate": "win_rate",
        }
        skey = sort_key_map.get(sort, "accuracy")
        files_data.sort(
            key=lambda x: x.get(skey) or 0,
            reverse=(order == "desc"),
        )

        return templates.TemplateResponse(
            request=request,
            name="model_detail.html",
            context={
                "files_data": files_data,
                "model_name": model_name,
                "current_signal": signal,
                "current_sort": sort,
                "current_order": order,
                "signal_stats": signal_stats,
                "current_user": current_user,
                "streamlit_url": STREAMLIT_URL,
            },
        )

    except Exception as e:
        logger.error(f"Ошибка при получении данных из базы данных: {e}")
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error_message": str(e)},
        )


@app.get("/get_all_results", tags=["Информация о обученных моделях"],
         description="Все результаты всех моделей")
async def get_latest_results():
    """
    Возвращает только самые последние результаты для каждой комбинации FIGI + модель
    """
    try:
        async with AsyncSessionLocal() as session:
            # Подзапрос для получения максимального timestamp для каждой комбинации db_name + model_name
            subquery = select(
                ModelResult.db_name,
                ModelResult.model_name,
                func.max(ModelResult.timestamp).label('max_timestamp')
            ).group_by(
                ModelResult.db_name,
                ModelResult.model_name
            ).subquery()

            # Основной запрос для получения записей с максимальным timestamp
            query = select(ModelResult).join(
                subquery,
                (ModelResult.db_name == subquery.c.db_name) &
                (ModelResult.model_name == subquery.c.model_name) &
                (ModelResult.timestamp == subquery.c.max_timestamp)
            )

            result = await session.execute(query)
            latest_results = result.scalars().all()

            logger.info(f"Получено {len(latest_results)} последних записей")

            # Группируем результаты
            grouped_results = defaultdict(dict)

            for model_result in latest_results:
                figi = model_result.db_name
                model_name = model_result.model_name

                model_data = {
                    "id": model_result.id,
                    "db_name": model_result.db_name,
                    "model_name": model_result.model_name,
                    "timestamp": model_result.timestamp.isoformat() if model_result.timestamp else None,
                    "text": model_result.text,
                    "test_mse": model_result.test_mse,
                    "test_rmse": model_result.test_rmse,
                    "test_mae": model_result.test_mae,
                    "test_r2": model_result.test_r2,
                    "test_mape": model_result.test_mape,
                    "test_direction_accuracy": model_result.test_direction_accuracy,
                    "train_direction_accuracy": model_result.train_direction_accuracy,
                    "current_price": model_result.current_price,
                    "predicted_price": model_result.predicted_price,
                    "expected_change": model_result.expected_change,
                    "trading_signal": model_result.trading_signal
                }

                grouped_results[figi][model_name] = model_data

            return {
                "status": "success",
                "data": dict(grouped_results),
                "total_figi": len(grouped_results),
                "total_records": len(latest_results)
            }

    except Exception as e:
        logger.error(f"Ошибка при получении последних данных: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении данных: {str(e)}"
        )


@app.get("/get_best_ten", tags=["Лучшие результаты обучения"])
async def get_best_ten():
    try:
        all_results_data = await get_latest_results()

        if all_results_data["status"] != "success":
            raise HTTPException(status_code=500, detail="Ошибка получения данных")

        models_list = []

        for figi, models in all_results_data["data"].items():
            for model_name, model_data in models.items():
                # Вычисляем оценку для каждой модели
                score = await calculate_model_score(model_data)
                model_data["score"] = score
                models_list.append(model_data)

        # Сортируем по оценке (по убыванию) и берем топ-10
        # Сортируем по оценке (по убыванию)
        sorted_models = sorted(
            models_list,
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        best_models = []
        unique_counter = {}
        for res in sorted_models:
            if len(best_models) == 10:
                break
            if res['db_name'] not in unique_counter.keys():
                unique_counter[res['db_name']] = 1
                best_models.append(res)
            else:
                continue

        return {
            "status": "success",
            "data": best_models,
            "total_best": len(best_models),
            "message": "Топ-10 лучших моделей по комплексной оценке"
        }

    except Exception as e:
        logger.error(f"Ошибка при получении лучших данных: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Вариант 1: Последние результаты по каждой модели для ТОП-4 инструментов
@app.get("/get_top_four", tags=["Результаты по ТОП-4"],
         description="Последние результаты для ТОП-4 инструментов по каждой модели")
async def get_best_four():
    """
    Возвращает последние результаты для каждой модели по ТОП-4 инструментам:
    - FUTBRM072500 (Brent Oil)
    - FUTSI0625000 (USD/RUB)
    - FUTRTS062500 (RTS Index)
    - FUTSBRF06250 (Sberbank)
    """
    try:
        # Список ТОП-4 инструментов
        top_four_instruments = [
            "FUTBRM072500",
            # "FUTSI0625000",
            # "FUTRTS062500",
            # "FUTSBRF06250",

            "FUTSI0925000",
            "FUTRTS092500",
            "FUTSBRF09250"


        ]

        async with AsyncSessionLocal() as session:
            # Подзапрос для получения максимального timestamp для каждой комбинации
            # db_name + model_name среди ТОП-4 инструментов
            subquery = select(
                ModelResult.db_name,
                ModelResult.model_name,
                func.max(ModelResult.timestamp).label('max_timestamp')
            ).where(
                ModelResult.db_name.in_(top_four_instruments)
            ).group_by(
                ModelResult.db_name,
                ModelResult.model_name
            ).subquery()

            # Основной запрос для получения записей с максимальным timestamp
            query = select(ModelResult).join(
                subquery,
                (ModelResult.db_name == subquery.c.db_name) &
                (ModelResult.model_name == subquery.c.model_name) &
                (ModelResult.timestamp == subquery.c.max_timestamp)
            ).where(
                ModelResult.db_name.in_(top_four_instruments)
            )

            result = await session.execute(query)
            latest_results = result.scalars().all()

            logger.info(f"Получено {len(latest_results)} записей для ТОП-4 инструментов")

            # Группируем результаты по инструментам
            grouped_results = defaultdict(dict)

            # Словарь для человекочитаемых названий инструментов
            instrument_names = {
                "FUTBRM072500": "Brent Oil",
                "FUTSI0625000": "USD/RUB",
                "FUTRTS062500": "RTS Index",
                "FUTSBRF06250": "Sberbank"
            }

            for model_result in latest_results:
                figi = model_result.db_name
                model_name = model_result.model_name

                model_data = {
                    "id": model_result.id,
                    "db_name": model_result.db_name,
                    "instrument_name": instrument_names.get(figi, figi),
                    "model_name": model_result.model_name,
                    "timestamp": model_result.timestamp.isoformat() if model_result.timestamp else None,
                    "text": model_result.text,
                    "test_mse": model_result.test_mse,
                    "test_rmse": model_result.test_rmse,
                    "test_mae": model_result.test_mae,
                    "test_r2": model_result.test_r2,
                    "test_mape": model_result.test_mape,
                    "test_direction_accuracy": model_result.test_direction_accuracy,
                    "train_direction_accuracy": model_result.train_direction_accuracy,
                    "current_price": model_result.current_price,
                    "predicted_price": model_result.predicted_price,
                    "expected_change": model_result.expected_change,
                    "trading_signal": model_result.trading_signal
                }

                grouped_results[figi][model_name] = model_data

            # Проверяем, что получили данные по всем ТОП-4 инструментам
            missing_instruments = set(top_four_instruments) - set(grouped_results.keys())
            if missing_instruments:
                logger.warning(f"Отсутствуют данные для инструментов: {missing_instruments}")

            return {
                "status": "success",
                "data": dict(grouped_results),
                "instruments": {
                    "total": len(grouped_results),
                    "expected": len(top_four_instruments),
                    "missing": list(missing_instruments) if missing_instruments else [],
                    "available": list(grouped_results.keys())
                },
                "total_records": len(latest_results),
                "instrument_mapping": instrument_names
            }

    except Exception as e:
        logger.error(f"Ошибка при получении данных ТОП-4: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении данных ТОП-4: {str(e)}"
        )


if __name__ == "__main__":
    # Проверяем наличие папки templates
    templates_dir = os.path.join(BASE_DIR, 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)

    # Проверяем наличие папки static
    static_dir = os.path.join(BASE_DIR, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Проверяем наличие папки output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logger.info("Запуск сервера...")
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)