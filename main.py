import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from utils.calculate_weight import calculate_model_score
from pydantic_models.model_result import ModelResult

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
    title="Financial Prediction Models API",
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
    """Главная страница со списком доступных моделей"""
    try:
        # Создаем сессию
        Session = sessionmaker(bind=engine)
        session = Session()

        # Запрашиваем все уникальные модели из базы данных
        models = session.query(ModelResult.model_name).distinct().all()
        logger.info(f"Найдено {len(models)} уникальных моделей в базе данных")

        models_info = []

        for model in models:
            model_name = model[0]  # Извлекаем название модели из кортежа

            # Если это не сводка, а модель
            if model_name != "all_models":
                # Находим самую последнюю дату для этой модели
                latest_timestamp = session.query(func.max(ModelResult.timestamp)).filter(
                    ModelResult.model_name == model_name
                ).scalar()

                if latest_timestamp:
                    # Получаем только записи с последней временной меткой
                    latest_results = session.query(ModelResult).filter(
                        ModelResult.model_name == model_name,
                        ModelResult.timestamp == latest_timestamp
                    ).all()

                    # Считаем количество записей для последней даты
                    total_files = len(latest_results)

                    # Анализируем сигналы в результатах модели
                    signals_count = {"BUY": 0, "SELL": 0, "HOLD": 0, "NEUTRAL": 0}

                    for result in latest_results:
                        content = result.text

                        # Определяем сигнал из содержимого
                        if "Торговый сигнал: BUY" in content:
                            signals_count["BUY"] += 1
                        elif "Торговый сигнал: SELL" in content:
                            signals_count["SELL"] += 1
                        elif "Торговый сигнал: HOLD" in content:
                            signals_count["HOLD"] += 1
                        elif "Торговый сигнал: NEUTRAL" in content:
                            signals_count["NEUTRAL"] += 1

                    # Собираем информацию о модели
                    models_info.append({
                        "name": model_name,
                        "total_files": total_files,
                        "signals": signals_count,
                        "latest_date": latest_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    })

        # Сортируем модели по количеству записей (по убыванию)
        models_info.sort(key=lambda x: x["total_files"], reverse=True)

        # Закрываем сессию
        session.close()

    except Exception as e:
        logger.error(f"Ошибка при получении данных из базы данных: {e}")
        models_info = []

    return templates.TemplateResponse(
        "models_list.html",
        {"request": request, "models_info": models_info}
    )


@app.get("/model/{model_name}", response_class=HTMLResponse, tags=["Информация о обученных моделях"],
         description="Результаты обучения конкретной модели")
async def view_model(request: Request,
                     model_name: str,
                     signal: Optional[str] = None,
                     sort: str = "accuracy",
                     order: str = "desc"):
    """Страница с результатами конкретной модели"""
    logger.info(f"Запрошена модель: {model_name}")

    try:
        # Создаем сессию SQLAlchemy
        Session = sessionmaker(bind=engine)
        session = Session()

        # Запрашиваем данные для выбранной модели из базы данных
        query = session.query(ModelResult).filter(ModelResult.model_name == model_name)
        model_results = query.all()

        logger.info(f"Найдено {len(model_results)} записей для модели {model_name}")

        # Если данных нет, возвращаемся на главную
        if not model_results:
            session.close()
            return RedirectResponse(url="/")

        files_data = []

        for result in model_results:
            try:
                content = result.text

                # Определяем сигнал из содержимого
                file_signal = None
                if "Торговый сигнал: BUY" in content:
                    file_signal = "BUY"
                elif "Торговый сигнал: SELL" in content:
                    file_signal = "SELL"
                elif "Торговый сигнал: HOLD" in content:
                    file_signal = "HOLD"
                elif "Торговый сигнал: NEUTRAL" in content:
                    file_signal = "NEUTRAL"

                # Если указан фильтр по сигналу и он не совпадает, пропускаем запись
                if signal and signal.upper() != file_signal:
                    continue

                # Извлекаем Direction Accuracy
                accuracy = None
                accuracy_match = re.search(r'Direction Accuracy: (\d+\.\d+)', content)
                if accuracy_match:
                    accuracy = float(accuracy_match.group(1))

                # Извлекаем R² (коэффициент детерминации)
                r_squared = None
                r2_match = re.search(r'R²: (\d+\.\d+)', content)
                if r2_match:
                    r_squared = float(r2_match.group(1))

                # Извлекаем MAPE
                mape = None
                mape_match = re.search(r'MAPE: (\d+\.\d+)', content)
                if mape_match:
                    mape = float(mape_match.group(1))

                # Получаем тикер из имени базы данных
                ticker = result.db_name if result.db_name else "Unknown"

                # Форматируем дату из timestamp
                date = result.timestamp.strftime("%Y%m%d") if result.timestamp else "Unknown"

                # Извлекаем ожидаемое изменение цены
                expected_change = None
                change_match = re.search(r'Ожидаемое изменение: ([+-]?\d+\.\d+)%', content)
                if change_match:
                    expected_change = float(change_match.group(1))

                files_data.append({
                    'name': result.db_name,  # Используем имя db файла
                    'content': content,
                    'signal': file_signal,
                    'accuracy': accuracy,
                    'r_squared': r_squared,
                    'mape': mape,
                    'ticker': ticker,
                    'algorithm': model_name,
                    'date': date,
                    'expected_change': expected_change
                })
            except Exception as e:
                logger.error(f"Ошибка при обработке записи {result.id} для {result.db_name}: {e}")
                files_data.append({
                    'name': result.db_name,
                    'content': f"Ошибка обработки данных: {str(e)}",
                    'signal': None,
                    'accuracy': None,
                    'ticker': "Error",
                    'algorithm': model_name,
                    'date': "Error"
                })

        # Закрываем сессию
        session.close()

        # Получаем статистику по сигналам для отображения в фильтрах
        signal_stats = {
            "BUY": sum(1 for file in files_data if file['signal'] == "BUY"),
            "SELL": sum(1 for file in files_data if file['signal'] == "SELL"),
            "HOLD": sum(1 for file in files_data if file['signal'] == "HOLD"),
            "NEUTRAL": sum(1 for file in files_data if file['signal'] == "NEUTRAL")
        }

        # Сортировка результатов
        if sort == "accuracy" and order == "desc":
            files_data.sort(key=lambda x: x.get('accuracy', 0) or 0, reverse=True)
        elif sort == "accuracy" and order == "asc":
            files_data.sort(key=lambda x: x.get('accuracy', 0) or 0)
        elif sort == "r_squared" and order == "desc":
            files_data.sort(key=lambda x: x.get('r_squared', 0) or 0, reverse=True)
        elif sort == "r_squared" and order == "asc":
            files_data.sort(key=lambda x: x.get('r_squared', 0) or 0)
        elif sort == "mape" and order == "desc":
            files_data.sort(key=lambda x: x.get('mape', 0) or 0, reverse=True)
        elif sort == "mape" and order == "asc":
            files_data.sort(key=lambda x: x.get('mape', 0) or 0)
        elif sort == "expected_change" and order == "desc":
            files_data.sort(key=lambda x: x.get('expected_change', 0) or 0, reverse=True)
        elif sort == "expected_change" and order == "asc":
            files_data.sort(key=lambda x: x.get('expected_change', 0) or 0)

        return templates.TemplateResponse(
            "model_detail.html",
            {
                "request": request,
                "files_data": files_data,
                "model_name": model_name,
                "current_signal": signal,
                "current_sort": sort,
                "current_order": order,
                "signal_stats": signal_stats
            }
        )

    except Exception as e:
        logger.error(f"Ошибка при получении данных из базы данных: {e}")
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": f"Произошла ошибка при загрузке данных модели: {str(e)}"
            }
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