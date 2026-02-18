# Financial Prediction Models

Система прогнозирования цен финансовых инструментов с использованием ML-моделей.

## Быстрый старт

```bash
# 1. Настройка окружения
cp .env.example .env
nano .env  # Укажите параметры БД и INVEST_TOKEN

# 2. Инициализация (установка зависимостей + создание таблиц)
make init

# 3. Сбор данных с Tinkoff API (обязательно перед обучением!)
make fetch-data

# 4. Обучение моделей
make train

# 5. Запуск сервера
make server

# 6. Открыть Swagger UI
open http://localhost:8000/docs
```

## Команды Make

```bash
make help           # Показать все команды

# Setup
make init           # Полная инициализация
make install        # Установить зависимости
make db-setup       # Создать таблицы в БД

# Data Collection
make fetch-data     # Собрать все данные (тикеры + свечи)
make fetch-tickers  # Получить список тикеров
make fetch-candles  # Получить исторические свечи
make data-status    # Проверить статус данных

# Server
make server         # Запустить сервер (dev)
make server-prod    # Запустить сервер (prod)
make health         # Проверить статус

# Training
make train                    # Обучить все модели
make train-model MODEL=ridge  # Обучить конкретную модель
make list-models              # Список моделей

# Docker
make docker-run     # Запустить в Docker
make docker-stop    # Остановить
make logs           # Логи
```

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
│  scripts/train_models.py → models/*.py → PostgreSQL         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      PostgreSQL                              │
│  ┌─────────────┐    ┌─────────────────────────────────┐     │
│  │  all_dfs.*  │    │     public.model_results        │     │
│  │  (данные)   │    │  (результаты обучения)          │     │
│  └─────────────┘    └─────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                           │
│  /docs (Swagger) │ /health │ /model/* │ /get_best_ten       │
└─────────────────────────────────────────────────────────────┘
```

## Установка

### 1. Зависимости

```bash
pip install -r requirements.txt
```

### 2. Конфигурация

Создайте `.env` файл:

```env
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password

# Tinkoff API (для сбора данных)
INVEST_TOKEN=your_token
```

### 3. База данных

```bash
# Создать таблицы
make db-setup

# Или вручную
python scripts/setup_db.py
```

Скрипт создаёт:
- Схему `public` с таблицей `model_results`
- Схему `all_dfs` для данных тикеров
- Индексы для быстрого поиска

## Сбор данных

Перед обучением моделей необходимо собрать данные с Tinkoff Invest API.

### Получение токена

1. Зайдите в [Тинькофф Инвестиции](https://www.tinkoff.ru/invest/)
2. Откройте настройки → "Токен для API"
3. Создайте токен (достаточно read-only доступа)
4. Добавьте токен в `.env`:
   ```
   INVEST_TOKEN=your_token_here
   ```

### Команды сбора данных

```bash
# Полный сбор данных (рекомендуется)
make fetch-data

# Или поэтапно:
make fetch-tickers  # Шаг 1: Получить список инструментов
make fetch-candles  # Шаг 2: Получить исторические свечи

# Проверить статус данных
make data-status
```

### Что собирается

| Скрипт | Что делает | Куда сохраняет |
|--------|------------|----------------|
| `all_figi_to_db.py` | Список всех инструментов (акции, облигации, ETF, валюты, фьючерсы) | `public.tickers` |
| `all_dfs_to_db.py` | Дневные свечи за 1000 дней для акций и фьючерсов | `all_dfs.{FIGI}` |

### Время выполнения

- Сбор тикеров: ~1 минута
- Сбор свечей: зависит от количества тикеров (может занять несколько часов для всех инструментов)

### Периодическое обновление (cron)

```bash
# Обновление данных каждый день в 5:00
0 5 * * * cd /path/to/arima && make fetch-candles >> /var/log/data_collection.log 2>&1
```

## Обучение моделей

### Доступные модели

| Модель | Описание |
|--------|----------|
| ridge | Ridge Regression |
| xgboost | XGBoost Gradient Boosting |
| lightgbm | LightGBM (Microsoft) |
| catboost | CatBoost (Yandex) |
| random_forest | Random Forest |
| arima | ARIMA Time Series |

### Запуск обучения

```bash
# Все модели на всех тикерах
python scripts/train_models.py

# Конкретная модель
python scripts/train_models.py --model ridge

# Конкретный тикер
python scripts/train_models.py --ticker BBG000Q7ZZY2

# Комбинация
python scripts/train_models.py --model ridge --model xgboost --ticker BBG000Q7ZZY2

# Тестовый запуск (без сохранения в БД)
python scripts/train_models.py --dry-run
```

### Периодическое обучение (cron)

```bash
# Каждый день в 6:00
0 6 * * * cd /path/to/arima && python scripts/train_models.py >> /var/log/training.log 2>&1
```

## API Server

### Запуск

```bash
# Development (с auto-reload)
make server
# или
uvicorn main:app --reload --port 8000

# Production
make server-prod
# или
uvicorn main:app --workers 4 --port 8000
```

### Endpoints

| URL | Метод | Описание |
|-----|-------|----------|
| `/docs` | GET | Swagger UI |
| `/health` | GET | Health check |
| `/health/ready` | GET | Health + DB check |
| `/` | GET | Главная (HTML) |
| `/model/{name}` | GET | Результаты модели |
| `/get_all_results` | GET | Все результаты (JSON) |
| `/get_best_ten` | GET | Топ-10 моделей |

### Фильтрация

```bash
# По сигналу
curl "http://localhost:8000/model/ridge?signal=BUY"

# С сортировкой
curl "http://localhost:8000/model/ridge?sort=accuracy&order=desc"
```

## Мониторинг

### Health Checks

```bash
# Базовый
curl http://localhost:8000/health
# {"status": "healthy", "version": "2.0.0"}

# С проверкой БД
curl http://localhost:8000/health/ready
# {"status": "ready", "database": "connected"}
```

### Логи

```bash
# Docker
make logs

# Systemd (если настроен как сервис)
journalctl -u arima-api -f
```

### Статистика

```bash
# Количество результатов в БД
curl -s http://localhost:8000/get_all_results | jq '.total_records'

# Топ модели
curl -s http://localhost:8000/get_best_ten | jq '.data[].model_name'
```

## Docker

### Запуск

```bash
# Сборка и запуск
docker-compose up -d

# Логи
docker-compose logs -f app

# Остановка
docker-compose down
```

### docker-compose.yml

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=${DB_HOST}
      - DB_PORT=${DB_PORT}
      - DB_PASSWORD=${DB_PASSWORD}
    restart: unless-stopped
```

## Структура проекта

```
arima/
├── main.py                 # FastAPI приложение
├── Makefile                # Команды make
├── requirements.txt        # Зависимости
├── .env.example            # Шаблон конфигурации
│
├── scripts/
│   ├── setup_db.py         # Создание таблиц БД
│   ├── train_models.py     # Обучение моделей
│   └── collect_data.py     # Сбор данных
│
├── all_figi_to_db.py       # Загрузка тикеров из Tinkoff API
├── all_dfs_to_db.py        # Загрузка свечей из Tinkoff API
│
├── core/                   # Базовые модули
│   ├── feature_engineering.py   # Создание фичей
│   ├── data_pipeline.py         # Подготовка данных
│   ├── base_model.py            # Базовый класс
│   └── metrics.py               # Метрики
│
├── models/                 # ML модели
│   ├── ridge.py
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   └── ...
│
├── utils/                  # Утилиты
│   ├── load_data_method.py
│   └── calculate_weight.py
│
├── templates/              # HTML шаблоны
└── static/                 # CSS, JS
```

## База данных

### Схема `model_results`

```sql
CREATE TABLE public.model_results (
    id SERIAL PRIMARY KEY,
    db_name VARCHAR(255),        -- Тикер (FIGI)
    model_name VARCHAR(255),     -- Название модели
    timestamp TIMESTAMP,         -- Время обучения

    -- Метрики
    test_mse FLOAT,
    test_rmse FLOAT,
    test_mae FLOAT,
    test_r2 FLOAT,
    test_mape FLOAT,
    test_direction_accuracy FLOAT,

    -- Прогноз
    current_price FLOAT,
    predicted_price FLOAT,
    expected_change FLOAT,       -- %
    trading_signal VARCHAR(10)   -- BUY/SELL/HOLD/NEUTRAL
);
```

### Полезные запросы

```sql
-- Последние результаты по модели
SELECT * FROM model_results
WHERE model_name = 'ridge'
ORDER BY timestamp DESC
LIMIT 10;

-- Статистика по моделям
SELECT model_name, COUNT(*), AVG(test_r2)
FROM model_results
GROUP BY model_name;

-- Лучшие сигналы BUY
SELECT db_name, model_name, expected_change, test_direction_accuracy
FROM model_results
WHERE trading_signal = 'BUY'
ORDER BY expected_change DESC
LIMIT 10;
```

## Troubleshooting

### База данных не подключается

```bash
# Проверить подключение
make db-check

# Проверить переменные
env | grep DB_
```

### Нет данных для обучения

```bash
# Проверить статус данных
make data-status

# Если данных нет - собрать их
make fetch-data

# Проверить, что INVEST_TOKEN установлен
echo $INVEST_TOKEN
```

### Ошибка при сборе данных

```bash
# Проверить токен Tinkoff API
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Token:', 'OK' if os.getenv('INVEST_TOKEN') else 'NOT SET')
"

# Запустить с детальными логами
python all_figi_to_db.py
```

### Модель не обучается

```bash
# Проверить данные
python -c "
from utils.load_data_method import load_data
df = load_data('BBG000Q7ZZY2')
print(len(df))
"

# Запустить с отладкой
python scripts/train_models.py --model ridge --dry-run
```

### Сервер не запускается

```bash
# Проверить порт
lsof -i :8000

# Проверить синтаксис
python -m py_compile main.py
```

## Лицензия

MIT
