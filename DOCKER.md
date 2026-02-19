# Docker Deployment

Руководство по запуску проекта в Docker контейнерах.

## Быстрый старт

```bash
# 1. Настроить переменные окружения
cp .env.example .env
nano .env  # Указать INVEST_TOKEN и пароль БД

# 2. Запустить все сервисы
docker-compose up -d

# 3. Проверить статус
docker-compose ps
```

После запуска доступны:
- **FastAPI**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **PostgreSQL**: localhost:5432

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  scheduler  │   │     app     │   │  dashboard  │       │
│  │             │   │   FastAPI   │   │  Streamlit  │       │
│  │ Data + Train│   │   :8000     │   │   :8501     │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                 │               │
│         └────────────────┼─────────────────┘               │
│                          │                                  │
│                   ┌──────▼──────┐                          │
│                   │  postgres   │                          │
│                   │    :5432    │                          │
│                   └─────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Сервисы

| Сервис | Порт | Описание |
|--------|------|----------|
| `postgres` | 5432 | PostgreSQL база данных |
| `app` | 8000 | FastAPI сервер (Swagger UI) |
| `scheduler` | — | Автоматический сбор данных + обучение |
| `dashboard` | 8501 | Streamlit дашборд с сигналами |

---

## Конфигурация

### Переменные окружения

Создайте файл `.env` в корне проекта:

```env
# PostgreSQL
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_NAME=postgres
DB_PORT=5432

# Tinkoff API (обязательно для сбора данных)
INVEST_TOKEN=your_tinkoff_api_token

# Scheduler
SCHEDULER_INTERVAL=3    # Интервал в минутах
SCHEDULER_DAYS=30       # Дней исторических данных
```

### Получение INVEST_TOKEN

1. Откройте [Тинькофф Инвестиции](https://www.tinkoff.ru/invest/)
2. Настройки → Токен для API
3. Создайте токен с read-only доступом
4. Скопируйте в `.env`

---

## Команды

### Запуск

```bash
# Все сервисы
docker-compose up -d

# Только определённые сервисы
docker-compose up -d postgres app
docker-compose up -d postgres scheduler
docker-compose up -d postgres dashboard

# С пересборкой образов
docker-compose up -d --build
```

### Остановка

```bash
# Остановить все
docker-compose down

# Остановить и удалить volumes (данные БД!)
docker-compose down -v
```

### Логи

```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker logs -f arima_scheduler
docker logs -f arima_app
docker logs -f arima_dashboard

# Последние 100 строк
docker logs --tail 100 arima_scheduler
```

### Статус

```bash
# Список контейнеров
docker-compose ps

# Использование ресурсов
docker stats
```

---

## Сценарии использования

### 1. Только сбор данных и обучение

```bash
# Запустить БД и scheduler
docker-compose up -d postgres scheduler

# Смотреть логи
docker logs -f arima_scheduler
```

### 2. Просмотр результатов

```bash
# Запустить БД и dashboard
docker-compose up -d postgres dashboard

# Открыть http://localhost:8501
```

### 3. Полный стек

```bash
# Всё вместе
docker-compose up -d

# FastAPI: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### 4. Разработка (с hot-reload)

```bash
# Запустить только БД в Docker
docker-compose up -d postgres

# Локально запустить приложения
make server      # FastAPI
make dashboard   # Streamlit
make scheduler   # Scheduler
```

---

## Настройка Scheduler

### Изменить интервал

```bash
# Через переменную окружения
SCHEDULER_INTERVAL=5 docker-compose up -d scheduler

# Или в .env
echo "SCHEDULER_INTERVAL=5" >> .env
docker-compose up -d scheduler
```

### Изменить период данных

```bash
# Собирать данные за 100 дней
SCHEDULER_DAYS=100 docker-compose up -d scheduler
```

### Однократный запуск pipeline

```bash
docker-compose run --rm scheduler python scripts/scheduler.py --once --days 30
```

---

## Управление данными

### Бэкап базы данных

```bash
# Создать бэкап
docker exec arima_postgres pg_dump -U postgres postgres > backup.sql

# Восстановить
cat backup.sql | docker exec -i arima_postgres psql -U postgres postgres
```

### Очистка данных

```bash
# Удалить volume с данными БД
docker-compose down -v

# Пересоздать БД
docker-compose up -d postgres
docker-compose run --rm app python scripts/setup_db.py
```

### Проверить данные в БД

```bash
# Подключиться к PostgreSQL
docker exec -it arima_postgres psql -U postgres

# SQL запросы
SELECT COUNT(*) FROM public.tickers;
SELECT COUNT(*) FROM public.model_results;
\dt all_dfs.*
```

---

## Troubleshooting

### Scheduler не собирает данные

```bash
# Проверить логи
docker logs arima_scheduler

# Проверить INVEST_TOKEN
docker exec arima_scheduler env | grep INVEST

# Перезапустить
docker-compose restart scheduler
```

### Dashboard не показывает данные

```bash
# Проверить подключение к БД
docker logs arima_dashboard

# Проверить что данные есть
docker exec arima_postgres psql -U postgres -c "SELECT COUNT(*) FROM public.model_results"
```

### Контейнер не запускается

```bash
# Посмотреть ошибки
docker-compose logs scheduler

# Пересобрать образ
docker-compose build --no-cache scheduler
docker-compose up -d scheduler
```

### Проблемы с памятью

Обучение моделей требует памяти. Увеличьте лимиты в `docker-compose.yml`:

```yaml
scheduler:
  ...
  deploy:
    resources:
      limits:
        memory: 4G
```

---

## Продакшен

### Рекомендации

1. **Используйте внешнюю БД** вместо контейнера PostgreSQL
2. **Настройте мониторинг** (Prometheus, Grafana)
3. **Добавьте health checks** для всех сервисов
4. **Используйте Docker Swarm или Kubernetes** для масштабирования

### Health checks

```yaml
# Добавить в docker-compose.yml
app:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Reverse proxy (nginx)

```nginx
server {
    listen 80;
    server_name trading.example.com;

    location / {
        proxy_pass http://localhost:8501;  # Dashboard
    }

    location /api/ {
        proxy_pass http://localhost:8000/;  # FastAPI
    }
}
```

---

## Полезные команды

```bash
# Войти в контейнер
docker exec -it arima_scheduler bash

# Выполнить команду
docker exec arima_scheduler python scripts/train_models.py --dry-run

# Скопировать файл из контейнера
docker cp arima_scheduler:/app/scheduler.log ./

# Очистить неиспользуемые образы
docker system prune -f
```
