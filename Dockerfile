FROM python:3.10-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git curl \
        libpq-dev \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Копирование исходного кода
COPY . .

# Порт FastAPI
EXPOSE 8002

# Дефолтная команда — можно переопределить из docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
