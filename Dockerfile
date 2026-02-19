FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей
# Используем requirements-docker.txt для Docker (без проблемных пакетов)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
#RUN mkdir -p /app/templates /app/static /app/output

# Открытие порта
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]