FROM python:3.10-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git curl wget \
        libpq-dev \
        tzdata \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Российские корневые сертификаты от Минцифры — нужны для подключения
# к Tinkoff Invest API, Сбер и др. (используют отечественный TLS).
# https://www.gosuslugi.ru/crt
RUN mkdir -p /usr/local/share/ca-certificates/russian_trusted && \
    curl -fsSL https://gu-st.ru/content/Other/doc/russian_trusted_root_ca.cer \
        -o /usr/local/share/ca-certificates/russian_trusted/russian_root_ca.crt && \
    curl -fsSL https://gu-st.ru/content/Other/doc/russian_trusted_sub_ca.cer \
        -o /usr/local/share/ca-certificates/russian_trusted/russian_sub_ca.crt && \
    update-ca-certificates

# certifi (используется requests/httpx) тоже должен видеть русский CA.
# Добавляем его в certifi-cabundle через монтаж после установки пакетов.
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# Для grpc (Tinkoff SDK использует grpc)
ENV GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/etc/ssl/certs/ca-certificates.crt

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
