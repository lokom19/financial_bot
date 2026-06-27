#!/usr/bin/env bash
# =========================================================================
# Развёртывание Trading Signals на сервере с нуля до HTTPS.
#
# Предполагается:
#   - server_init.sh уже выполнен (Docker есть)
#   - .env создан и заполнен (см. .env.example)
#   - DNS-запись A на этот сервер уже настроена (для домена из .env)
#
# Что делает:
#   1. Поднимает временный nginx (HTTP) для ACME challenge
#   2. Получает SSL-сертификат Let's Encrypt
#   3. Включает HTTPS-конфиг
#   4. Стартует всё стек целиком
# =========================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ОШИБКА: нет файла .env. Скопируй .env.example → .env и заполни."
    exit 1
fi

# Загрузим env, чтобы взять DOMAIN
set -a
. ./.env
set +a

DOMAIN="${DOMAIN:-}"
EMAIL="${LETSENCRYPT_EMAIL:-}"

if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
    echo "ОШИБКА: в .env должны быть DOMAIN и LETSENCRYPT_EMAIL"
    exit 1
fi

echo "============================================"
echo "  Деплой Trading Signals"
echo "  Домен: $DOMAIN"
echo "  Email: $EMAIL"
echo "============================================"

# --- 1) Bootstrap nginx (только HTTP) ---
if [ ! -d nginx/certbot/conf/live/"$DOMAIN" ]; then
    echo "[1/4] SSL-сертификата ещё нет. Запускаю bootstrap-режим..."

    # Заменяем HTTPS-конфиг на bootstrap (HTTP-only)
    mv nginx/conf.d/app.conf nginx/conf.d/app.conf.full 2>/dev/null || true
    cp nginx/conf.d/app.bootstrap.conf.disabled nginx/conf.d/app.conf

    # Подменяем домен в bootstrap-конфиге (на случай если в .env другой)
    sed -i "s|vm123123new.firstbyte.club|$DOMAIN|g" nginx/conf.d/app.conf

    echo "[2/4] Поднимаю стек без HTTPS чтобы certbot мог достучаться..."
    docker compose up -d --build postgres migrate app dashboard nginx

    echo "[3/4] Жду 10 секунд пока nginx стартует..."
    sleep 10

    echo "Запрашиваю SSL-сертификат у Let's Encrypt..."
    # --entrypoint certbot ОБЯЗАТЕЛЬНО — иначе сработает entrypoint из compose
    # (бесконечный renew-loop) и наш certonly будет проигнорирован.
    docker compose run --rm --entrypoint certbot certbot \
        certonly --webroot --webroot-path=/var/www/certbot \
        --email "$EMAIL" --agree-tos --no-eff-email \
        --non-interactive \
        -d "$DOMAIN"

    echo "Возвращаю полный HTTPS-конфиг..."
    rm nginx/conf.d/app.conf
    mv nginx/conf.d/app.conf.full nginx/conf.d/app.conf
    # Подменяем домен в полном конфиге тоже
    sed -i "s|vm123123new.firstbyte.club|$DOMAIN|g" nginx/conf.d/app.conf

    docker compose restart nginx
else
    echo "[1/4] Сертификат уже есть, пропускаю bootstrap."
fi

# --- 2) Поднимаем всё ---
echo "[4/4] Стартую полный стек..."
docker compose up -d --build

# --- 3) Статус ---
echo ""
echo "============================================"
echo "  Статус контейнеров:"
echo "============================================"
docker compose ps

echo ""
echo "============================================"
echo "  Готово!"
echo "============================================"
echo "  Сайт:       https://$DOMAIN"
echo "  Streamlit:  https://$DOMAIN/dashboard/"
echo "  Логи:       docker compose logs -f app"
echo "  Стоп:       docker compose down"
echo "============================================"
