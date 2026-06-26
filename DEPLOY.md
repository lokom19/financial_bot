# Деплой Trading Signals на сервер

Полная инструкция от пустого Debian-сервера до работающего HTTPS-сайта.

## Что у нас есть

- Сервер: `vm123123new.firstbyte.club` (104.128.139.126)
- OS: Debian 13 amd64
- Сервисы (всё в Docker):
  - `postgres` — БД
  - `app` — FastAPI (порт 8002 внутри сети)
  - `dashboard` — Streamlit (порт 8501)
  - `scheduler` — ночной пайплайн (cron 02:00 МСК)
  - `nginx` — реверс-прокси + HTTPS (порты 80/443 наружу)
  - `certbot` — автообновление SSL
  - `migrate` — миграции БД (запускается один раз)

После деплоя:
- `https://vm123123new.firstbyte.club/` — FastAPI (логин, модели, AI-отчёты)
- `https://vm123123new.firstbyte.club/dashboard/` — Streamlit

---

## Шаг 1. SSH на сервер и базовая настройка

```bash
# С локальной машины
ssh root@104.128.139.126
# (введи пароль)
```

Первым делом — **сменить пароль root и создать пользователя**:

```bash
# На сервере, под root
passwd                       # новый сильный пароль
adduser deploy               # создаём отдельного юзера
usermod -aG sudo deploy      # права sudo
```

Скопируй с локальной машины свой SSH-ключ:

```bash
# На локальной машине
ssh-copy-id deploy@104.128.139.126
ssh-copy-id root@104.128.139.126   # для надёжности
```

Запрети вход по паролю:

```bash
# На сервере под root
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
systemctl restart ssh
```

## Шаг 2. Установка Docker и базовых пакетов

```bash
# На сервере (под root или sudo)
apt update
apt install -y curl git ca-certificates gnupg lsb-release ufw
```

Запусти готовый скрипт инициализации:

```bash
git clone <твой-репо> /opt/arima
cd /opt/arima
bash scripts/server_init.sh
```

(Если репо ещё не доступен публично — `scp` всё в `/opt/arima` или загружай через rsync.)

Скрипт установит Docker + compose plugin + откроет порты 22/80/443 в файрволе + включит автообновления безопасности.

## Шаг 3. DNS

Убедись, что A-запись домена смотрит на IP:

```bash
dig +short vm123123new.firstbyte.club
# должно вернуть: 104.128.139.126
```

Если нет — добавь A-запись в DNS-панели хостинга и подожди ~5-10 минут.

## Шаг 4. Заполни .env

```bash
cd /opt/arima
cp .env.example .env
nano .env
```

Минимум что должно быть заполнено:

```ini
# Деплой
DOMAIN=vm123123new.firstbyte.club
LETSENCRYPT_EMAIL=твоя@почта.com    # реальная — для Let's Encrypt

# БД
DB_HOST=postgres
DB_PORT=5432
DB_NAME=arima
DB_USER=arima_user
DB_PASSWORD=<длинный_случайный_пароль>     # сгенерируй: openssl rand -hex 24

# Auth
SECRET_KEY=<длинный_случайный_ключ>        # openssl rand -hex 32

# LLM (Groq — бесплатно, console.groq.com)
GROQ_API_KEY=gsk_твой_новый_ключ
GROQ_MODEL=llama-3.3-70b-versatile

# Tinkoff API (для подтягивания свечей)
INVEST_TOKEN=твой_тинькофф_токен

# Расписание
NIGHTLY_HOUR=2
NIGHTLY_MINUTE=0
TZ=Europe/Moscow
```

## Шаг 5. Развёртывание

```bash
cd /opt/arima
bash scripts/deploy.sh
```

Скрипт сделает:
1. Поднимет временный nginx без HTTPS
2. Запросит SSL у Let's Encrypt (через ACME challenge)
3. Включит полный HTTPS-конфиг
4. Поднимет весь стек

В конце увидишь:

```
============================================
  Готово!
============================================
  Сайт:       https://vm123123new.firstbyte.club
  Streamlit:  https://vm123123new.firstbyte.club/dashboard/
```

## Шаг 6. Первый прогон данных

После запуска БД пуста. Залей первые данные:

```bash
# Подтянуть список тикеров (SBER, OZON, VTBR в test-режиме)
docker compose exec app python all_figi_to_db.py --test

# Подтянуть исторические свечи (1500 дней)
docker compose exec app python all_dfs_to_db.py --days 1500 --interval day

# Запустить ночной пайплайн прямо сейчас (а не ждать 02:00)
docker compose exec scheduler python scripts/nightly_pipeline.py
```

Через несколько минут (зависит от мощности сервера) откройте `https://домен/` и зарегистрируйтесь.

---

## Управление в продакшне

```bash
# Статус всех контейнеров
docker compose ps

# Логи (хвост)
docker compose logs -f app          # FastAPI
docker compose logs -f scheduler    # ночной пайплайн
docker compose logs -f nginx

# Перезапуск одного сервиса
docker compose restart app

# Полная перезапуск
docker compose down && docker compose up -d

# Обновление кода
git pull
docker compose build app dashboard scheduler
docker compose up -d
```

## Бэкап БД

```bash
# Создать дамп
docker compose exec postgres pg_dump -U arima_user arima > backup_$(date +%F).sql

# Восстановить
cat backup_2026-06-26.sql | docker compose exec -T postgres psql -U arima_user arima
```

## Безопасность

**После деплоя**:
1. ✅ Пароль root изменён
2. ✅ SSH вход только по ключу
3. ✅ Файрвол открыт только на 22/80/443
4. ✅ HTTPS включён
5. **Ротируй** `GROQ_API_KEY` и `INVEST_TOKEN` если они уже где-то засветились

## Если что-то не так

```bash
# nginx не стартует — проверь конфиг
docker compose exec nginx nginx -t

# certbot не получил серт — проверь что DNS уже резолвится в этот IP
dig +short vm123123new.firstbyte.club

# app падает — посмотри миграции
docker compose logs migrate
docker compose logs app

# Принудительно пересоздать всё
docker compose down -v   # ⚠️ удалит данные БД!
bash scripts/deploy.sh
```
