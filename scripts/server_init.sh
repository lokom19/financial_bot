#!/usr/bin/env bash
# =========================================================================
# Скрипт первоначальной настройки сервера для Trading Signals.
# Запускается на чистом Debian 12/13 от root (или sudo).
#
# Делает:
#   1. apt update + базовые утилиты
#   2. Устанавливает Docker + compose plugin
#   3. Создаёт пользователя deploy (опц.) с правами docker
#   4. Открывает порты 22/80/443 в ufw
#
# Использование:
#   ssh root@<сервер>
#   curl -fsSL https://raw.githubusercontent.com/<репо>/main/scripts/server_init.sh | bash
#   # или после git clone:
#   bash scripts/server_init.sh
# =========================================================================
set -euo pipefail

echo "============================================"
echo "  Trading Signals — настройка сервера"
echo "============================================"

if [ "$EUID" -ne 0 ]; then
    echo "Запусти от root или через sudo"; exit 1
fi

# --- 1) Базовые пакеты ---
echo "[1/4] Устанавливаю базовые пакеты..."
apt update
apt install -y --no-install-recommends \
    curl wget git ca-certificates gnupg lsb-release \
    ufw unattended-upgrades htop

# --- 2) Docker ---
echo "[2/4] Устанавливаю Docker..."
if ! command -v docker >/dev/null 2>&1; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    . /etc/os-release
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/debian $VERSION_CODENAME stable" \
        > /etc/apt/sources.list.d/docker.list

    apt update
    apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
fi
docker --version
docker compose version

# --- 3) Файрвол ---
echo "[3/4] Настраиваю файрвол (ufw)..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp     # SSH
ufw allow 80/tcp     # HTTP (для certbot + редирект)
ufw allow 443/tcp    # HTTPS
ufw --force enable

# --- 4) Auto-updates ---
echo "[4/4] Включаю автообновления безопасности..."
dpkg-reconfigure -plow unattended-upgrades || true

echo ""
echo "============================================"
echo "  Готово! Следующие шаги:"
echo "============================================"
echo "  1. Скопируй .env (с реальными ключами) в директорию проекта"
echo "  2. cd <project>"
echo "  3. bash scripts/deploy.sh"
echo "============================================"
