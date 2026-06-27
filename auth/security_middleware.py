"""
Security: rate-limiting + Cloudflare Turnstile (CAPTCHA).
"""
import os
import logging
from typing import Optional

import requests
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)


# ============================================================
# Rate limiter
# ============================================================
# Учитывает X-Forwarded-For при работе за nginx
def _key_func(request: Request) -> str:
    # Если за reverse-proxy (nginx) — берём реальный IP клиента
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri
    return get_remote_address(request)


limiter = Limiter(key_func=_key_func)


# ============================================================
# Cloudflare Turnstile
# ============================================================
# Регистрируем сайт на https://dash.cloudflare.com/?to=/:account/turnstile
# Получаем site_key (для фронта) и secret_key (для бэка).
# Тестовый ключ "1x00000000000000000000AA" / "1x0000000000000000000000000000000AA"
# всегда возвращает success (полезно для разработки).
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "").strip()
TURNSTILE_SITE_KEY = os.getenv("TURNSTILE_SITE_KEY", "").strip()
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def is_turnstile_enabled() -> bool:
    """Включён ли Turnstile (есть оба ключа)."""
    return bool(TURNSTILE_SECRET_KEY and TURNSTILE_SITE_KEY)


def verify_turnstile(token: str, client_ip: Optional[str] = None) -> bool:
    """
    Проверяет токен от Turnstile через API Cloudflare.
    Возвращает True/False. Если ключи не настроены — всегда True (отключено).
    """
    if not is_turnstile_enabled():
        return True   # Капча не настроена → пропускаем
    if not token:
        return False

    try:
        data = {"secret": TURNSTILE_SECRET_KEY, "response": token}
        if client_ip:
            data["remoteip"] = client_ip
        resp = requests.post(TURNSTILE_VERIFY_URL, data=data, timeout=10)
        if resp.status_code == 200:
            return bool(resp.json().get("success"))
        logger.warning("Turnstile verify HTTP %s: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("Turnstile verify error: %s", e)
    return False
