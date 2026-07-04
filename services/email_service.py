"""
Отправка email-писем через SMTP.

Универсальный клиент — подойдёт любой провайдер: Яндекс.360,
Mail.ru, SendGrid, Mailgun, локальный postfix.
Настройки через переменные окружения (см. .env.example).
"""
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from typing import Optional

logger = logging.getLogger(__name__)


def _is_configured() -> bool:
    return bool(
        os.getenv("EMAIL_SMTP_HOST")
        and os.getenv("EMAIL_SMTP_USER")
        and os.getenv("EMAIL_SMTP_PASSWORD")
        and os.getenv("EMAIL_FROM")
    )


def send_email(
    to_email: str,
    subject: str,
    html_body: str,
    text_body: Optional[str] = None,
) -> bool:
    """Отправляет одно письмо. Возвращает True при успехе."""
    if os.getenv("EMAIL_ENABLED", "true").lower() != "true":
        logger.info("Email disabled via EMAIL_ENABLED=false")
        return False
    if not _is_configured():
        logger.warning("SMTP не настроен (EMAIL_SMTP_* переменные пустые) — письмо не отправлено")
        return False

    host = os.getenv("EMAIL_SMTP_HOST")
    port = int(os.getenv("EMAIL_SMTP_PORT", "465"))
    user = os.getenv("EMAIL_SMTP_USER")
    password = os.getenv("EMAIL_SMTP_PASSWORD")
    from_email = os.getenv("EMAIL_FROM")
    from_name = os.getenv("EMAIL_FROM_NAME", "Trading Signals")
    use_ssl = os.getenv("EMAIL_SMTP_SSL", "true").lower() == "true"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((from_name, from_email))
    msg["To"] = to_email
    msg.set_content(text_body or "Откройте письмо в HTML-режиме.")
    msg.add_alternative(html_body, subtype="html")

    try:
        if use_ssl:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=30) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                smtp.starttls(context=ssl.create_default_context())
                smtp.login(user, password)
                smtp.send_message(msg)
        logger.info(f"Письмо отправлено на {to_email}")
        return True
    except Exception as e:
        logger.error(f"SMTP ошибка при отправке на {to_email}: {e}")
        return False
