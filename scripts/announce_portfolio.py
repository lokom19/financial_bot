#!/usr/bin/env python3
"""
Разовая рассылка знакомства с фичей "Калькулятор портфеля".

Отличия от ежедневного дайджеста:
- Один HTML-шаблон, без свежих сигналов и итогов вчера.
- По умолчанию отправляет ВСЕМ активным юзерам (не только тем, у кого
  email_subscribed=true) — это функциональное уведомление о новой возможности.
- Опционально можно ограничить только подписчиками через --subscribed-only.

Примеры:
    # dry-run — покажет кому пошло бы, но не отправит
    python scripts/announce_portfolio.py --dry-run

    # тестовое письмо только на один адрес
    python scripts/announce_portfolio.py --test-to lokom19@mail.ru

    # реальная рассылка ВСЕМ активным юзерам
    python scripts/announce_portfolio.py

    # только подписчикам (email_subscribed=true)
    python scripts/announce_portfolio.py --subscribed-only
"""
import argparse
import logging
import os
import secrets
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from services.email_service import send_email

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("announce_portfolio")


def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', '')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    return create_engine(url)


def _ensure_token(session, user_id: int, current_token) -> str:
    """Возвращает unsubscribe_token, создавая новый если пусто."""
    if current_token:
        return current_token
    token = secrets.token_urlsafe(32)[:64]
    session.execute(
        text("UPDATE public.users SET unsubscribe_token = :t WHERE id = :i"),
        {"t": token, "i": user_id},
    )
    session.commit()
    return token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="показать кому пошло бы, но не отправлять")
    parser.add_argument("--test-to",
                        help="отправить одно тестовое письмо на указанный email")
    parser.add_argument("--subscribed-only", action="store_true",
                        help="только тем, у кого email_subscribed=true")
    args = parser.parse_args()

    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8002").rstrip("/")
    subject = "Trading Signals: калькулятор портфеля"

    jinja = Environment(
        loader=FileSystemLoader(str(PROJECT_ROOT / "templates")),
        autoescape=True,
    )
    template = jinja.get_template("email_portfolio_announce.html")

    if args.test_to:
        subs = [(0, args.test_to, "тест", None)]
    else:
        query = """
            SELECT id, email, username, unsubscribe_token
            FROM public.users
            WHERE is_active = TRUE
              AND email IS NOT NULL
        """
        if args.subscribed_only:
            query += " AND email_subscribed = TRUE"
        rows = session.execute(text(query)).fetchall()
        subs = [(r[0], r[1], r[2] or r[1], r[3]) for r in rows]

    if not subs:
        logger.info("Нет получателей — рассылка пропущена")
        return

    logger.info(f"Готовлю рассылку для {len(subs)} получателей…")
    sent, failed = 0, 0
    for user_id, email, username, token in subs:
        if args.test_to:
            token = "test"
        else:
            token = _ensure_token(session, user_id, token)

        unsubscribe_url = f"{base_url}/auth/unsubscribe?token={token}" if token else None
        html = template.render(
            username=username, base_url=base_url,
            unsubscribe_url=unsubscribe_url,
        )
        # Текстовый вариант — многие спам-фильтры (mail.ru особенно) любят
        # видеть plaintext альтернативу и меньше подозрительно относятся к письму.
        text_body = (
            f"Здравствуйте, {username}.\n\n"
            f"В личном кабинете появилась новая страница — калькулятор портфеля.\n"
            f"Он показывает, как вёл бы себя депозит, если бы каждый день следовать\n"
            f"сохранённым вердиктам: кривая капитала, win rate, лучшая/худшая сделка,\n"
            f"разбивка выходов (цель / стоп / закрытие дня), список сделок.\n\n"
            f"Открыть: {base_url}/portfolio\n\n"
            f"Результаты гипотетические, без учёта проскальзывания и налогов.\n"
            f"Не является инвестиционной рекомендацией.\n"
        )
        if unsubscribe_url:
            text_body += f"\nОтписаться: {unsubscribe_url}\n"

        if args.dry_run:
            logger.info(f"[DRY-RUN] → {email}")
            sent += 1
            continue

        if send_email(email, subject, html, text_body=text_body):
            sent += 1
        else:
            failed += 1

    logger.info(f"Готово: отправлено {sent}, ошибок {failed}")


if __name__ == "__main__":
    main()
