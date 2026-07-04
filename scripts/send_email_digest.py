#!/usr/bin/env python3
"""
Ежедневная рассылка дайджеста по email.

Кому: users где email_subscribed=true и есть unsubscribe_token.
Что: список свежих сигналов (свежий prediction_date из ticker_reports)
     + краткие итоги вчера (закрытые вердикты).

Запуск вручную:  python scripts/send_email_digest.py
                 python scripts/send_email_digest.py --dry-run
"""
import argparse
import logging
import os
import secrets
import sys
from datetime import datetime
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
logger = logging.getLogger("email_digest")


def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', '')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    return create_engine(url)


def _load_signals(engine):
    """Свежие AI-вердикты: последняя запись на каждый тикер."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (figi)
                ticker, verdict, entry_price, target_price,
                prediction_date, timestamp
            FROM public.ticker_reports
            WHERE prediction_date IS NOT NULL
              AND verdict IS NOT NULL
            ORDER BY figi, timestamp DESC
        """)).fetchall()
    return [
        {
            "ticker": r[0],
            "verdict": r[1],
            "entry_price": float(r[2]) if r[2] is not None else None,
            "target_price": float(r[3]) if r[3] is not None else None,
            "prediction_date": r[4],
        }
        for r in rows
    ]


def _load_resolved(engine):
    """Свежие закрытые вердикты: последний resolved на каждый тикер."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (figi)
                ticker, verdict, actual_close, correct_direction, prediction_date
            FROM public.ticker_reports
            WHERE actual_close IS NOT NULL
            ORDER BY figi, prediction_date DESC
        """)).fetchall()
    return [
        {
            "ticker": r[0],
            "verdict": r[1],
            "actual_close": float(r[2]) if r[2] is not None else None,
            "correct": r[3],
            "prediction_date": r[4],
        }
        for r in rows
    ]


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
    parser.add_argument("--dry-run", action="store_true", help="не отправлять, только напечатать")
    parser.add_argument("--test-to", help="отправить одно тестовое письмо на указанный email")
    args = parser.parse_args()

    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    signals = _load_signals(engine)
    resolved = _load_resolved(engine)
    if not signals and not resolved:
        logger.info("Нечего рассылать: нет ни новых сигналов, ни закрытых вердиктов")
        return

    prediction_date = signals[0]["prediction_date"] if signals else None
    base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8002").rstrip("/")

    jinja = Environment(
        loader=FileSystemLoader(str(PROJECT_ROOT / "templates")),
        autoescape=True,
    )
    template = jinja.get_template("email_digest.html")

    if args.test_to:
        subs = [(0, args.test_to, "тест", None)]
    else:
        rows = session.execute(text("""
            SELECT id, email, username, unsubscribe_token
            FROM public.users
            WHERE email_subscribed = TRUE
              AND is_active = TRUE
              AND email IS NOT NULL
        """)).fetchall()
        subs = [(r[0], r[1], r[2] or r[1], r[3]) for r in rows]

    if not subs:
        logger.info("Нет подписчиков — рассылка пропущена")
        return

    logger.info(f"Готовлю рассылку для {len(subs)} получателей…")
    sent, failed = 0, 0
    for user_id, email, username, token in subs:
        if args.test_to:
            token = "test"
        else:
            token = _ensure_token(session, user_id, token)

        html = template.render(
            username=username,
            date_str=datetime.now().strftime("%d.%m.%Y"),
            prediction_date=prediction_date.isoformat() if prediction_date else "",
            new_signals=signals,
            resolved=resolved,
            base_url=base_url,
            unsubscribe_url=f"{base_url}/auth/unsubscribe?token={token}",
        )
        subject = f"📈 Trading Signals · сигналы на {prediction_date}" if prediction_date \
                 else "📈 Trading Signals · дайджест"

        if args.dry_run:
            logger.info(f"[DRY-RUN] → {email} (subject: {subject})")
            sent += 1
            continue

        if send_email(email, subject, html):
            sent += 1
        else:
            failed += 1

    logger.info(f"Готово: отправлено {sent}, ошибок {failed}")


if __name__ == "__main__":
    main()
