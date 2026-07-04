from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Ежедневная email-рассылка. Опт-ин по умолчанию false — иначе
    # можно спамить неверифицированные адреса. Включается в личном
    # кабинете. unsubscribe_token — HMAC-подобный секрет для ссылки
    # отписки в футере, чтобы не требовать логина.
    email_subscribed = Column(Boolean, default=False, nullable=False)
    unsubscribe_token = Column(String(64), nullable=True, unique=True)

    # Верификация email при регистрации.
    # is_active=false до ввода правильного 6-значного кода.
    email_verified_at = Column(DateTime, nullable=True)
    verification_code = Column(String(6), nullable=True)
    verification_expires_at = Column(DateTime, nullable=True)

    # Восстановление пароля через email.
    # Ссылка "/auth/reset-password?token=…" валидна 1 час.
    password_reset_token = Column(String(64), nullable=True, index=True)
    password_reset_expires_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"
