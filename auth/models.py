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

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"
