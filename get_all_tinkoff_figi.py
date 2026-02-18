from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST") # "localhost"  # Change this to your database host
DB_PORT = os.getenv("DB_PORT")  # "5432"  # Default PostgreSQL port
DB_NAME = os.getenv("DB_NAME")    # "postgres"  # Change to your database name
DB_USER = os.getenv("DB_USER")      # "postgres"  # Change to your username
DB_PASSWORD = os.getenv("DB_PASSWORD")    # "mysecretpassword"  # Change to your password
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)


def get_tickers_from_bd():
    """Получить список тикеров из БД."""
    with engine.connect() as conn:
        res = conn.execute(text("SELECT figi, ticker, name FROM tickers WHERE type IN ('shares', 'futures')")).fetchall()
        return res

