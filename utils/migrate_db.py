import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")  # "localhost"  # Change this to your database host
DB_PORT = os.getenv("DB_PORT")  # "5432"  # Default PostgreSQL port
DB_NAME = os.getenv("DB_NAME")  # "postgres"  # Change to your database name
DB_USER = os.getenv("DB_USER")  # "postgres"  # Change to your username
DB_PASSWORD = os.getenv("DB_PASSWORD")  # "mysecretpassword"  # Change to your password
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def quick_migration():
    """
    Быстрая миграция - добавляет только нужные колонки
    """
    from sqlalchemy import create_engine, text

    # Ваш connection string
    engine = create_engine(DATABASE_URL)

    sql = """
    ALTER TABLE public.model_results 
    ADD COLUMN IF NOT EXISTS test_mse FLOAT,
    ADD COLUMN IF NOT EXISTS test_rmse FLOAT,
    ADD COLUMN IF NOT EXISTS test_mae FLOAT,
    ADD COLUMN IF NOT EXISTS test_r2 FLOAT,
    ADD COLUMN IF NOT EXISTS test_mape FLOAT,
    ADD COLUMN IF NOT EXISTS test_direction_accuracy FLOAT,
    ADD COLUMN IF NOT EXISTS train_direction_accuracy FLOAT,
    ADD COLUMN IF NOT EXISTS current_price FLOAT,
    ADD COLUMN IF NOT EXISTS predicted_price FLOAT,
    ADD COLUMN IF NOT EXISTS expected_change FLOAT,
    ADD COLUMN IF NOT EXISTS trading_signal VARCHAR(10);
    """

    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text(sql))

        print("✅ Миграция завершена успешно!")
        print("Добавлены колонки:")
        print("  • Метрики модели: test_mse, test_rmse, test_mae, test_r2, test_mape")
        print("  • Точность направления: test_direction_accuracy, train_direction_accuracy")
        print("  • Прогнозные данные: current_price, predicted_price, expected_change, trading_signal")

    except Exception as e:
        print(f"❌ Ошибка миграции: {e}")


quick_migration()