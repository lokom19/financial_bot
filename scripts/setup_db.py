#!/usr/bin/env python3
"""
Database setup script.

Creates all required tables and schemas for the financial prediction system.

Usage:
    python scripts/setup_db.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker

load_dotenv()


def get_database_url():
    """Get database URL from environment variables."""
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def setup_database():
    """Create all required database objects."""
    database_url = get_database_url()
    print(f"Connecting to: {database_url.replace(os.getenv('DB_PASSWORD', ''), '***')}")

    try:
        engine = create_engine(database_url)

        with engine.connect() as conn:
            # Create schemas
            print("\n1. Creating schemas...")
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS public;"))
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS all_dfs;"))
            conn.commit()
            print("   ✓ Schemas created: public, all_dfs")

            # Create model_results table
            print("\n2. Creating model_results table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS public.model_results (
                    id SERIAL PRIMARY KEY,
                    db_name VARCHAR(255) NOT NULL,
                    ticker_name VARCHAR(50),
                    model_name VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    text TEXT NOT NULL,

                    -- Dataset info
                    train_samples INTEGER,
                    test_samples INTEGER,
                    data_start_date DATE,
                    data_end_date DATE,

                    -- Test metrics
                    test_mse FLOAT,
                    test_rmse FLOAT,
                    test_mae FLOAT,
                    test_r2 FLOAT,
                    test_mape FLOAT,
                    test_direction_accuracy FLOAT,

                    -- Train metrics
                    train_mse FLOAT,
                    train_rmse FLOAT,
                    train_mae FLOAT,
                    train_r2 FLOAT,
                    train_direction_accuracy FLOAT,

                    -- Predictions
                    current_price FLOAT,
                    predicted_price FLOAT,
                    expected_change FLOAT,
                    prediction_std FLOAT,
                    trading_signal VARCHAR(10),

                    -- Trading performance (backtest)
                    total_trades INTEGER,
                    profitable_trades INTEGER,
                    win_rate FLOAT,
                    profit_factor FLOAT,
                    cumulative_return FLOAT
                );
            """))
            conn.commit()
            print("   ✓ Table created: public.model_results")

            # Add new columns if they don't exist (migration for existing tables)
            print("\n2.1. Adding new columns (if missing)...")
            new_columns = [
                ("ticker_name", "VARCHAR(50)"),
                ("train_samples", "INTEGER"),
                ("test_samples", "INTEGER"),
                ("data_start_date", "DATE"),
                ("data_end_date", "DATE"),
                ("train_mse", "FLOAT"),
                ("train_rmse", "FLOAT"),
                ("train_mae", "FLOAT"),
                ("train_r2", "FLOAT"),
                ("prediction_std", "FLOAT"),
                ("total_trades", "INTEGER"),
                ("profitable_trades", "INTEGER"),
                ("win_rate", "FLOAT"),
                ("profit_factor", "FLOAT"),
                ("cumulative_return", "FLOAT"),
            ]
            for col_name, col_type in new_columns:
                try:
                    conn.execute(text(f"""
                        ALTER TABLE public.model_results
                        ADD COLUMN IF NOT EXISTS {col_name} {col_type};
                    """))
                except Exception:
                    pass  # Column might already exist
            conn.commit()
            print("   ✓ New columns added")

            # Create indexes
            print("\n3. Creating indexes...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_results_model_name
                ON public.model_results(model_name);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_results_db_name
                ON public.model_results(db_name);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_results_timestamp
                ON public.model_results(timestamp DESC);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_results_signal
                ON public.model_results(trading_signal);
            """))
            conn.commit()
            print("   ✓ Indexes created")

            # Verify setup
            print("\n4. Verifying setup...")
            inspector = inspect(engine)

            # Check tables
            tables = inspector.get_table_names(schema='public')
            print(f"   Tables in public schema: {tables}")

            # Check columns in model_results
            if 'model_results' in tables:
                columns = [col['name'] for col in inspector.get_columns('model_results', schema='public')]
                print(f"   Columns in model_results: {len(columns)}")

            print("\n" + "=" * 50)
            print("Database setup completed successfully!")
            print("=" * 50)

        engine.dispose()
        return True

    except Exception as e:
        print(f"\n✗ Error setting up database: {e}")
        return False


def check_connection():
    """Quick connection check."""
    database_url = get_database_url()
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        engine.dispose()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database setup script")
    parser.add_argument("--check", action="store_true", help="Only check connection")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_connection() else 1)
    else:
        sys.exit(0 if setup_database() else 1)
