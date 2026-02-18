#!/usr/bin/env python3
"""
Data collection script for financial prediction system.

Collects market data from Tinkoff Invest API and saves to PostgreSQL.

Usage:
    python scripts/collect_data.py              # Full collection (tickers + candles)
    python scripts/collect_data.py --tickers    # Only fetch tickers
    python scripts/collect_data.py --candles    # Only fetch candles (requires tickers)
    python scripts/collect_data.py --status     # Check data status
"""

import argparse
import asyncio
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_database_url():
    """Get database URL from environment variables."""
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def check_env():
    """Check required environment variables."""
    token = os.getenv("INVEST_TOKEN")
    if not token:
        logger.error("INVEST_TOKEN not found in environment!")
        logger.error("Please add INVEST_TOKEN to your .env file")
        logger.error("Get token at: https://www.tinkoff.ru/invest/settings/")
        return False

    db_host = os.getenv("DB_HOST")
    if not db_host:
        logger.error("Database configuration not found!")
        logger.error("Please configure DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD in .env")
        return False

    return True


def check_status():
    """Check current data status."""
    try:
        engine = create_engine(get_database_url())
        with engine.connect() as conn:
            # Check tickers
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM public.tickers"))
                tickers_count = result.scalar()
                logger.info(f"Tickers in database: {tickers_count}")

                # Show breakdown by type
                result = conn.execute(text(
                    "SELECT type, COUNT(*) FROM public.tickers GROUP BY type ORDER BY COUNT(*) DESC"
                ))
                for row in result:
                    logger.info(f"  - {row[0]}: {row[1]}")
            except Exception:
                logger.warning("No tickers table found")
                tickers_count = 0

            # Check data tables
            try:
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'all_dfs'"
                ))
                tables_count = result.scalar()
                logger.info(f"Data tables in all_dfs schema: {tables_count}")

                if tables_count > 0:
                    # Sample a few tables to show row counts
                    result = conn.execute(text("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'all_dfs' LIMIT 5
                    """))
                    sample_tables = [row[0] for row in result]

                    for table in sample_tables:
                        try:
                            count_result = conn.execute(
                                text(f'SELECT COUNT(*) FROM all_dfs."{table}"')
                            )
                            count = count_result.scalar()
                            logger.info(f"  - {table}: {count} rows")
                        except Exception:
                            pass
            except Exception:
                logger.warning("No all_dfs schema found")
                tables_count = 0

            return tickers_count, tables_count

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return 0, 0


def fetch_tickers():
    """Fetch tickers from Tinkoff API."""
    logger.info("=" * 50)
    logger.info("Fetching tickers from Tinkoff Invest API...")
    logger.info("=" * 50)

    try:
        # Import and run the tickers script
        import all_figi_to_db
        all_figi_to_db.main()
        logger.info("Tickers fetched successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return False


def fetch_candles():
    """Fetch historical candles for all tickers."""
    logger.info("=" * 50)
    logger.info("Fetching historical candles from Tinkoff Invest API...")
    logger.info("This may take a while depending on the number of tickers...")
    logger.info("=" * 50)

    try:
        # Import and run the candles script
        import all_dfs_to_db
        result = all_dfs_to_db.main()

        if result:
            logger.info(f"Candles fetched successfully!")
            logger.info(f"  - Success: {result.get('success', 0)}")
            logger.info(f"  - Errors: {result.get('errors', 0)}")
            logger.info(f"  - Total tickers: {result.get('total', 0)}")
            logger.info(f"  - Execution time: {result.get('execution_time', 0):.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Failed to fetch candles: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Collect market data from Tinkoff Invest API"
    )
    parser.add_argument(
        "--tickers",
        action="store_true",
        help="Only fetch tickers (instrument list)"
    )
    parser.add_argument(
        "--candles",
        action="store_true",
        help="Only fetch candles (requires tickers first)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check current data status"
    )
    args = parser.parse_args()

    # Status check doesn't require API token
    if args.status:
        check_status()
        return 0

    # Check environment
    if not check_env():
        return 1

    # Determine what to do
    if args.tickers:
        success = fetch_tickers()
    elif args.candles:
        success = fetch_candles()
    else:
        # Full collection
        logger.info("Starting full data collection...")
        success = fetch_tickers()
        if success:
            success = fetch_candles()

    # Show final status
    logger.info("")
    logger.info("=" * 50)
    logger.info("Final data status:")
    logger.info("=" * 50)
    check_status()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
