"""
Database utilities for Streamlit dashboard.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()


def get_engine():
    """Create database engine."""
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


def get_latest_signals(engine=None) -> pd.DataFrame:
    """
    Get latest trading signals for all tickers from all models.
    Returns one row per ticker+model combination with the most recent prediction.
    """
    if engine is None:
        engine = get_engine()

    query = """
    WITH ranked AS (
        SELECT
            mr.*,
            ROW_NUMBER() OVER (PARTITION BY db_name, model_name ORDER BY timestamp DESC) as rn
        FROM public.model_results mr
        WHERE trading_signal IS NOT NULL
    )
    SELECT
        db_name as figi,
        ticker_name as ticker,
        model_name,
        timestamp,
        trading_signal as signal,
        current_price,
        predicted_price,
        expected_change,
        test_r2 as r2,
        test_direction_accuracy as direction_accuracy,
        win_rate,
        cumulative_return,
        profit_factor
    FROM ranked
    WHERE rn = 1
    ORDER BY
        ticker_name,
        CASE trading_signal
            WHEN 'BUY' THEN 1
            WHEN 'SELL' THEN 2
            ELSE 3
        END,
        ABS(expected_change) DESC
    """

    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def get_model_comparison(engine=None) -> pd.DataFrame:
    """
    Compare model performance across all tickers.
    """
    if engine is None:
        engine = get_engine()

    query = """
    SELECT
        model_name,
        COUNT(DISTINCT db_name) as tickers_count,
        AVG(test_r2) as avg_r2,
        AVG(test_rmse) as avg_rmse,
        AVG(test_mae) as avg_mae,
        AVG(test_direction_accuracy) as avg_direction_accuracy,
        AVG(win_rate) as avg_win_rate,
        AVG(cumulative_return) as avg_return,
        AVG(profit_factor) as avg_profit_factor,
        COUNT(CASE WHEN trading_signal = 'BUY' THEN 1 END) as buy_signals,
        COUNT(CASE WHEN trading_signal = 'SELL' THEN 1 END) as sell_signals,
        COUNT(CASE WHEN trading_signal IN ('HOLD', 'NEUTRAL') THEN 1 END) as hold_signals
    FROM public.model_results
    WHERE timestamp > NOW() - INTERVAL '7 days'
    GROUP BY model_name
    ORDER BY avg_r2 DESC NULLS LAST
    """

    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def get_ticker_history(ticker_figi: str, engine=None) -> pd.DataFrame:
    """
    Get prediction history for a specific ticker.
    """
    if engine is None:
        engine = get_engine()

    query = """
    SELECT
        timestamp,
        model_name,
        trading_signal as signal,
        current_price,
        predicted_price,
        expected_change,
        test_r2 as r2,
        test_direction_accuracy as direction_accuracy,
        win_rate,
        cumulative_return
    FROM public.model_results
    WHERE db_name = :figi
    ORDER BY timestamp DESC
    LIMIT 100
    """

    try:
        return pd.read_sql(query, engine, params={'figi': ticker_figi})
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def get_available_tickers(engine=None) -> pd.DataFrame:
    """
    Get list of tickers with data.
    """
    if engine is None:
        engine = get_engine()

    query = """
    SELECT DISTINCT
        mr.db_name as figi,
        COALESCE(mr.ticker_name, t.ticker) as ticker,
        t.name as name,
        COUNT(*) as predictions_count,
        MAX(mr.timestamp) as last_prediction
    FROM public.model_results mr
    LEFT JOIN public.tickers t ON mr.db_name = t.figi
    GROUP BY mr.db_name, mr.ticker_name, t.ticker, t.name
    ORDER BY ticker
    """

    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def get_buy_recommendations(min_confidence: float = 0.6, engine=None) -> pd.DataFrame:
    """
    Get strong BUY recommendations with consensus from multiple models.
    Uses the latest prediction from each model for each ticker.
    """
    if engine is None:
        engine = get_engine()

    # Простой запрос с CTE
    query = """
    WITH latest AS (
        SELECT DISTINCT ON (db_name, model_name)
            db_name,
            ticker_name,
            model_name,
            trading_signal,
            expected_change,
            test_r2,
            test_direction_accuracy
        FROM public.model_results
        WHERE trading_signal IS NOT NULL
        ORDER BY db_name, model_name, timestamp DESC
    )
    SELECT
        db_name as figi,
        MAX(ticker_name) as ticker,
        COUNT(*) as total_models,
        SUM(CASE WHEN UPPER(trading_signal) = 'BUY' THEN 1 ELSE 0 END) as buy_votes,
        ROUND(
            SUM(CASE WHEN UPPER(trading_signal) = 'BUY' THEN 1 ELSE 0 END)::numeric /
            COUNT(*) * 100, 1
        ) as consensus_pct,
        COALESCE(ROUND(AVG(CASE WHEN UPPER(trading_signal) = 'BUY' THEN expected_change END)::numeric, 2), 0) as expected_change,
        COALESCE(ROUND(AVG(test_r2)::numeric, 4), 0) as avg_r2,
        ROUND(AVG(test_direction_accuracy)::numeric, 2) as direction_accuracy
    FROM latest
    GROUP BY db_name
    HAVING SUM(CASE WHEN UPPER(trading_signal) = 'BUY' THEN 1 ELSE 0 END) > 0
    ORDER BY consensus_pct DESC, expected_change DESC
    """

    try:
        df = pd.read_sql(query, engine)
        # Фильтруем по min_confidence на уровне Python
        if min_confidence > 0:
            df = df[df['consensus_pct'] >= min_confidence * 100]
        return df
    except Exception as e:
        print(f"Error get_buy_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def get_sell_recommendations(min_confidence: float = 0.6, engine=None) -> pd.DataFrame:
    """
    Get strong SELL recommendations with consensus from multiple models.
    Uses the latest prediction from each model for each ticker.
    """
    if engine is None:
        engine = get_engine()

    # Простой запрос с CTE
    query = """
    WITH latest AS (
        SELECT DISTINCT ON (db_name, model_name)
            db_name,
            ticker_name,
            model_name,
            trading_signal,
            expected_change,
            test_r2,
            test_direction_accuracy
        FROM public.model_results
        WHERE trading_signal IS NOT NULL
        ORDER BY db_name, model_name, timestamp DESC
    )
    SELECT
        db_name as figi,
        MAX(ticker_name) as ticker,
        COUNT(*) as total_models,
        SUM(CASE WHEN UPPER(trading_signal) = 'SELL' THEN 1 ELSE 0 END) as sell_votes,
        ROUND(
            SUM(CASE WHEN UPPER(trading_signal) = 'SELL' THEN 1 ELSE 0 END)::numeric /
            COUNT(*) * 100, 1
        ) as consensus_pct,
        COALESCE(ROUND(AVG(CASE WHEN UPPER(trading_signal) = 'SELL' THEN expected_change END)::numeric, 2), 0) as expected_change,
        COALESCE(ROUND(AVG(test_r2)::numeric, 4), 0) as avg_r2,
        ROUND(AVG(test_direction_accuracy)::numeric, 2) as direction_accuracy
    FROM latest
    GROUP BY db_name
    HAVING SUM(CASE WHEN UPPER(trading_signal) = 'SELL' THEN 1 ELSE 0 END) > 0
    ORDER BY consensus_pct DESC, expected_change ASC
    """

    try:
        df = pd.read_sql(query, engine)
        # Фильтруем по min_confidence на уровне Python
        if min_confidence > 0:
            df = df[df['consensus_pct'] >= min_confidence * 100]
        return df
    except Exception as e:
        print(f"Error get_sell_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def get_price_data(ticker_figi: str, engine=None) -> pd.DataFrame:
    """
    Get historical price data for a ticker from all_dfs schema.
    """
    if engine is None:
        engine = get_engine()

    query = f"""
    SELECT
        timestamp,
        open,
        high,
        low,
        close,
        volume
    FROM all_dfs."{ticker_figi}"
    ORDER BY timestamp DESC
    LIMIT 365
    """

    try:
        df = pd.read_sql(query, engine)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    except Exception as e:
        print(f"Error loading price data: {e}")
        return pd.DataFrame()


def get_summary_stats(engine=None) -> dict:
    """
    Get summary statistics for the dashboard.
    """
    if engine is None:
        engine = get_engine()

    stats = {}

    try:
        with engine.connect() as conn:
            # Total tickers
            result = conn.execute(text("SELECT COUNT(DISTINCT db_name) FROM public.model_results"))
            stats['total_tickers'] = result.scalar() or 0

            # Total predictions
            result = conn.execute(text("SELECT COUNT(*) FROM public.model_results"))
            stats['total_predictions'] = result.scalar() or 0

            # Predictions today
            result = conn.execute(text("""
                SELECT COUNT(*) FROM public.model_results
                WHERE timestamp::date = CURRENT_DATE
            """))
            stats['predictions_today'] = result.scalar() or 0

            # Average R2
            result = conn.execute(text("""
                SELECT AVG(test_r2) FROM public.model_results
                WHERE test_r2 IS NOT NULL
            """))
            stats['avg_r2'] = result.scalar() or 0

            # Buy/Sell signals today
            result = conn.execute(text("""
                SELECT
                    COUNT(CASE WHEN trading_signal = 'BUY' THEN 1 END) as buys,
                    COUNT(CASE WHEN trading_signal = 'SELL' THEN 1 END) as sells
                FROM public.model_results
                WHERE timestamp::date = CURRENT_DATE
            """))
            row = result.fetchone()
            stats['buy_signals_today'] = row[0] if row else 0
            stats['sell_signals_today'] = row[1] if row else 0

    except Exception as e:
        print(f"Error getting stats: {e}")
        stats = {
            'total_tickers': 0,
            'total_predictions': 0,
            'predictions_today': 0,
            'avg_r2': 0,
            'buy_signals_today': 0,
            'sell_signals_today': 0
        }

    return stats
