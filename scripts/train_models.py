#!/usr/bin/env python3
"""
Unified model training script.

Trains all (or selected) models on available data and saves results to PostgreSQL.

Usage:
    # Train all models on all tickers
    python scripts/train_models.py

    # Train specific model
    python scripts/train_models.py --model ridge

    # Train on specific ticker
    python scripts/train_models.py --ticker BBG000Q7ZZY2

    # List available models
    python scripts/train_models.py --list-models

    # Dry run (no DB save)
    python scripts/train_models.py --dry-run
"""

import os
import sys
import re
import io
import argparse
import logging
from datetime import datetime
from contextlib import redirect_stdout
from typing import List, Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Available models registry
AVAILABLE_MODELS = {
    'ridge': {
        'module': 'models.ridge',
        'function': 'main',
        'description': 'Ridge Regression'
    },
    'xgboost': {
        'module': 'models.xgboost_model',
        'function': 'main',
        'description': 'XGBoost Gradient Boosting'
    },
    'lightgbm': {
        'module': 'models.lightgbm_model',
        'function': 'main',
        'description': 'LightGBM Gradient Boosting'
    },
    'catboost': {
        'module': 'models.cat_boost_model',
        'function': 'main',
        'description': 'CatBoost Gradient Boosting'
    },
    'random_forest': {
        'module': 'models.random_forest_regression_model',
        'function': 'main',
        'description': 'Random Forest Regression'
    },
    'arima': {
        'module': 'models.arima',
        'function': 'main',
        'description': 'ARIMA Time Series'
    },
    'rf_classifier': {
        'module': 'models.rf_classifier',
        'function': 'main',
        'description': 'Random Forest Classifier'
    },
}


def get_database_engine():
    """Create database engine from environment."""
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


def get_available_tickers(engine) -> List[str]:
    """Get list of available tickers from database."""
    try:
        with engine.connect() as conn:
            # Get all tables in all_dfs schema
            result = conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'all_dfs'
                ORDER BY table_name
            """))
            tickers = [row[0] for row in result]
            return tickers
    except Exception as e:
        logger.error(f"Error getting tickers: {e}")
        return []


def extract_metrics(output_text: str) -> Dict[str, Any]:
    """Extract metrics from model output text."""
    # Паттерны для извлечения метрик из текстового вывода модели
    patterns = {
        # Test metrics
        'test_mse': r'MSE: ([\d.]+)',
        'test_rmse': r'RMSE: ([\d.]+)',
        'test_mae': r'MAE: ([\d.]+)',
        'test_r2': r'R²: ([\d.]+)',
        'test_mape': r'MAPE: ([\d.]+)',
        'test_direction_accuracy': r'Direction Accuracy: ([\d.]+)',

        # Predictions
        'current_price': r'Текущая цена: ([\d.]+)',
        'predicted_price': r'Прогнозируемая цена: ([\d.]+)',
        'expected_change': r'Ожидаемое изменение: ([-+]?[\d.]+)%',
        'trading_signal': r'Торговый сигнал: (\w+)',

        # Trading performance (backtest)
        'total_trades': r'Всего сделок: (\d+)',
        'profitable_trades': r'Прибыльных сделок: (\d+)',
        'win_rate': r'Прибыльных сделок: \d+ \(([\d.]+)%',
        'cumulative_return': r'Общая доходность: ([-+]?[\d.]+)%',
        'profit_factor': r'Коэффициент прибыли.*?: ([\d.]+|inf)',

        # Dataset info
        'total_samples': r'Загружено (\d+) записей',
    }

    metrics = {}

    # Извлекаем метрики
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if key == 'trading_signal':
                metrics[key] = value.strip()
            elif key == 'profit_factor' and value == 'inf':
                metrics[key] = None  # Бесконечный profit factor = нет убыточных сделок
            elif key in ['total_trades', 'profitable_trades', 'total_samples']:
                try:
                    metrics[key] = int(value)
                except ValueError:
                    metrics[key] = None
            else:
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = None
        else:
            metrics[key] = None

    # Извлекаем период данных
    period_match = re.search(
        r'за период с (\d{4}-\d{2}-\d{2})[T\s][\d:]+\s+по\s+(\d{4}-\d{2}-\d{2})',
        output_text
    )
    if period_match:
        metrics['data_start_date'] = period_match.group(1)
        metrics['data_end_date'] = period_match.group(2)
    else:
        metrics['data_start_date'] = None
        metrics['data_end_date'] = None

    # Рассчитываем train/test samples (80/20 split)
    if metrics.get('total_samples'):
        total = metrics.pop('total_samples')
        metrics['train_samples'] = int(total * 0.8)
        metrics['test_samples'] = int(total * 0.2)
    else:
        metrics['train_samples'] = None
        metrics['test_samples'] = None

    return metrics


def get_ticker_name(engine, figi: str) -> Optional[str]:
    """Get human-readable ticker name from FIGI."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT ticker FROM public.tickers WHERE figi = :figi"
            ), {'figi': figi})
            row = result.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def save_result(engine, ticker: str, model_name: str, output_text: str) -> bool:
    """Save training result to database."""
    try:
        metrics = extract_metrics(output_text)
        timestamp = datetime.now()

        # Get human-readable ticker name
        ticker_name = get_ticker_name(engine, ticker)

        Session = sessionmaker(bind=engine)
        session = Session()

        # Insert with all new columns
        session.execute(text("""
            INSERT INTO public.model_results (
                db_name, ticker_name, model_name, timestamp, text,
                train_samples, test_samples, data_start_date, data_end_date,
                test_mse, test_rmse, test_mae, test_r2, test_mape,
                test_direction_accuracy, current_price, predicted_price,
                expected_change, trading_signal,
                total_trades, profitable_trades, win_rate,
                profit_factor, cumulative_return
            ) VALUES (
                :db_name, :ticker_name, :model_name, :timestamp, :text,
                :train_samples, :test_samples, :data_start_date, :data_end_date,
                :test_mse, :test_rmse, :test_mae, :test_r2, :test_mape,
                :test_direction_accuracy, :current_price, :predicted_price,
                :expected_change, :trading_signal,
                :total_trades, :profitable_trades, :win_rate,
                :profit_factor, :cumulative_return
            )
        """), {
            'db_name': ticker,
            'ticker_name': ticker_name,
            'model_name': model_name,
            'timestamp': timestamp,
            'text': output_text,
            **metrics
        })

        session.commit()
        session.close()

        # Log with signal and key metrics
        signal = metrics.get('trading_signal', 'N/A')
        r2 = metrics.get('test_r2')
        r2_str = f"R²={r2:.3f}" if r2 else ""
        logger.info(f"  Saved: {ticker_name or ticker} | {signal} | {r2_str}")
        return True

    except Exception as e:
        logger.error(f"  Error saving result: {e}")
        return False


def train_model(model_name: str, ticker: str) -> Optional[str]:
    """
    Train a single model on a ticker and capture output.

    Returns the captured output text or None on failure.
    """
    if model_name not in AVAILABLE_MODELS:
        logger.error(f"Unknown model: {model_name}")
        return None

    model_info = AVAILABLE_MODELS[model_name]

    try:
        # Import the model module
        module = __import__(model_info['module'], fromlist=[model_info['function']])
        main_func = getattr(module, model_info['function'])

        # Capture stdout
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            main_func(ticker)

        output_text = output_buffer.getvalue()
        return output_text

    except Exception as e:
        logger.error(f"  Training failed: {e}")
        return None


def run_training(
    models: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
    dry_run: bool = False
):
    """Run training for specified models and tickers."""
    engine = get_database_engine()

    # Get available tickers if not specified
    if not tickers:
        tickers = get_available_tickers(engine)
        if not tickers:
            logger.warning("No tickers found in database. Add data first.")
            return

    # Use all models if not specified
    if not models:
        models = list(AVAILABLE_MODELS.keys())

    logger.info(f"Training {len(models)} models on {len(tickers)} tickers")
    logger.info(f"Models: {models}")
    logger.info(f"Tickers: {tickers[:5]}{'...' if len(tickers) > 5 else ''}")

    total = len(models) * len(tickers)
    success = 0
    failed = 0

    for i, ticker in enumerate(tickers):
        logger.info(f"\n[{i+1}/{len(tickers)}] Processing ticker: {ticker}")

        for model_name in models:
            logger.info(f"  Training {model_name}...")

            output = train_model(model_name, ticker)

            if output:
                if dry_run:
                    logger.info(f"  [DRY RUN] Would save result for {model_name}/{ticker}")
                    success += 1
                else:
                    if save_result(engine, ticker, model_name, output):
                        success += 1
                    else:
                        failed += 1
            else:
                failed += 1

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Training complete: {success} successful, {failed} failed")
    logger.info(f"{'=' * 50}")

    engine.dispose()


def list_models():
    """Print available models."""
    print("\nAvailable models:")
    print("-" * 50)
    for name, info in AVAILABLE_MODELS.items():
        print(f"  {name:<15} - {info['description']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train financial prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_models.py                     # Train all
  python scripts/train_models.py --model ridge       # Train only ridge
  python scripts/train_models.py --ticker BBG000Q7ZZY2  # Train on one ticker
  python scripts/train_models.py --dry-run           # Test without saving
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        action="append",
        help="Model to train (can be specified multiple times)"
    )
    parser.add_argument(
        "--ticker", "-t",
        type=str,
        action="append",
        help="Ticker to train on (can be specified multiple times)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run training but don't save to database"
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    run_training(
        models=args.model,
        tickers=args.ticker,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
