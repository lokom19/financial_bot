#!/usr/bin/env python3
"""
Scheduler for automated data collection and model training.

Runs the pipeline (data collection + training) at specified intervals.
Default interval: 3 minutes.

Usage:
    python scripts/scheduler.py
    python scripts/scheduler.py --interval 5  # Run every 5 minutes
    python scripts/scheduler.py --once        # Run once and exit
"""
import os
import sys
import time
import logging
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.interval import IntervalTrigger
except ImportError:
    print("APScheduler not installed. Run: pip install apscheduler")
    sys.exit(1)

# Load environment
load_dotenv()

# Configure logging
# In Docker, prefer stdout only (LOG_TO_FILE=false)
log_handlers = [logging.StreamHandler()]
if os.getenv('LOG_TO_FILE', 'true').lower() == 'true':
    log_handlers.append(logging.FileHandler('scheduler.log'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class PipelineRunner:
    """Runs data collection and training pipeline."""

    def __init__(self, days: int = 30, interval: str = "day", skip_data_collection: bool = False):
        self.days = days
        self.interval = interval
        self.skip_data_collection = skip_data_collection
        self.is_running = False
        self.last_run = None
        self.run_count = 0
        self.error_count = 0

    def run_command(self, command: list, description: str) -> bool:
        """Run a shell command and return success status."""
        logger.info(f"Starting: {description}")
        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"✓ {description} completed in {elapsed:.1f}s")
                return True
            else:
                logger.error(f"✗ {description} failed (code {result.returncode})")
                logger.error(f"stderr: {result.stderr[:500]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ {description} timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"✗ {description} error: {e}")
            return False

    def run_data_collection(self) -> bool:
        """Run data collection from Tinkoff API."""
        return self.run_command(
            ["python", "all_dfs_to_db.py", "--days", str(self.days), "--interval", self.interval],
            f"Data collection ({self.days} days, {self.interval})"
        )

    def run_training(self) -> bool:
        """Run model training."""
        return self.run_command(
            ["python", "scripts/train_models.py"],
            "Model training"
        )

    def run_pipeline(self):
        """Run the full pipeline: data collection → training."""
        if self.is_running:
            logger.warning("Pipeline is already running, skipping this run")
            return

        self.is_running = True
        self.run_count += 1
        run_start = datetime.now()

        logger.info("=" * 60)
        logger.info(f"Pipeline run #{self.run_count} started at {run_start}")
        logger.info("=" * 60)

        try:
            # Step 1: Data collection (skip if flag is set)
            if self.skip_data_collection:
                logger.info("Skipping data collection (--skip-data flag)")
            else:
                if not self.run_data_collection():
                    logger.error("Data collection failed, skipping training")
                    self.error_count += 1
                    return

            # Step 2: Training
            if not self.run_training():
                logger.error("Training failed")
                self.error_count += 1
                return

            self.last_run = datetime.now()
            elapsed = (self.last_run - run_start).total_seconds()

            logger.info("=" * 60)
            logger.info(f"✓ Pipeline completed successfully in {elapsed:.1f}s")
            logger.info(f"  Total runs: {self.run_count}, Errors: {self.error_count}")
            logger.info("=" * 60)

        finally:
            self.is_running = False


def main():
    parser = argparse.ArgumentParser(
        description="Scheduler for automated data collection and training"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=3,
        help="Interval in minutes between runs (default: 3)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Days of historical data to fetch (default: 30)"
    )
    parser.add_argument(
        "--candle-interval",
        type=str,
        default="day",
        choices=["1min", "5min", "15min", "hour", "day", "week", "month"],
        help="Candle interval (default: day)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run pipeline once and exit"
    )
    parser.add_argument(
        "--no-initial",
        action="store_true",
        help="Don't run pipeline immediately on start"
    )

    args = parser.parse_args()

    # Create pipeline runner
    runner = PipelineRunner(days=args.days, interval=args.candle_interval)

    # Single run mode
    if args.once:
        logger.info("Running pipeline once...")
        runner.run_pipeline()
        return

    # Scheduler mode
    logger.info("=" * 60)
    logger.info("Pipeline Scheduler Started")
    logger.info(f"  Interval: every {args.interval} minutes")
    logger.info(f"  Data: {args.days} days, {args.candle_interval} candles")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")

    scheduler = BlockingScheduler()

    # Schedule recurring job
    scheduler.add_job(
        runner.run_pipeline,
        trigger=IntervalTrigger(minutes=args.interval),
        id='pipeline',
        name='Data Collection + Training Pipeline',
        max_instances=1,
        coalesce=True
    )

    # Run immediately on start (unless --no-initial)
    if not args.no_initial:
        logger.info("Running initial pipeline...")
        runner.run_pipeline()

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("\nScheduler stopped by user")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
