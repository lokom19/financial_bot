.PHONY: all setup install db-setup db-check server train train-model help clean docker-build docker-run docker-stop logs fetch-tickers fetch-candles fetch-data fetch-test data-status

# Default target
all: help

# ============================================================
# SETUP
# ============================================================

# Create required directories
setup:
	mkdir -p dataframes output templates static scripts

# Install Python dependencies
install:
	pip install -r requirements.txt

# Full setup: directories + dependencies + database
init: setup install db-setup
	@echo "✓ Project initialized successfully!"

# ============================================================
# DATABASE
# ============================================================

# Setup database tables
db-setup:
	python scripts/setup_db.py

# Check database connection
db-check:
	python scripts/setup_db.py --check

# ============================================================
# SERVER
# ============================================================

# Run FastAPI server (development mode with auto-reload)
server:
	uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Run FastAPI server (production mode)
server-prod:
	uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Health check
health:
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Server not running"

# ============================================================
# DATA COLLECTION (Tinkoff API)
# ============================================================

# Fetch all tickers from Tinkoff API and save to database
fetch-tickers:
	@echo "Fetching tickers from Tinkoff API..."
	python all_figi_to_db.py --test
	@echo "✓ Tickers saved to public.tickers"

# Fetch historical candles for all tickers (requires fetch-tickers first)
fetch-candles:
	@echo "Fetching historical candles from Tinkoff API..."
	@echo "This may take a while depending on the number of tickers..."
	python all_dfs_to_db.py
	@echo "✓ Candles saved to all_dfs schema"

# Full data collection: tickers + candles
fetch-data: fetch-tickers fetch-candles
	@echo "✓ Data collection completed!"

# Test mode: only SBER, YNDX, VTBR, TCSG, OZON
fetch-test:
	@echo "Fetching test tickers (SBER, YNDX, VTBR, TCSG, OZON)..."
	python all_figi_to_db.py --test
	@echo "Fetching candles for test tickers..."
	python all_dfs_to_db.py
	@echo "✓ Test data collection completed!"

# Check data status
data-status:
	@echo "=== Tickers count ==="
	@python -c "from sqlalchemy import create_engine, text; import os; from dotenv import load_dotenv; load_dotenv(); e=create_engine(f\"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}\"); print(e.connect().execute(text('SELECT COUNT(*) FROM public.tickers')).scalar(), 'tickers')" 2>/dev/null || echo "No tickers table"
	@echo ""
	@echo "=== Data tables count ==="
	@python -c "from sqlalchemy import create_engine, text; import os; from dotenv import load_dotenv; load_dotenv(); e=create_engine(f\"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}\"); print(e.connect().execute(text(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='all_dfs'\")).scalar(), 'data tables')" 2>/dev/null || echo "No data tables"

# ============================================================
# TRAINING
# ============================================================

# Train all models on all available tickers
train:
	python scripts/train_models.py

# Train specific model (usage: make train-model MODEL=ridge)
train-model:
	python scripts/train_models.py --model $(MODEL)

# Train on specific ticker (usage: make train-ticker TICKER=BBG000Q7ZZY2)
train-ticker:
	python scripts/train_models.py --ticker $(TICKER)

# List available models
list-models:
	python scripts/train_models.py --list-models

# Dry run (test training without saving)
train-dry:
	python scripts/train_models.py --dry-run

# ============================================================
# DOCKER
# ============================================================

# Build Docker images
docker-build: setup
	docker-compose build

# Run with Docker
docker-run:
	docker-compose up -d

# Stop Docker containers
docker-stop:
	docker-compose down

# View Docker logs
logs:
	docker-compose logs -f app

# ============================================================
# UTILITIES
# ============================================================

# Clean generated files
clean:
	rm -rf output/*.txt output/*.json
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache

# Full clean (including Docker)
clean-all: clean docker-stop
	docker-compose down -v
	docker system prune -f

# View data directory
list-data:
	@ls -la dataframes/ 2>/dev/null || echo "No dataframes directory"

# View output directory
list-output:
	@ls -la output/ 2>/dev/null || echo "No output directory"

# Check project status
status:
	@echo "=== Database ==="
	@make db-check 2>/dev/null || echo "DB not configured"
	@echo ""
	@echo "=== Server ==="
	@make health 2>/dev/null
	@echo ""
	@echo "=== Docker ==="
	@docker-compose ps 2>/dev/null || echo "Docker not running"

# ============================================================
# HELP
# ============================================================

help:
	@echo "Financial Prediction Models - Available commands:"
	@echo ""
	@echo "  SETUP:"
	@echo "    make init          - Full project initialization"
	@echo "    make install       - Install Python dependencies"
	@echo "    make db-setup      - Create database tables"
	@echo ""
	@echo "  DATA COLLECTION:"
	@echo "    make fetch-data    - Full data collection (tickers + candles)"
	@echo "    make fetch-tickers - Fetch tickers from Tinkoff API"
	@echo "    make fetch-candles - Fetch historical candles"
	@echo "    make data-status   - Check data collection status"
	@echo ""
	@echo "  SERVER:"
	@echo "    make server        - Run development server (port 8000)"
	@echo "    make server-prod   - Run production server"
	@echo "    make health        - Check server health"
	@echo ""
	@echo "  TRAINING:"
	@echo "    make train              - Train all models"
	@echo "    make train-model MODEL=ridge  - Train specific model"
	@echo "    make list-models        - List available models"
	@echo ""
	@echo "  DOCKER:"
	@echo "    make docker-build  - Build Docker images"
	@echo "    make docker-run    - Start containers"
	@echo "    make docker-stop   - Stop containers"
	@echo "    make logs          - View container logs"
	@echo ""
	@echo "  UTILITIES:"
	@echo "    make status        - Check project status"
	@echo "    make clean         - Clean temporary files"
	@echo ""
	@echo "Documentation: http://localhost:8000/docs (when server is running)"
