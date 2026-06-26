#!/usr/bin/env bash
# Запускает все сервисы локально:
#   - FastAPI (порт 8002)
#   - Streamlit dashboard (порт 8501)
#   - Scheduler (опционально, через --with-scheduler)
#
# Использование:
#   ./scripts/run_all.sh              # FastAPI + Streamlit
#   ./scripts/run_all.sh --with-scheduler  # + автообучение
#   ./scripts/run_all.sh --stop       # остановить всё

set -e
cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/.pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

PYTHON="$PROJECT_ROOT/.venv_310/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python"
fi

stop_service() {
    local name="$1"
    local pidfile="$PID_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "  ✓ $name (pid $pid) остановлен"
        else
            echo "  - $name не запущен (stale pid $pid)"
        fi
        rm -f "$pidfile"
    else
        echo "  - $name: pid-файл не найден"
    fi
}

start_service() {
    local name="$1"
    local cmd="$2"
    local port="$3"
    local pidfile="$PID_DIR/$name.pid"
    local logfile="$LOG_DIR/$name.log"

    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "  ! $name уже запущен (pid $(cat "$pidfile"))"
        return
    fi

    echo "  ▶ Запуск $name (порт $port)..."
    nohup bash -c "$cmd" > "$logfile" 2>&1 &
    echo $! > "$pidfile"
    sleep 0.5
    echo "    pid=$(cat "$pidfile"), лог: $logfile"
}

if [ "${1:-}" = "--stop" ]; then
    echo "Остановка сервисов..."
    stop_service fastapi
    stop_service streamlit
    stop_service scheduler
    stop_service nightly
    echo "Готово."
    exit 0
fi

if [ "${1:-}" = "--status" ]; then
    for s in fastapi streamlit scheduler nightly; do
        pidfile="$PID_DIR/$s.pid"
        if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            echo "  ✓ $s (pid $(cat "$pidfile"))"
        else
            echo "  ✗ $s не запущен"
        fi
    done
    exit 0
fi

echo "============================================"
echo "  Запуск Trading Signals (все сервисы)"
echo "============================================"

start_service fastapi \
    "$PYTHON -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload" \
    8002

start_service streamlit \
    "$PYTHON -m streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true" \
    8501

if [ "${1:-}" = "--with-scheduler" ]; then
    start_service scheduler \
        "$PYTHON scripts/scheduler.py --interval 60 --days 30" \
        "(scheduler)"
fi

if [ "${1:-}" = "--with-nightly" ]; then
    start_service nightly \
        "$PYTHON scripts/scheduler.py --nightly --no-initial" \
        "(nightly cron)"
fi

echo ""
echo "============================================"
echo "  Сервисы запущены:"
echo "============================================"
echo "  FastAPI:    http://localhost:8002"
echo "  Swagger UI: http://localhost:8002/docs"
echo "  Streamlit:  http://localhost:8501"
echo ""
echo "  Логи:       tail -f $LOG_DIR/*.log"
echo "  Статус:     ./scripts/run_all.sh --status"
echo "  Остановка:  ./scripts/run_all.sh --stop"
echo "============================================"
