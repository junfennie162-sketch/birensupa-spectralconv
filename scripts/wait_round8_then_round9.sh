#!/usr/bin/env bash
# 等 ROUND8 结束，自动开 ROUND9 继续优化
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
PIDFILE8="$LOG_DIR/fno_public_round8_autochain.pid"
PIDFILE9="$LOG_DIR/fno_public_round9_autochain.pid"
WAITLOG="$LOG_DIR/wait_round8_then_round9.log"

if [[ -f "$PIDFILE9" ]]; then
  old="$(cat "$PIDFILE9" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "round9 already running pid=$old"
    exit 0
  fi
fi

R8_PID="${1:-}"
if [[ -z "$R8_PID" && -f "$PIDFILE8" ]]; then
  R8_PID="$(cat "$PIDFILE8")"
fi
# also match by process name
if [[ -z "$R8_PID" ]] || ! kill -0 "$R8_PID" 2>/dev/null; then
  R8_PID="$(pgrep -f 'python3 -u run_public_round8_chain.py' | head -1 || true)"
fi

{
  echo "[$(date -Is)] wait round8 pid=${R8_PID:-none}"
  if [[ -n "${R8_PID}" ]] && kill -0 "$R8_PID" 2>/dev/null; then
    while kill -0 "$R8_PID" 2>/dev/null; do
      sleep 60
      echo "[$(date -Is)] still waiting round8 pid=$R8_PID"
    done
    echo "[$(date -Is)] round8 pid $R8_PID exited"
  else
    echo "[$(date -Is)] round8 not running; start round9 immediately"
  fi
  cd "$ROOT/fno_ns"
  nohup python3 -u run_public_round9_chain.py >"$LOG_DIR/fno_public_round9_console_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  echo $! | tee "$PIDFILE9"
  echo "[$(date -Is)] started round9 pid=$(cat "$PIDFILE9")"
} >>"$WAITLOG" 2>&1 &

echo $! >"$LOG_DIR/wait_round8_then_round9.pid"
echo "waiter_pid=$(cat "$LOG_DIR/wait_round8_then_round9.pid")"
echo "wait_log=$WAITLOG"
echo "round8_pid=${R8_PID:-none}"
