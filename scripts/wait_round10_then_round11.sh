#!/usr/bin/env bash
# 等 ROUND10 结束自动开 ROUND11
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
PIDFILE10="$LOG_DIR/fno_public_round10_autochain.pid"
PIDFILE11="$LOG_DIR/fno_public_round11_autochain.pid"
WAITLOG="$LOG_DIR/wait_round10_then_round11.log"

if [[ -f "$PIDFILE11" ]]; then
  old="$(cat "$PIDFILE11" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "round11 already running pid=$old"
    exit 0
  fi
fi

R10_PID="${1:-}"
if [[ -z "$R10_PID" && -f "$PIDFILE10" ]]; then
  R10_PID="$(cat "$PIDFILE10")"
fi
if [[ -z "$R10_PID" ]] || ! kill -0 "$R10_PID" 2>/dev/null; then
  R10_PID="$(pgrep -f 'python3 -u run_public_round10_chain.py' | head -1 || true)"
fi

{
  echo "[$(date -Is)] wait round10 pid=${R10_PID:-none}"
  if [[ -n "${R10_PID}" ]] && kill -0 "$R10_PID" 2>/dev/null; then
    while kill -0 "$R10_PID" 2>/dev/null; do
      sleep 60
      echo "[$(date -Is)] still waiting round10 pid=$R10_PID"
    done
    echo "[$(date -Is)] round10 exited"
  else
    echo "[$(date -Is)] round10 not running; start round11 now"
  fi
  cd "$ROOT/fno_ns"
  nohup python3 -u run_public_round11_chain.py \
    >"$LOG_DIR/fno_public_round11_console_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  echo $! | tee "$PIDFILE11"
  echo "[$(date -Is)] started round11 pid=$(cat "$PIDFILE11")"
} >>"$WAITLOG" 2>&1 &

echo $! >"$LOG_DIR/wait_round10_then_round11.pid"
echo "waiter=$(cat "$LOG_DIR/wait_round10_then_round11.pid") round10=${R10_PID:-none}"
echo "log=$WAITLOG"
