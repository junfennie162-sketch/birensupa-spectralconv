#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CONSOLE="$LOG_DIR/fno_public_round10_console_${STAMP}.log"
PIDFILE="$LOG_DIR/fno_public_round10_autochain.pid"

if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "round10 already running pid=$old"
    exit 0
  fi
fi

cd "$ROOT/fno_ns"
nohup python3 -u run_public_round10_chain.py "$@" >"$CONSOLE" 2>&1 &
echo $! | tee "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "console=$CONSOLE"
echo "log=$LOG_DIR/fno_public_round10_chain.log"
