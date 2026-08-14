#!/usr/bin/env bash
# 持续挤压公开 NS64 L2（多轮 continue/freeze/unfreeze，平台停）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CONSOLE="$LOG_DIR/fno_public_squeeze_console_${STAMP}.log"
PIDFILE="$LOG_DIR/fno_public_squeeze_autochain.pid"

if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "squeeze loop already running pid=$old"
    exit 0
  fi
fi

cd "$ROOT/fno_ns"
nohup python3 -u run_public_squeeze_loop.py --max-rounds 4 "$@" >"$CONSOLE" 2>&1 &
echo $! | tee "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "console=$CONSOLE"
echo "log=$LOG_DIR/fno_public_squeeze_loop.log"
echo "state=$LOG_DIR/fno_public_squeeze_loop_state.json"
