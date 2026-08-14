#!/usr/bin/env bash
# 一键：针对公开 NS64「更难」特性的方法增强链（可断网后台）
# A 高频损失+周期增广 → B 残差学习 → C 低 lr 续训
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CONSOLE="$LOG_DIR/fno_public_boost_autochain_console_${STAMP}.log"
PIDFILE="$LOG_DIR/fno_public_boost_autochain.pid"

if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "boost chain already running pid=$old"
    exit 0
  fi
fi

cd "$ROOT/fno_ns"
nohup python3 -u run_public_boost_chain.py "$@" >"$CONSOLE" 2>&1 &
echo $! | tee "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "console=$CONSOLE"
echo "chain_log=$LOG_DIR/fno_public_boost_chain.log"
