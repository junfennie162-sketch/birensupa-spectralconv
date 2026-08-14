#!/usr/bin/env bash
# 一键：公开 NS64 上 FNO 自动训练链（可断网后台跑）
#
# 说明：
#   - 公开 NS64 只用于进阶题 FNO-NS，不必选 SpectralConv 不需要这份数据。
#   - 顺序：scratch 100ep → continue 50ep → 终评 / 写披露 / 合入 ai4s
#   - 若已有 train_public_ns64.py 在跑，会先等它结束再续跑。
#
# 用法：
#   cd /workspace/ai4s-f/submission
#   ./scripts/run_public_ns64_autochain.sh
#   # 或指定先等某个 PID：
#   ./scripts/run_public_ns64_autochain.sh --wait-pid 694760

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FNO="$ROOT/fno_ns"
LOG_DIR="$ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CONSOLE="$LOG_DIR/fno_public_ns64_autochain_console_${STAMP}.log"
PIDFILE="$LOG_DIR/fno_public_ns64_autochain.pid"

# 避免重复拉起多条链
if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "autochain already running pid=$old log=$LOG_DIR/fno_public_ns64_chain.log"
    exit 0
  fi
fi

cd "$FNO"
nohup python3 -u run_public_ns64_chain.py "$@" >"$CONSOLE" 2>&1 &
echo $! | tee "$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "console=$CONSOLE"
echo "chain_log=$LOG_DIR/fno_public_ns64_chain.log"
echo "state=$LOG_DIR/fno_public_ns64_chain_state.json"
