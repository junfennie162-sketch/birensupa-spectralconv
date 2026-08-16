#!/usr/bin/env bash
# 一键复现：编译 + 官方三案正确性 + 裁剪 DFT 非正式计时。
# 不跑 test_perf.py，不覆写 summary.json 里的正式 idle ms。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh"

echo "[1/2] build spectral_conv"
(cd "${ROOT}/spectral_conv" && ./build.sh)

echo "[2/2] unofficial pruned probe (official 3-case + warmup=10/iters=100; no formal ms)"
(cd "${ROOT}/spectral_conv" && python3 probe_pruned_continue.py)

echo "done. formal idle still frozen in results/summary.json"
echo "optional FNO: cd fno_ns && python3 test_forward.py"
echo "optional FNO chain: cd fno_ns && python3 test_chain_cpu_supa_consistency.py"
