#!/usr/bin/env bash
# One-command BIREN check: env, build, official accuracy, unofficial pruned probe.
# Does NOT run test_perf.py and does NOT overwrite formal idle ms.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh"

echo "[validate] build SpectralConv"
(cd "${ROOT}/spectral_conv" && ./build.sh)

echo "[validate] official 3-case accuracy (writes accuracy fields; keeps frozen idle)"
(cd "${ROOT}/spectral_conv" && python3 test_accuracy.py)

echo "[validate] unofficial pruned probe (warmup=10, iters=100; not formal idle)"
(cd "${ROOT}/spectral_conv" && python3 probe_pruned_continue.py)

echo "[validate] OPT-loop dry-run (materials / protocol gates)"
(cd "${ROOT}" && python3 skills/operator_opt_loop/run_loop.py --dry-run --strict)

echo
echo "validate done."
echo "  formal idle remains frozen in results/summary.json"
echo "  optional FNO (needs official .pt): cd fno_ns && python3 render_official_demo.py"
echo "  optional formal idle (idle GPU only): cd spectral_conv && python3 test_perf.py"
