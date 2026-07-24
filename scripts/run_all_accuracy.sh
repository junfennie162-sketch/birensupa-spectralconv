#!/usr/bin/env bash
# 串行跑必选 + 进阶正确性；结束后跑资产维护检查（不自动 mark-done）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh"

echo "[1/2] spectral_conv accuracy"
if [[ -f "${ROOT}/spectral_conv/test_accuracy.py" ]]; then
  (cd "${ROOT}/spectral_conv" && ./build.sh && python3 test_accuracy.py)
else
  echo "SKIP: spectral_conv not implemented yet"
fi

echo "[2/2] fno_ns forward"
if [[ -f "${ROOT}/fno_ns/test_forward.py" ]]; then
  (cd "${ROOT}/fno_ns" && python3 test_forward.py)
else
  echo "SKIP: fno_ns not implemented yet"
fi

echo "[assets] phase status / next checklist"
"${ROOT}/scripts/maintain_assets.sh" status
"${ROOT}/scripts/maintain_assets.sh" next

echo "done"
echo "tip: after metrics are written, run: ./scripts/maintain_assets.sh mark-done <phase>"
