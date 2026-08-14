#!/usr/bin/env bash
# 统一回归：按「改完什么 → 测什么」串行跑（禁止并发 GPU）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh"

MODE="${1:-all}"

run_spectral_accuracy() {
  echo "[test] spectral_conv accuracy  ← 文件: spectral_conv/test_accuracy.py"
  (cd "${ROOT}/spectral_conv" && ./build.sh && python3 test_accuracy.py)
}

run_spectral_perf() {
  echo "[test] spectral_conv perf      ← 文件: spectral_conv/test_perf.py"
  (cd "${ROOT}/spectral_conv" && python3 test_perf.py)
}

run_fno() {
  echo "[test] fno_ns forward+viz      ← 文件: fno_ns/test_forward.py + visualize.py"
  (cd "${ROOT}/fno_ns" && python3 test_forward.py && python3 visualize.py)
}

run_spectral_sufft() {
  echo "[test] spectral_conv suFFT   ← 文件: spectral_conv/test_sufft_accuracy.py"
  (cd "${ROOT}/spectral_conv" && ./build.sh && python3 test_sufft_accuracy.py)
}

run_spectral_sufft_perf() {
  echo "[test] spectral_conv suFFT perf ← 文件: spectral_conv/test_sufft_perf.py"
  (cd "${ROOT}/spectral_conv" && python3 test_sufft_perf.py)
}

run_spectral_backward() {
  echo "[test] spectral_conv backward ← 文件: spectral_conv/test_backward.py"
  (cd "${ROOT}/spectral_conv" && python3 test_backward.py)
}

run_spectral_3d() {
  echo "[test] spectral_conv 3d      ← 文件: spectral_conv/test_3d_accuracy.py"
  (cd "${ROOT}/spectral_conv" && python3 test_3d_accuracy.py)
}

case "${MODE}" in
  accuracy|acc)
    run_spectral_accuracy
    ;;
  sufft)
    run_spectral_sufft
    ;;
  sufft-perf)
    run_spectral_sufft_perf
    ;;
  perf)
    run_spectral_perf
    ;;
  backward)
    run_spectral_backward
    ;;
  3d)
    run_spectral_3d
    ;;
  fno)
    run_fno
    ;;
  all)
    run_spectral_accuracy
    run_spectral_sufft
    run_spectral_perf
    run_spectral_sufft_perf
    run_spectral_backward
    run_spectral_3d
    run_fno
    ;;
  *)
    echo "usage: $0 [all|accuracy|sufft|sufft-perf|perf|backward|3d|fno]"
    exit 2
    ;;
esac

echo "[test] OK mode=${MODE}"
echo "results: ${ROOT}/results/summary.json  |  ${ROOT}/results.md"
