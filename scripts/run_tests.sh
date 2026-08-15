#!/usr/bin/env bash
# 统一回归：按「改完什么 → 测什么」串行跑（禁止并发 GPU）
# `all` 不含 test_perf.py，避免覆写正式 idle ms。
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
  echo "WARN: writes summary.json formal idle; idle GPU only"
  (cd "${ROOT}/spectral_conv" && python3 test_perf.py)
}

run_fno() {
  echo "[test] fno_ns forward+viz      ← 文件: fno_ns/test_forward.py + visualize.py"
  (cd "${ROOT}/fno_ns" && python3 test_forward.py && python3 visualize.py)
}

run_fno_chain() {
  echo "[test] fno_ns chain consistency ← 文件: fno_ns/test_chain_cpu_supa_consistency.py"
  (cd "${ROOT}/fno_ns" && python3 test_chain_cpu_supa_consistency.py)
}

run_fno_batch16() {
  echo "[test] fno_ns batch16 perf     ← 文件: fno_ns/benchmark_fno_batch16.py"
  echo "WARN: writes summary.json fno_ns.perf_batch16"
  (cd "${ROOT}/fno_ns" && python3 benchmark_fno_batch16.py)
}

run_spectral_tune() {
  echo "[test] spectral_conv tune      ← 文件: spectral_conv/tune.py --quick"
  (cd "${ROOT}/spectral_conv" && python3 tune.py --quick)
}

run_spectral_sufft() {
  echo "[test] spectral_conv suFFT   ← 文件: spectral_conv/test_sufft_accuracy.py"
  (cd "${ROOT}/spectral_conv" && python3 test_sufft_accuracy.py)
}

run_spectral_backward() {
  echo "[test] spectral_conv backward ← 文件: spectral_conv/test_backward.py"
  (cd "${ROOT}/spectral_conv" && python3 test_backward.py)
}

run_spectral_3d() {
  echo "[test] spectral_conv 3d      ← 文件: spectral_conv/test_3d_accuracy.py"
  (cd "${ROOT}/spectral_conv" && python3 test_3d_accuracy.py)
}

run_spectral_irregular() {
  echo "[test] spectral_conv irregular ← 文件: spectral_conv/test_irregular_shapes.py"
  (cd "${ROOT}/spectral_conv" && python3 test_irregular_shapes.py)
}

run_reproduce() {
  echo "[test] reproduce pruned (no formal ms)"
  "${ROOT}/scripts/reproduce.sh"
}

case "${MODE}" in
  accuracy|acc)
    run_spectral_accuracy
    ;;
  sufft)
    run_spectral_sufft
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
  irregular)
    run_spectral_irregular
    ;;
  fno)
    run_fno
    ;;
  fno-chain|chain)
    run_fno_chain
    ;;
  fno-batch16|batch16)
    run_fno_batch16
    ;;
  tune)
    run_spectral_tune
    ;;
  reproduce)
    run_reproduce
    ;;
  all)
    run_spectral_accuracy
    run_spectral_sufft
    run_spectral_backward
    run_spectral_3d
    run_spectral_irregular
    run_fno_chain
    ;;
  *)
    echo "usage: $0 [all|accuracy|sufft|perf|backward|3d|irregular|fno|fno-chain|fno-batch16|tune|reproduce]"
    exit 2
    ;;
esac

echo "[test] OK mode=${MODE}"
echo "results: ${ROOT}/results/summary.json  |  ${ROOT}/results.md"
