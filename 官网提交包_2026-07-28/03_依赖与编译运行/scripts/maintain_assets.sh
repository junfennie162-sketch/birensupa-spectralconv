#!/usr/bin/env bash
# Phase 完成后维护 submission 资产：校验清单 → 更新 phase_status → 提示必写文件
# 用法:
#   ./scripts/maintain_assets.sh status
#   ./scripts/maintain_assets.sh check <phase>
#   ./scripts/maintain_assets.sh mark-done <phase>
#   ./scripts/maintain_assets.sh next
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
HELPER="${ROOT}/scripts/_maintain_assets.py"

if [[ ! -f "${HELPER}" ]]; then
  echo "ERROR: missing ${HELPER}" >&2
  exit 1
fi

exec python3 "${HELPER}" --repo-root "${REPO_ROOT}" --submission-root "${ROOT}" "$@"
