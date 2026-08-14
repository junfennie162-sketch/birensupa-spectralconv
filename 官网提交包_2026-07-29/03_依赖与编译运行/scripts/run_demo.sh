#!/usr/bin/env bash
# 演示入口：同步 media、校验 demo 资产（默认不重跑长测，避免占 GPU）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh"

MEDIA="${ROOT}/demo/media"
mkdir -p "${MEDIA}"

FIGURE="$(ls -1t "${ROOT}/results/figures"/fno_ns_pred_vs_gt_*.png 2>/dev/null | head -1 || true)"
if [[ -n "${FIGURE}" ]]; then
  cp -f "${FIGURE}" "${MEDIA}/"
  echo "[demo] copied figure: ${FIGURE}"
else
  echo "[demo] WARN: no FNO figure under results/figures/"
fi

if command -v brsmi >/dev/null 2>&1; then
  brsmi > "${MEDIA}/brsmi_snapshot.txt" || true
fi

if [[ -f "${ROOT}/results/summary.json" ]]; then
  python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
root = Path("${ROOT}")
summary = json.loads((root / "results/summary.json").read_text())
sc = summary.get("spectral_conv", {})
fno = summary.get("fno_ns", {})
perf = (sc.get("perf") or {}).get("rows") or []
lines = [
    "# Run snapshot (ai4s-f)",
    "",
    f"- captured_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
    f"- gpu: {summary.get('env', {}).get('gpu')}",
    "",
    "## SpectralConv",
    f"- worst_rel_error: {sc.get('rel_error')}",
    f"- status: {sc.get('status')}",
    "",
    "## Perf",
]
for row in perf:
    lines.append(f"- {row.get('resolution')}: {row.get('forward_time_ms')} ms, {row.get('memory_MB')} MB")
lines += [
    "",
    "## FNO-NS",
    f"- fourier_layers: {fno.get('fourier_layers')}",
    f"- rel_l2: {fno.get('rel_l2')}",
    f"- data: {fno.get('data')}",
    "",
]
(root / "demo/media/metrics_snapshot.md").write_text("\n".join(lines))
print("[demo] wrote metrics_snapshot.md")
PY
fi

echo "[demo] asset gate"
"${ROOT}/scripts/maintain_assets.sh" check demo
"${ROOT}/scripts/maintain_assets.sh" next

echo "SCP text: ${ROOT}/demo/scp_description.md"
echo "media: ${MEDIA}"
echo "To finish phase: ./scripts/maintain_assets.sh mark-done demo"
