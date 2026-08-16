#!/usr/bin/env bash
# 整理并打提交包（不删源文件）
# 用法：cd submission && ./scripts/pack_submission.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT/results/archives"
STAGE="$OUT_DIR/fandougarden_submit_${STAMP}"
TGZ="$OUT_DIR/fandougarden_submit_${STAMP}.tar.gz"
AI4S="${AI4S_ROOT:-/workspace/ai4s/submission}"

mkdir -p "$STAGE" "$OUT_DIR"

echo "[pack] staging -> $STAGE"
rsync -a \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.cache/' \
  --exclude 'results/archives/' \
  --exclude 'fno_ns/data/.cache/' \
  --exclude '*.incomplete' \
  --exclude '*.lock' \
  "$ROOT/" "$STAGE/"

# 包内简短提交说明
cat > "$STAGE/SUBMIT_README.md" <<EOF
# 翻斗花园 · 提交包 ${STAMP}

## 正式指标
- SpectralConv idle：0.599 / 1.405 / 5.099 ms @64/128/256（pinned_src_r1 · v13）；worst rel ≈7.16e-6
- FNO 公开 NS64（1000/128）：relative L2 = **0.035012**（\`spec_ref_r2\` · **v10**）
  - 上一正式 v9：0.035115（\`dualview_r2\`）
  - data: \`fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt\`
  - ckpt: \`fno_ns/checkpoints/fno_ns_public_demo.pt\`
  - 机制：Spectral-Refiner lite（仅训 spectral_conv + Sobolev H⁻¹ 频域损失；见 \`train_public_spectral_refiner_probe.py\`）

## 工程对照（非公开分）
- 自建 v2 continue3 L2≈0.005144（\`fno_ns_demo.pt\`）
- 完整归档另见工作区 \`results/archives/fno_v2_continue3_pre_public_20260731.tar.gz\`（本提交包为减体积未内嵌）

## 快速复现
\`\`\`bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd spectral_conv && ./build.sh && python3 test_accuracy.py && python3 test_perf.py
cd ../fno_ns && python3 - <<'PY'
import torch
from torch.utils.data import DataLoader
from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d
from test_forward import relative_l2
data, src = load_or_build_ns_like(n_samples=1128, resolution=64, n_times=20, seed=20260722)
assert src.startswith('file:navier_stokes'), src
_, te = split_train_test(data, 1000, 128, seed=20260722)
m = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
m.load_state_dict(torch.load('checkpoints/fno_ns_public_demo.pt', map_location='cpu', weights_only=False)['model'])
m.eval()
scores=[]
with torch.no_grad():
  for x,y in DataLoader(SequenceVorticityDataset(te,10,1), batch_size=16):
    scores.append(relative_l2(m(x, use_supa=False), y))
print(src, sum(scores)/len(scores))
PY
\`\`\`

对照清单：\`SUBMISSION_CHECKLIST.md\`
EOF

echo "[pack] tar -> $TGZ"
tar -czf "$TGZ" -C "$OUT_DIR" "$(basename "$STAGE")"
sha256sum "$TGZ" | tee "${TGZ}.sha256"

if [[ -d "$AI4S" ]]; then
  echo "[pack] sync -> $AI4S"
  rsync -a \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.cache/' \
    --exclude 'results/archives/' \
    --exclude 'fno_ns/data/.cache/' \
    "$ROOT/" "$AI4S/"
  # 提交包副本放到 ai4s/results/archives
  mkdir -p "$AI4S/results/archives"
  cp -a "$TGZ" "${TGZ}.sha256" "$AI4S/results/archives/"
  cp -a "$STAGE/SUBMIT_README.md" "$AI4S/results/archives/SUBMIT_README_${STAMP}.md"
fi

echo "DONE"
echo "TGZ=$TGZ"
ls -lah "$TGZ"
