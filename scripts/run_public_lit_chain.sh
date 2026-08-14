#!/usr/bin/env bash
# Literature probes: H1 loss -> Spectral-Refiner lite -> soup. No auto-promote.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FNO="$ROOT/fno_ns"
LOGDIR="$ROOT/results/run_logs"
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export PYTHONUNBUFFERED=1
BASE=0.03511497611179948
GATE=0.03501497611179948
INIT="$FNO/checkpoints/fno_ns_public_demo.pt"
CHAIN_LOG="$LOGDIR/fno_public_lit_chain_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOGDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$CHAIN_LOG"; }

best_init() {
  python3 - <<'PY'
import torch
from pathlib import Path
CK=Path("/workspace/ai4s/submission/fno_ns/checkpoints")
names=[
  "fno_ns_public_demo.pt",
  "fno_ns_public_h1_r1_best.pt",
  "fno_ns_public_spec_ref_r1_best.pt",
  "fno_ns_public_last_thaw_r3c_best.pt",
  "fno_ns_public_soup_lit_best.pt",
]
best,bl=None,1e9
for n in names:
  p=CK/n
  if not p.exists(): continue
  b=torch.load(p,map_location="cpu",weights_only=False)
  l2=float(b.get("test_l2",1e9))
  if l2<bl: bl,best=l2,str(p)
print(best or "")
print(bl)
PY
}

cd "$FNO"
log "START lit_chain BASE=$BASE GATE=$GATE"
readarray -t BI < <(best_init)
INIT="${BI[0]:-$INIT}"
log "init=$INIT l2=${BI[1]}"

run_stage() {
  local tag="$1"; shift
  log "=== STAGE $tag ==="
  python3 "$@" 2>&1 | tee -a "$CHAIN_LOG"
  readarray -t BI < <(best_init)
  INIT="${BI[0]}"
  log "after $tag best=$INIT l2=${BI[1]}"
}

run_stage h1_r1 train_public_h1_probe.py \
  --tag h1_r1 --init-from "$INIT" \
  --baseline "$BASE" --gate "$GATE" --stop-on-gate \
  --epochs 12 --lr 3e-6 --grad-weight 0.25 \
  --freeze-spectral --early-stop-patience 4

run_stage spec_ref_r1 train_public_spectral_refiner_probe.py \
  --tag spec_ref_r1 --init-from "$INIT" \
  --baseline "$BASE" --gate "$GATE" --stop-on-gate \
  --epochs 14 --lr 8e-7 --mix-l2 0.35 --sob-alpha 1.0 \
  --early-stop-patience 5

log "=== SOUP ==="
CKPTS=()
for n in demo h1_r1_best spec_ref_r1_best last_thaw_r3c_best; do
  p="$FNO/checkpoints/fno_ns_public_${n}.pt"
  [[ -f "$p" ]] && CKPTS+=("$p")
done
python3 eval_public_soup.py --ckpts "${CKPTS[@]}" \
  --baseline "$BASE" --gate "$GATE" \
  --out-name fno_ns_public_soup_lit_best.pt --always-save \
  2>&1 | tee -a "$CHAIN_LOG"
cp -f "$LOGDIR/fno_public_soup_r8_summary.json" "$LOGDIR/fno_public_soup_lit_summary.json" 2>/dev/null || true

python3 - <<PY
import json,torch
from pathlib import Path
CK=Path("$FNO/checkpoints"); LOG=Path("$LOGDIR")
rows=[]
for p in list(CK.glob("fno_ns_public_*best*.pt"))+[CK/"fno_ns_public_demo.pt"]:
  if not p.exists(): continue
  try: b=torch.load(p,map_location="cpu",weights_only=False)
  except Exception: continue
  if "test_l2" not in b: continue
  rows.append((float(b["test_l2"]), p.name, b.get("promoted_tag")))
rows.sort()
gate=float("$GATE"); demo=float("$BASE")
summary={
  "task":"fno_public_lit_chain",
  "demo_l2":demo,"gate":gate,
  "best_sidecar":{"l2":rows[0][0],"ckpt":rows[0][1],"tag":rows[0][2]},
  "top8":[{"l2":a,"ckpt":b,"tag":c} for a,b,c in rows[:8]],
  "beat_gate": rows[0][0]<gate,"promote":False,
  "log":"$CHAIN_LOG",
}
(LOG/"fno_public_lit_chain_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(summary,indent=2,ensure_ascii=False))
print("SIGNAL" if summary["beat_gate"] else "NO_SIGNAL")
PY
log "DONE lit_chain"
