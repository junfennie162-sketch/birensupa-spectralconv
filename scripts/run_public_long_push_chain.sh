#!/usr/bin/env bash
# Long push chain: qt_oversample → last_thaw → dualview → soup
# Gate vs live demo 0.035223 − 1e-4. No auto-promote.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FNO="$ROOT/fno_ns"
LOGDIR="$ROOT/results/run_logs"
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export PYTHONUNBUFFERED=1
BASE=0.03522327123209834
GATE=0.03512327123209834
INIT="$FNO/checkpoints/fno_ns_public_pf_delta_r1_best.pt"
CHAIN_LOG="$LOGDIR/fno_public_long_push_chain_$(date +%Y%m%d_%H%M%S).log"
STATE="$LOGDIR/fno_public_long_push_state.json"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$CHAIN_LOG"; }

best_init() {
  python3 - <<'PY'
import torch
from pathlib import Path
CK=Path("/workspace/ai4s-f/submission/fno_ns/checkpoints")
cands=[
  "fno_ns_public_pf_delta_r1_best.pt",
  "fno_ns_public_qt_over_r1_best.pt",
  "fno_ns_public_last_thaw_r1_best.pt",
  "fno_ns_public_dualview_r1_best.pt",
  "fno_ns_public_soup_long_best.pt",
  "fno_ns_public_demo.pt",
]
best=None; best_l2=1e9
for n in cands:
  p=CK/n
  if not p.exists(): continue
  b=torch.load(p,map_location="cpu",weights_only=False)
  l2=float(b.get("test_l2",1e9))
  if l2<best_l2:
    best_l2=l2; best=str(p)
print(best if best else "")
print(best_l2)
PY
}

cd "$FNO"
log "START long_push gate=$GATE init0=$INIT"
readarray -t BI < <(best_init)
INIT="${BI[0]:-$INIT}"
log "live_best_init=${BI[0]} l2=${BI[1]}"

run_stage() {
  local tag="$1"; shift
  log "=== STAGE $tag ==="
  if ! python3 "$@" 2>&1 | tee -a "$CHAIN_LOG"; then
    log "STAGE $tag FAILED (continue)"
    return 1
  fi
  readarray -t BI < <(best_init)
  INIT="${BI[0]}"
  log "after $tag best_init=$INIT l2=${BI[1]}"
  python3 - <<PY
import json
from pathlib import Path
p=Path("$STATE")
d=json.loads(p.read_text()) if p.exists() else {"stages":[]}
d["stages"].append({"tag":"$tag","best_init":"$INIT","l2":float("${BI[1]}")})
d["gate"]=float("$GATE")
p.write_text(json.dumps(d,indent=2)+"\n")
PY
}

run_stage qt_over_r1 train_public_qt_oversample_probe.py \
  --tag qt_over_r1 --init-from "$INIT" \
  --baseline "$BASE" --gate "$GATE" --stop-on-gate \
  --epochs 12 --lr 3e-6 --qt-power 1.5 \
  --lambda-pf 0.6 --lambda-delta 0.4 --hf-weight 0.15 \
  --freeze-spectral --early-stop-patience 4

run_stage last_thaw_r1 train_public_last_thaw_probe.py \
  --tag last_thaw_r1 --init-from "$INIT" \
  --baseline "$BASE" --gate "$GATE" --stop-on-gate \
  --epochs 10 --lr 2e-6 --spectral-lr 5e-7 --thaw-k 1 \
  --lambda-delta 0.45 --hf-weight 0.12 --early-stop-patience 4

run_stage dualview_r1 train_public_dualview_probe.py \
  --tag dualview_r1 --init-from "$INIT" \
  --baseline "$BASE" --gate "$GATE" --stop-on-gate \
  --epochs 10 --lr 3e-6 --lambda-cons 0.25 --lambda-delta 0.35 \
  --hf-weight 0.12 --freeze-spectral --early-stop-patience 4

log "=== SOUP ==="
CKPTS=()
for n in demo pf_delta_r1_best qt_over_r1_best last_thaw_r1_best dualview_r1_best; do
  p="$FNO/checkpoints/fno_ns_public_${n}.pt"
  [[ -f "$p" ]] && CKPTS+=("$p")
done
python3 eval_public_soup.py \
  --ckpts "${CKPTS[@]}" \
  --baseline "$BASE" --gate "$GATE" \
  --out-name fno_ns_public_soup_long_best.pt --always-save \
  2>&1 | tee -a "$CHAIN_LOG"
cp -f "$LOGDIR/fno_public_soup_r8_summary.json" "$LOGDIR/fno_public_soup_long_summary.json" 2>/dev/null || true

python3 - <<PY
import json, torch
from pathlib import Path
CK=Path("$FNO/checkpoints")
LOG=Path("$LOGDIR")
rows=[]
for p in sorted(CK.glob("fno_ns_public_*best*.pt"))+ [CK/"fno_ns_public_demo.pt"]:
  if not p.exists(): continue
  b=torch.load(p,map_location="cpu",weights_only=False)
  rows.append((float(b.get("test_l2",1e9)), p.name, b.get("promoted_tag")))
rows.sort()
gate=float("$GATE")
demo=0.03522327123209834
best=rows[0]
summary={
  "task":"fno_public_long_push_chain",
  "demo_l2":demo,
  "gate":gate,
  "best_sidecar":{"l2":best[0],"ckpt":best[1],"tag":best[2]},
  "top5":[{"l2":a,"ckpt":b,"tag":c} for a,b,c in rows[:5]],
  "beat_gate": best[0] < gate,
  "promote": False,
  "log":"$CHAIN_LOG",
}
(LOG/"fno_public_long_push_chain_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(summary,indent=2,ensure_ascii=False))
print("SIGNAL" if summary["beat_gate"] else "NO_SIGNAL")
PY
log "DONE long_push"
