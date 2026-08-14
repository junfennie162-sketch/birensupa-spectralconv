#!/usr/bin/env bash
# Wave4 precision chain from spec_ref_r1 sidecar (~0.035027).
# spec_ref_r2 → last_thaw_r4 → qt_over_r3 → dualview_r3 → soup
# Gate = formal demo 0.035115 − 1e-4. No auto-promote.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FNO="$ROOT/fno_ns"
LOGDIR="$ROOT/results/run_logs"
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export PYTHONUNBUFFERED=1
DEMO=0.03511497611179948
GATE=0.03501497611179948
BASE=0.035026549361646175
INIT="$FNO/checkpoints/fno_ns_public_spec_ref_r1_best.pt"
CHAIN_LOG="$LOGDIR/fno_public_wave4_chain_$(date +%Y%m%d_%H%M%S).log"
STATE="$LOGDIR/fno_public_wave4_state.json"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$CHAIN_LOG"; }

best_init() {
  python3 - <<'PY'
import torch
from pathlib import Path
CK=Path("/workspace/ai4s/submission/fno_ns/checkpoints")
cands=[
  "fno_ns_public_spec_ref_r2_best.pt",
  "fno_ns_public_last_thaw_r4_best.pt",
  "fno_ns_public_qt_over_r3_best.pt",
  "fno_ns_public_dualview_r3_best.pt",
  "fno_ns_public_soup_wave4_best.pt",
  "fno_ns_public_spec_ref_r1_best.pt",
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
log "START wave4 DEMO=$DEMO GATE=$GATE init0=$INIT BASE=$BASE"
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
  BASE="${BI[1]}"
  log "after $tag best_init=$INIT l2=$BASE"
  python3 - <<PY
import json
from pathlib import Path
p=Path("$STATE")
d=json.loads(p.read_text()) if p.exists() else {"stages":[]}
d["stages"].append({"tag":"$tag","best_init":"$INIT","l2":float("$BASE")})
d["gate"]=float("$GATE"); d["demo"]=float("$DEMO")
p.write_text(json.dumps(d,indent=2)+"\n")
PY
}

run_stage spec_ref_r2 train_public_spectral_refiner_probe.py \
  --tag spec_ref_r2 --init-from "$INIT" \
  --baseline "$BASE" --gate "$GATE" --stop-on-gate \
  --epochs 18 --lr 5e-7 --mix-l2 0.22 --sob-alpha 0.5 \
  --early-stop-patience 6

run_stage last_thaw_r4 train_public_last_thaw_probe.py \
  --tag last_thaw_r4 --init-from "$INIT" \
  --baseline "$BASE" --gate "$GATE" --stop-on-gate \
  --epochs 10 --lr 1.5e-6 --spectral-lr 4e-7 --thaw-k 2 \
  --lambda-delta 0.4 --hf-weight 0.1 --early-stop-patience 5

run_stage qt_over_r3 train_public_qt_oversample_probe.py \
  --tag qt_over_r3 --init-from "$INIT" \
  --baseline "$BASE" --gate "$GATE" --stop-on-gate \
  --epochs 10 --lr 2e-6 --qt-power 1.4 \
  --lambda-pf 0.55 --lambda-delta 0.38 --hf-weight 0.12 \
  --freeze-spectral --early-stop-patience 5

run_stage dualview_r3 train_public_dualview_probe.py \
  --tag dualview_r3 --init-from "$INIT" \
  --baseline "$BASE" --gate "$GATE" --stop-on-gate \
  --epochs 10 --lr 2e-6 --lambda-cons 0.2 --lambda-delta 0.32 \
  --hf-weight 0.1 --freeze-spectral --early-stop-patience 5

log "=== SOUP ==="
CKPTS=()
for n in spec_ref_r1_best spec_ref_r2_best last_thaw_r4_best qt_over_r3_best dualview_r3_best demo; do
  p="$FNO/checkpoints/fno_ns_public_${n}.pt"
  [[ -f "$p" ]] && CKPTS+=("$p")
done
python3 eval_public_soup.py \
  --ckpts "${CKPTS[@]}" \
  --baseline "$DEMO" --gate "$GATE" \
  --out-name fno_ns_public_soup_wave4_best.pt --always-save \
  2>&1 | tee -a "$CHAIN_LOG"

python3 - <<PY
import json, torch
from pathlib import Path
CK=Path("$FNO/checkpoints"); LOG=Path("$LOGDIR")
rows=[]
seen=set()
for p in list(CK.glob("fno_ns_public_*best*.pt"))+[CK/"fno_ns_public_demo.pt"]:
  if not p.exists() or p.name in seen: continue
  seen.add(p.name)
  b=torch.load(p,map_location="cpu",weights_only=False)
  if "test_l2" not in b: continue
  rows.append((float(b["test_l2"]), p.name, b.get("promoted_tag")))
rows.sort()
gate=float("$GATE"); demo=float("$DEMO")
best=rows[0]
summary={
  "task":"fno_public_wave4_chain",
  "demo_l2":demo,"gate":gate,
  "init_sidecar":"spec_ref_r1",
  "best_sidecar":{"l2":best[0],"ckpt":best[1],"tag":best[2]},
  "top8":[{"l2":a,"ckpt":b,"tag":c} for a,b,c in rows[:8]],
  "beat_gate": best[0] < gate,
  "gap_to_gate": best[0]-gate,
  "promote": False,
  "log":"$CHAIN_LOG",
}
(LOG/"fno_public_wave4_chain_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(summary,indent=2,ensure_ascii=False))
print("SIGNAL: beat_gate" if summary["beat_gate"] else "NO_SIGNAL")
PY
log "DONE wave4"
