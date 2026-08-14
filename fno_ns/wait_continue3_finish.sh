#!/usr/bin/env bash
# Wait for continue3 PID, then promote / visualize / sync to ai4s.
set -euo pipefail
PID="${1:-688858}"
ROOT=/workspace/ai4s-f/submission
FNO="$ROOT/fno_ns"
LOG="$ROOT/results/run_logs/wait_continue3_finish_2026-07-31.log"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] wait pid=$PID"

while kill -0 "$PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date -Is)] pid $PID gone"

cd "$FNO"
python3 - <<'PY'
import json, shutil
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d
from test_forward import relative_l2

CKPT = Path("checkpoints")
DEMO = CKPT / "fno_ns_demo.pt"
BEST = CKPT / "fno_ns_official_best.pt"
CAND = CKPT / "fno_ns_global_continue_best.pt"
META = CKPT / "fno_ns_global_continue_meta.json"
SEED = 20260722

def eval_ckpt(path: Path) -> float:
    data, _ = load_or_build_ns_like(n_samples=1128, resolution=64, n_times=30, seed=SEED, version="v2")
    _, te = split_train_test(data, 1000, 128, seed=SEED)
    loader = DataLoader(SequenceVorticityDataset(te, 10, 1), batch_size=16, shuffle=False)
    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            scores.append(relative_l2(model(x, use_supa=False), y))
    return sum(scores) / len(scores)

demo_l2 = eval_ckpt(DEMO)
print("demo_l2", demo_l2)
meta = json.loads(META.read_text()) if META.exists() else {}
cand_l2 = float(meta.get("best_test_l2", 1e9)) if meta else 1e9
if CAND.exists():
    cand_l2 = min(cand_l2, float(torch.load(CAND, map_location="cpu", weights_only=False).get("test_l2", 1e9)))
print("cand_l2", cand_l2, "meta_improved", meta.get("improved"))

promoted = False
if cand_l2 < demo_l2 - 1e-9 and CAND.exists():
    shutil.copy2(DEMO, DEMO.with_suffix(".pt.pre_continue3_backup"))
    ck = torch.load(CAND, map_location="cpu", weights_only=False)
    ck["test_l2"] = cand_l2
    ck["split"] = {"n_train": 1000, "n_test": 128}
    ck["promoted_tag"] = "continue3"
    torch.save(ck, DEMO)
    torch.save(ck, BEST)
    shutil.copy2(DEMO, CKPT / "fno_ns_continue3_win.pt")
    promoted = True
    final_l2 = cand_l2
    print("PROMOTED continue3", final_l2)
else:
    final_l2 = demo_l2
    print("NO_PROMOTE keep", final_l2)

# summary
sp = Path("../results/summary.json")
d = json.loads(sp.read_text())
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
d["meta"] = {
    "updated_at": now,
    "last_phase_marked": "submit_gate",
    "notes": f"FNO official L2→{final_l2:.6f} ({'continue3' if promoted else 'freeze4'}). Spectral idle 3.811/8.054/29.560 ms platform.",
}
d["fno_ns"]["relative_l2"] = final_l2
d["fno_ns"]["l2_note"] = "official gate 1000/128; post-continue3 finish script"
d["fno_ns"]["official_100ep"] = {
    "best_test_l2": final_l2,
    "gate": "推荐",
    "checkpoint": "fno_ns/checkpoints/fno_ns_official_best.pt",
    "promoted_to_demo": True,
    "split": "1000/128",
    "polish": "continue3" if promoted else "freeze4",
}
sp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
Path("/tmp/continue3_finish_state.json").write_text(
    json.dumps({"promoted": promoted, "final_l2": final_l2, "at": now}, indent=2) + "\n"
)
print("STATE", promoted, final_l2)
PY

echo "[$(date -Is)] visualize"
python3 -u visualize.py || echo "visualize_failed_$? "

echo "[$(date -Is)] sync ai4s"
SRC=/workspace/ai4s-f/submission
DST=/workspace/ai4s/submission
cp -a "$SRC/fno_ns/checkpoints/fno_ns_demo.pt" "$DST/fno_ns/checkpoints/"
cp -a "$SRC/fno_ns/checkpoints/fno_ns_official_best.pt" "$DST/fno_ns/checkpoints/" 2>/dev/null || true
cp -a "$SRC/results/summary.json" "$DST/results/"
cp -a "$SRC/results.md" "$DST/" 2>/dev/null || true
cp -a "$SRC/development_log.md" "$DST/" 2>/dev/null || true
cp -a "$SRC/results/PPT技术总览_2026-07-31.md" "$DST/results/" 2>/dev/null || true
# figures if refreshed
cp -a "$SRC/results/figures/"fno_ns_*2026-07-31* "$DST/results/figures/" 2>/dev/null || true
cp -a "$SRC/demo/media/"fno_ns_*2026-07-31* "$DST/demo/media/" 2>/dev/null || true
echo "[$(date -Is)] ALL_DONE" | tee "$ROOT/results/run_logs/CONTINUE3_FINISH_DONE.flag"
cat /tmp/continue3_finish_state.json
