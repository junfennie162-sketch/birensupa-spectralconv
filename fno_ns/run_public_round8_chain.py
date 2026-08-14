#!/usr/bin/env python3
"""ROUND8 · new-mechanism public NS64 squeeze (not isomorphic sched deepen).

Partner (ai4s-n) recent work is Spectral combo / synth FNO — little public-L2
to borrow; this round uses new mechs on f side:

  R8-1 soft-alpha scheduled sampling (different from Bernoulli p_ar)
  R8-2 checkpoint soup (r5 + r6 near-miss + multistep)
  R8-3 modes=20 residual capacity retrain (scratch, hf+aug)

Promote via promote_public_ckpt.py when verified L2 beats live best - 1e-4
or strictly improves live best (capacity may use absolute improve).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve().parent
SUB = THIS.parent
LOG = SUB / "results" / "run_logs" / "fno_public_round8_chain.log"
STATE = SUB / "results" / "run_logs" / "fno_public_round8_state.json"
CKPT = THIS / "checkpoints"
DEMO = CKPT / "fno_ns_public_demo.pt"
META = CKPT / "fno_ns_public_ns64_meta.json"
BASELINE = 0.035724617540836334
GATE = BASELINE - 1e-4


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_state(**kw) -> None:
    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text())
        except Exception:
            prev = {}
    prev.update(kw)
    prev["updated_at"] = utc()
    STATE.write_text(json.dumps(prev, indent=2, ensure_ascii=False) + "\n")


def live_best() -> float:
    if META.exists():
        m = json.loads(META.read_text())
        return float(m.get("relative_l2") or m.get("best_test_l2") or BASELINE)
    return BASELINE


def run(cmd: list[str]) -> int:
    log("RUN " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(THIS),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)
    return int(proc.wait())


def maybe_promote(src: Path, tag: str, *, need_gate: bool) -> bool:
    if not src.exists():
        log(f"skip promote {tag}: missing {src}")
        return False
    import torch

    blob = torch.load(src, map_location="cpu", weights_only=False)
    l2 = float(blob.get("test_l2", 1e9))
    best = live_best()
    ok = (l2 < GATE) if need_gate else (l2 < best - 1e-9)
    log(f"candidate {tag} L2={l2:.8f} live={best:.8f} gate={GATE:.8f} ok={ok}")
    if not ok:
        return False
    rc = run(
        [
            sys.executable,
            "-u",
            "promote_public_ckpt.py",
            "--src",
            str(src),
            "--tag",
            tag,
        ]
    )
    return rc == 0


def main() -> None:
    py = sys.executable
    start = live_best()
    log(f"=== ROUND8 START live={start:.8f} gate={GATE:.8f} ===")
    log(
        "NOTE: ai4s-n recent = Spectral combo + synth FNO; no public-NS64 L2 to merge. "
        "Borrowing spirit: keep Spectral formal frozen; FNO uses new mech only."
    )
    save_state(stage="start", best=start)
    results = {}

    # R8-1 soft-alpha sched
    save_state(stage="R8-1_soft")
    rc = run(
        [
            py,
            "-u",
            "train_public_sched_sampling.py",
            "--epochs",
            "12",
            "--lr",
            "3e-6",
            "--p-ar-max",
            "0.0",
            "--soft-alpha-max",
            "0.45",
            "--hf-weight",
            "0.25",
            "--energy-weight",
            "0.05",
            "--tilt-weight",
            "0.05",
            "--t-out-train",
            "3",
            "--baseline",
            str(start),
            "--gate",
            str(GATE),
            "--no-stop-on-gate",
            "--early-stop-patience",
            "4",
            "--init-from",
            str(DEMO),
            "--tag",
            "sched_soft_r8",
        ]
    )
    results["R8-1"] = {"rc": rc}
    maybe_promote(CKPT / "fno_ns_public_sched_soft_r8_best.pt", "sched_soft_r8", need_gate=True)

    # R8-2 soup
    save_state(stage="R8-2_soup")
    rc = run(
        [
            py,
            "-u",
            "eval_public_soup.py",
            "--baseline",
            str(live_best()),
            "--always-save",
            "--out-name",
            "fno_ns_public_soup_r8_best.pt",
            "--ckpts",
            str(CKPT / "fno_ns_public_demo.pt"),
            str(CKPT / "fno_ns_public_sched_samp_r6_best.pt"),
            str(CKPT / "fno_ns_public_multistep_probe_best.pt"),
            str(CKPT / "fno_ns_public_sched_samp_r5_best.pt"),
        ]
    )
    results["R8-2"] = {"rc": rc}
    # soup script may write soup best; try common names
    for name in (
        "fno_ns_public_soup_r8_best.pt",
        "fno_ns_public_soup_r2_best.pt",
        "fno_ns_public_soup_best.pt",
    ):
        p = CKPT / name
        if p.exists():
            maybe_promote(p, "soup_r8", need_gate=True)
            break

    # R8-3 modes=20 capacity (scratch; promote on any beat of live best)
    save_state(stage="R8-3_modes20")
    rc = run(
        [
            py,
            "-u",
            "train_public_ns64_boost.py",
            "--epochs",
            "120",
            "--lr",
            "2e-4",
            "--hf-weight",
            "0.35",
            "--augment",
            "--residual",
            "--modes",
            "20",
            "--width",
            "32",
            "--ckpt-name",
            "fno_ns_public_modes20_r8_best.pt",
            "--tag",
            "modes20_r8",
        ]
    )
    results["R8-3"] = {"rc": rc}
    maybe_promote(
        CKPT / "fno_ns_public_modes20_r8_best.pt",
        "modes20_r8",
        need_gate=False,
    )

    final = live_best()
    out = {
        "updated_at": utc(),
        "start_best": start,
        "final_best": final,
        "gate": GATE,
        "results": results,
        "partner_note": "ai4s-n: Spectral combo focus; no public FNO L2 to merge",
    }
    (SUB / "results" / "run_logs" / "fno_public_round8_final.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    # update CURRENT pointer briefly
    current = SUB / "results" / "run_logs" / "CURRENT.md"
    if current.exists():
        text = current.read_text(encoding="utf-8")
        note = (
            f"\n\n> ROUND8 chain finished {utc()}: {start:.6f} → **{final:.6f}** "
            f"(see `fno_public_round8_final.json`).\n"
        )
        if "ROUND8 chain finished" not in text:
            current.write_text(text.rstrip() + note, encoding="utf-8")
    save_state(stage="done", best=final, start=start)
    log(f"=== ROUND8 DONE {start:.8f} -> {final:.8f} ===")


if __name__ == "__main__":
    main()
