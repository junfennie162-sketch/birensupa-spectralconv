#!/usr/bin/env python3
"""ROUND10 · keep squeezing from freeze_r9 (0.035302).

HISTORICAL / DO NOT RE-RUN as default:
  width48 / hybrid / modes20 are OPT_WAVE D9 KILL stages.
  maybe_promote requires gate (live-1e-4) + ALLOW_AUTO_PROMOTE=1.

  R10-1 unfreeze low-lr continue + hf/aug from demo
  R10-2 soft-α sched deepen from demo (longer)
  R10-3 width48 continue (was 0.03613; more epochs)  # KILL — do not re-open
  R10-4 freeze polish again if R10-1/2 improved
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from promote_guard import evaluate_promote

THIS = Path(__file__).resolve().parent
SUB = THIS.parent
LOG = SUB / "results" / "run_logs" / "fno_public_round10_chain.log"
STATE = SUB / "results" / "run_logs" / "fno_public_round10_state.json"
CKPT = THIS / "checkpoints"
DEMO = CKPT / "fno_ns_public_demo.pt"
META = CKPT / "fno_ns_public_ns64_meta.json"
W48 = CKPT / "fno_ns_public_width48_r9_best.pt"


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
        return float(m.get("relative_l2") or m.get("best_test_l2") or 1e9)
    return 1e9


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


def maybe_promote(src: Path, tag: str) -> bool:
    best = live_best()
    ok, _l2, _gate = evaluate_promote(src, tag, best, log)
    if not ok:
        return False
    rc = run(
        [sys.executable, "-u", "promote_public_ckpt.py", "--src", str(src), "--tag", tag]
    )
    return rc == 0


def demo_arch() -> tuple[int, int]:
    import torch

    blob = torch.load(DEMO, map_location="cpu", weights_only=False)
    return int(blob.get("modes", 16)), int(blob.get("width", 32))


def main() -> None:
    py = sys.executable
    start = live_best()
    modes, width = demo_arch()
    log(f"=== ROUND10 START live={start:.8f} demo modes={modes} width={width} ===")
    save_state(stage="start", best=start)

    # R10-1 unfreeze continue
    save_state(stage="R10-1_cont")
    run(
        [
            py,
            "-u",
            "train_public_ns64_boost.py",
            "--epochs",
            "100",
            "--lr",
            "2e-5",
            "--hf-weight",
            "0.3",
            "--augment",
            "--residual",
            "--modes",
            str(modes),
            "--width",
            str(width),
            "--init-from",
            str(DEMO),
            "--ckpt-name",
            "fno_ns_public_cont_r10_best.pt",
            "--tag",
            "cont_r10",
        ]
    )
    maybe_promote(CKPT / "fno_ns_public_cont_r10_best.pt", "cont_r10")

    # R10-2 soft sched
    save_state(stage="R10-2_soft")
    best = live_best()
    run(
        [
            py,
            "-u",
            "train_public_sched_sampling.py",
            "--epochs",
            "20",
            "--lr",
            "1.5e-6",
            "--p-ar-max",
            "0.0",
            "--soft-alpha-max",
            "0.55",
            "--hf-weight",
            "0.3",
            "--energy-weight",
            "0.08",
            "--tilt-weight",
            "0.08",
            "--t-out-train",
            "4",
            "--baseline",
            str(best),
            "--gate",
            str(best - 8e-5),
            "--no-stop-on-gate",
            "--early-stop-patience",
            "6",
            "--init-from",
            str(DEMO),
            "--tag",
            "sched_soft_r10",
        ]
    )
    maybe_promote(CKPT / "fno_ns_public_sched_soft_r10_best.pt", "sched_soft_r10")

    # R10-3 width48 continue
    save_state(stage="R10-3_w48")
    if W48.exists():
        run(
            [
                py,
                "-u",
                "train_public_ns64_boost.py",
                "--epochs",
                "120",
                "--lr",
                "5e-5",
                "--hf-weight",
                "0.35",
                "--augment",
                "--residual",
                "--modes",
                "16",
                "--width",
                "48",
                "--init-from",
                str(W48),
                "--ckpt-name",
                "fno_ns_public_width48_r10_best.pt",
                "--tag",
                "width48_r10",
            ]
        )
        maybe_promote(CKPT / "fno_ns_public_width48_r10_best.pt", "width48_r10")
    else:
        log("skip R10-3: no width48 ckpt")

    # R10-4 freeze polish on whatever is live demo now
    save_state(stage="R10-4_freeze")
    modes, width = demo_arch()
    run(
        [
            py,
            "-u",
            "train_public_ns64_boost.py",
            "--epochs",
            "80",
            "--lr",
            "8e-6",
            "--hf-weight",
            "0.15",
            "--augment",
            "--residual",
            "--freeze-spectral",
            "--weight-decay",
            "0",
            "--modes",
            str(modes),
            "--width",
            str(width),
            "--init-from",
            str(DEMO),
            "--ckpt-name",
            "fno_ns_public_freeze_r10_best.pt",
            "--tag",
            "freeze_r10",
        ]
    )
    maybe_promote(CKPT / "fno_ns_public_freeze_r10_best.pt", "freeze_r10")

    final = live_best()
    out = {
        "updated_at": utc(),
        "start_best": start,
        "final_best": final,
        "delta": start - final,
    }
    (SUB / "results" / "run_logs" / "fno_public_round10_final.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    current = SUB / "results" / "run_logs" / "CURRENT.md"
    if current.exists():
        text = current.read_text(encoding="utf-8")
        note = (
            f"\n\n> ROUND10 finished {utc()}: {start:.6f} → **{final:.6f}** "
            f"(`fno_public_round10_final.json`).\n"
        )
        if "ROUND10 finished" not in text:
            current.write_text(text.rstrip() + note, encoding="utf-8")
    save_state(stage="done", best=final, start=start)
    log(f"=== ROUND10 DONE {start:.8f} -> {final:.8f} ===")


if __name__ == "__main__":
    main()
