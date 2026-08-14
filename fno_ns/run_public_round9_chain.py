#!/usr/bin/env python3
"""ROUND9 · continue public optimization after ROUND8.

  R9-1 modes20 continue (if R8 modes20 ckpt exists)
  R9-2 width=48 residual capacity (scratch)
  R9-3 soft+Bernoulli hybrid sched from live demo
  R9-4 freeze-spectral polish on live best
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve().parent
SUB = THIS.parent
LOG = SUB / "results" / "run_logs" / "fno_public_round9_chain.log"
STATE = SUB / "results" / "run_logs" / "fno_public_round9_state.json"
CKPT = THIS / "checkpoints"
DEMO = CKPT / "fno_ns_public_demo.pt"
META = CKPT / "fno_ns_public_ns64_meta.json"
MODES20 = CKPT / "fno_ns_public_modes20_r8_best.pt"


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
    if DEMO.exists():
        import torch

        return float(
            torch.load(DEMO, map_location="cpu", weights_only=False).get("test_l2", 1e9)
        )
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
    if not src.exists():
        log(f"skip promote {tag}: missing {src}")
        return False
    import torch

    blob = torch.load(src, map_location="cpu", weights_only=False)
    l2 = float(blob.get("test_l2", 1e9))
    best = live_best()
    if l2 >= best - 1e-9:
        log(f"no promote {tag}: {l2:.8f} >= {best:.8f}")
        return False
    rc = run(
        [sys.executable, "-u", "promote_public_ckpt.py", "--src", str(src), "--tag", tag]
    )
    return rc == 0


def main() -> None:
    py = sys.executable
    start = live_best()
    log(f"=== ROUND9 START live={start:.8f} ===")
    save_state(stage="start", best=start)

    # R9-1 modes20 continue
    save_state(stage="R9-1_modes20_cont")
    if MODES20.exists():
        run(
            [
                py,
                "-u",
                "train_public_ns64_boost.py",
                "--epochs",
                "80",
                "--lr",
                "5e-5",
                "--hf-weight",
                "0.3",
                "--augment",
                "--residual",
                "--modes",
                "20",
                "--width",
                "32",
                "--init-from",
                str(MODES20),
                "--ckpt-name",
                "fno_ns_public_modes20_r9_best.pt",
                "--tag",
                "modes20_r9",
            ]
        )
        maybe_promote(CKPT / "fno_ns_public_modes20_r9_best.pt", "modes20_r9")
    else:
        log("skip R9-1: no modes20_r8 ckpt yet")

    # R9-2 width48 scratch
    save_state(stage="R9-2_width48")
    run(
        [
            py,
            "-u",
            "train_public_ns64_boost.py",
            "--epochs",
            "100",
            "--lr",
            "2e-4",
            "--hf-weight",
            "0.35",
            "--augment",
            "--residual",
            "--modes",
            "16",
            "--width",
            "48",
            "--ckpt-name",
            "fno_ns_public_width48_r9_best.pt",
            "--tag",
            "width48_r9",
        ]
    )
    maybe_promote(CKPT / "fno_ns_public_width48_r9_best.pt", "width48_r9")

    # R9-3 hybrid sched (soft + p_ar) from current demo
    save_state(stage="R9-3_hybrid_sched")
    best = live_best()
    run(
        [
            py,
            "-u",
            "train_public_sched_sampling.py",
            "--epochs",
            "16",
            "--lr",
            "2e-6",
            "--p-ar-max",
            "0.4",
            "--soft-alpha-max",
            "0.25",
            "--hf-weight",
            "0.25",
            "--energy-weight",
            "0.05",
            "--tilt-weight",
            "0.05",
            "--t-out-train",
            "4",
            "--baseline",
            str(best),
            "--gate",
            str(best - 1e-4),
            "--no-stop-on-gate",
            "--early-stop-patience",
            "5",
            "--init-from",
            str(DEMO),
            "--tag",
            "sched_hybrid_r9",
        ]
    )
    maybe_promote(CKPT / "fno_ns_public_sched_hybrid_r9_best.pt", "sched_hybrid_r9")

    # R9-4 freeze polish — only if demo is modes16/width32 compatible
    save_state(stage="R9-4_freeze")
    import torch

    demo_blob = torch.load(DEMO, map_location="cpu", weights_only=False)
    modes = int(demo_blob.get("modes", 16))
    width = int(demo_blob.get("width", 32))
    run(
        [
            py,
            "-u",
            "train_public_ns64_boost.py",
            "--epochs",
            "60",
            "--lr",
            "1e-5",
            "--hf-weight",
            "0.2",
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
            "fno_ns_public_freeze_r9_best.pt",
            "--tag",
            "freeze_r9",
        ]
    )
    maybe_promote(CKPT / "fno_ns_public_freeze_r9_best.pt", "freeze_r9")

    final = live_best()
    out = {
        "updated_at": utc(),
        "start_best": start,
        "final_best": final,
        "delta": start - final,
    }
    (SUB / "results" / "run_logs" / "fno_public_round9_final.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    save_state(stage="done", best=final, start=start)
    log(f"=== ROUND9 DONE {start:.8f} -> {final:.8f} ===")


if __name__ == "__main__":
    main()
