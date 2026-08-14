#!/usr/bin/env python3
"""ROUND11 · follow-on after ROUND10.

HISTORICAL / DO NOT RE-RUN as default:
  hybrid / modes20 / isomorphic freeze deepen are OPT_WAVE D9 KILL.
  maybe_promote requires gate (live-1e-4) + ALLOW_AUTO_PROMOTE=1.

  R11-1 hybrid sched (soft+p_ar) from live demo  # KILL
  R11-2 modes20 long continue if ckpt exists     # KILL
  R11-3 tiny-lr freeze polish                    # KILL deepen
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
LOG = SUB / "results" / "run_logs" / "fno_public_round11_chain.log"
STATE = SUB / "results" / "run_logs" / "fno_public_round11_state.json"
CKPT = THIS / "checkpoints"
DEMO = CKPT / "fno_ns_public_demo.pt"
META = CKPT / "fno_ns_public_ns64_meta.json"
M20 = CKPT / "fno_ns_public_modes20_r9_best.pt"


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
    return (
        run(
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
        == 0
    )


def demo_arch() -> tuple[int, int]:
    import torch

    b = torch.load(DEMO, map_location="cpu", weights_only=False)
    return int(b.get("modes", 16)), int(b.get("width", 32))


def main() -> None:
    py = sys.executable
    start = live_best()
    modes, width = demo_arch()
    log(f"=== ROUND11 START live={start:.8f} modes={modes} width={width} ===")
    save_state(stage="start", best=start)

    save_state(stage="R11-1_hybrid")
    best = live_best()
    run(
        [
            py,
            "-u",
            "train_public_sched_sampling.py",
            "--epochs",
            "24",
            "--lr",
            "1.2e-6",
            "--p-ar-max",
            "0.35",
            "--soft-alpha-max",
            "0.35",
            "--hf-weight",
            "0.28",
            "--energy-weight",
            "0.06",
            "--tilt-weight",
            "0.06",
            "--t-out-train",
            "5",
            "--baseline",
            str(best),
            "--gate",
            str(best - 6e-5),
            "--no-stop-on-gate",
            "--early-stop-patience",
            "7",
            "--init-from",
            str(DEMO),
            "--tag",
            "sched_hybrid_r11",
        ]
    )
    maybe_promote(CKPT / "fno_ns_public_sched_hybrid_r11_best.pt", "sched_hybrid_r11")

    save_state(stage="R11-2_m20")
    if M20.exists():
        run(
            [
                py,
                "-u",
                "train_public_ns64_boost.py",
                "--epochs",
                "100",
                "--lr",
                "3e-5",
                "--hf-weight",
                "0.35",
                "--augment",
                "--residual",
                "--modes",
                "20",
                "--width",
                "32",
                "--init-from",
                str(M20),
                "--ckpt-name",
                "fno_ns_public_modes20_r11_best.pt",
                "--tag",
                "modes20_r11",
            ]
        )
        maybe_promote(CKPT / "fno_ns_public_modes20_r11_best.pt", "modes20_r11")
    else:
        log("skip R11-2: no modes20 ckpt")

    save_state(stage="R11-3_freeze")
    modes, width = demo_arch()
    run(
        [
            py,
            "-u",
            "train_public_ns64_boost.py",
            "--epochs",
            "80",
            "--lr",
            "5e-6",
            "--hf-weight",
            "0.12",
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
            "fno_ns_public_freeze_r11_best.pt",
            "--tag",
            "freeze_r11",
        ]
    )
    maybe_promote(CKPT / "fno_ns_public_freeze_r11_best.pt", "freeze_r11")

    final = live_best()
    (SUB / "results" / "run_logs" / "fno_public_round11_final.json").write_text(
        json.dumps(
            {
                "updated_at": utc(),
                "start_best": start,
                "final_best": final,
                "delta": start - final,
            },
            indent=2,
        )
        + "\n"
    )
    save_state(stage="done", best=final, start=start)
    log(f"=== ROUND11 DONE {start:.8f} -> {final:.8f} ===")


if __name__ == "__main__":
    main()
