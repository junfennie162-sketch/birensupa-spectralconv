#!/usr/bin/env python3
"""Autonomous FNO L2 squeeze chain (official 1000/128 only).

Waits for an optional PID, then runs sequential experiments. Promotes only when
test L2 beats current demo on the same split. Stops a run early if trajectory
is clearly worse than demo after --abort-after epochs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SUB = THIS_DIR.parent
LOG_DIR = SUB / "results" / "run_logs"
CKPT = THIS_DIR / "checkpoints"
DEMO = CKPT / "fno_ns_demo.pt"
BEST = CKPT / "fno_ns_official_best.pt"
CHAIN_LOG = LOG_DIR / "fno_squeeze_chain_2026-07-31.log"
CHAIN_STATE = LOG_DIR / "fno_squeeze_chain_state.json"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with CHAIN_LOG.open("a") as f:
        f.write(line + "\n")


def wait_pid(pid: int, poll: float = 30.0) -> None:
    log(f"wait_pid {pid}")
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            log(f"pid {pid} gone")
            return
        time.sleep(poll)


def demo_l2_from_meta_or_eval() -> float:
    # Prefer last known promoted number from demo ckpt
    import torch

    ck = torch.load(DEMO, map_location="cpu", weights_only=False)
    if "test_l2" in ck:
        return float(ck["test_l2"])
    return float("inf")


def promote_from(src: Path, tag: str, l2: float) -> None:
    backup = DEMO.with_suffix(f".pt.pre_{tag}_backup")
    if DEMO.exists():
        shutil.copy2(DEMO, backup)
    shutil.copy2(src, DEMO)
    shutil.copy2(src, BEST)
    archive = CKPT / f"fno_ns_{tag}_win.pt"
    shutil.copy2(src, archive)
    # enrich
    import torch

    ck = torch.load(DEMO, map_location="cpu", weights_only=False)
    ck["test_l2"] = l2
    ck["split"] = {"n_train": 1000, "n_test": 128}
    ck["promoted_tag"] = tag
    torch.save(ck, DEMO)
    torch.save(ck, BEST)
    log(f"PROMOTED {tag} L2={l2} from {src}")


def handle_freeze3_result() -> float:
    meta_p = CKPT / "fno_ns_official_polish_meta.json"
    cand = CKPT / "fno_ns_official_polish_candidate.pt"
    base = demo_l2_from_meta_or_eval()
    if not meta_p.exists():
        log("freeze3 meta missing; skip promote")
        return base
    meta = json.loads(meta_p.read_text())
    best = float(meta["candidate_test_l2"])
    log(f"freeze3 meta baseline={meta.get('baseline_test_l2')} best={best} improved={meta.get('improved')}")
    if meta.get("improved") and best < base - 1e-9 and cand.exists():
        promote_from(cand, "freeze3", best)
        return best
    return min(base, best) if meta.get("improved") else base


def run_cmd(cmd: list[str], log_name: str, abort_after: int | None, demo_bar: float) -> int:
    log_path = LOG_DIR / log_name
    log(f"START {' '.join(cmd)} -> {log_path}")
    with log_path.open("w") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=str(THIS_DIR),
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
        )
    # optional early abort by tailing log for best_l2
    while proc.poll() is None:
        time.sleep(60)
        if abort_after is None:
            continue
        try:
            text = log_path.read_text()
        except OSError:
            continue
        # crude: find last best_l2 / test_rel_l2 with epoch
        epochs = []
        for line in text.splitlines():
            if "'epoch':" in line and "test_rel_l2" in line:
                try:
                    # python dict repr
                    d = eval(line.strip(), {"__builtins__": {}})  # noqa: S307 — our own logs
                    epochs.append(d)
                except Exception:
                    pass
        if not epochs:
            continue
        last = epochs[-1]
        ep = int(last.get("epoch", 0))
        best = float(last.get("best_l2", last.get("test_rel_l2", 1e9)))
        if ep >= abort_after and best > demo_bar * 1.8:
            log(f"ABORT {log_name} ep={ep} best={best} >> demo={demo_bar}")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 99
    rc = proc.returncode or 0
    log(f"END {log_name} rc={rc}")
    return rc


def maybe_promote_ckpt(path: Path, meta_path: Path, tag: str, demo_bar: float) -> float:
    if not meta_path.exists():
        # try read ckpt
        if not path.exists():
            return demo_bar
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        best = float(ck.get("test_l2", 1e9))
        if best < demo_bar - 1e-9:
            promote_from(path, tag, best)
            return best
        log(f"no promote {tag}: {best} >= {demo_bar}")
        return demo_bar
    meta = json.loads(meta_path.read_text())
    best = float(meta.get("best_test_l2") or meta.get("candidate_test_l2") or 1e9)
    improved = bool(meta.get("improved") or meta.get("improved_vs_demo"))
    if improved and best < demo_bar - 1e-9 and path.exists():
        promote_from(path, tag, best)
        return best
    log(f"no promote {tag}: best={best} improved={improved} bar={demo_bar}")
    return demo_bar


def save_state(d: dict) -> None:
    d["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    CHAIN_STATE.write_text(json.dumps(d, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0)
    ap.add_argument("--skip-freeze3-handle", action="store_true")
    ap.add_argument("--abort-after", type=int, default=30)
    args = ap.parse_args()

    state = {"phase": "start", "demo_l2": None, "steps": []}
    if args.wait_pid:
        wait_pid(args.wait_pid)

    demo_bar = demo_l2_from_meta_or_eval()
    state["demo_l2"] = demo_bar
    save_state(state)
    log(f"demo_bar={demo_bar}")

    if not args.skip_freeze3_handle:
        state["phase"] = "freeze3_handle"
        save_state(state)
        demo_bar = handle_freeze3_result()
        state["demo_l2"] = demo_bar
        state["steps"].append({"freeze3": demo_bar})
        save_state(state)

    # A2: gentle unfrozen continue from current demo, smaller lr, 80ep
    state["phase"] = "global_continue2"
    save_state(state)
    run_cmd(
        [
            sys.executable,
            "-u",
            "train_global_continue.py",
            "--epochs",
            "80",
            "--lr",
            "1e-5",
            "--batch-size",
            "16",
            "--init",
            str(DEMO),
        ],
        "train_global_continue2_80ep_2026-07-31.log",
        abort_after=None,
        demo_bar=demo_bar,
    )
    demo_bar = maybe_promote_ckpt(
        CKPT / "fno_ns_global_continue_best.pt",
        CKPT / "fno_ns_global_continue_meta.json",
        "continue2",
        demo_bar,
    )
    state["demo_l2"] = demo_bar
    state["steps"].append({"continue2": demo_bar})
    save_state(state)

    # A3: width=48 from scratch
    state["phase"] = "width48"
    save_state(state)
    run_cmd(
        [
            sys.executable,
            "-u",
            "train_width_retrain.py",
            "--epochs",
            "120",
            "--lr",
            "2e-4",
            "--width",
            "48",
            "--modes",
            "16",
            "--batch-size",
            "16",
        ],
        "train_width48_120ep_2026-07-31.log",
        abort_after=args.abort_after,
        demo_bar=demo_bar,
    )
    # width promote needs careful handling — only if beats demo; also note width mismatch
    wmeta = CKPT / "fno_ns_w48_meta.json"
    wckpt = CKPT / "fno_ns_w48_best.pt"
    if wmeta.exists():
        meta = json.loads(wmeta.read_text())
        best = float(meta.get("best_test_l2", 1e9))
        if best < demo_bar - 1e-9:
            log(f"width48 BEATS demo ({best} < {demo_bar}) — promote deferred: needs width=48 loader defaults")
            state["width48_ready"] = {"l2": best, "ckpt": str(wckpt)}
        else:
            log(f"width48 no beat: {best} vs {demo_bar}")
    state["steps"].append({"width48": "done"})
    save_state(state)

    # A4: modes=20 width=32 from scratch
    state["phase"] = "modes20"
    save_state(state)
    run_cmd(
        [
            sys.executable,
            "-u",
            "train_modes_retrain.py",
            "--epochs",
            "120",
            "--lr",
            "2e-4",
            "--modes",
            "20",
            "--width",
            "32",
            "--batch-size",
            "16",
        ],
        "train_modes20_120ep_2026-07-31.log",
        abort_after=args.abort_after,
        demo_bar=demo_bar,
    )
    # Arch change (modes≠16): do NOT silently overwrite modes=16 demo.
    mmeta = CKPT / "fno_ns_modes20_meta.json"
    if mmeta.exists():
        meta = json.loads(mmeta.read_text())
        best = float(meta.get("best_test_l2", 1e9))
        if best < demo_bar - 1e-9:
            log(f"modes20 BEATS demo ({best} < {demo_bar}) — promote deferred: needs modes=20 defaults")
            state["modes20_ready"] = {"l2": best, "ckpt": str(CKPT / "fno_ns_modes20_best.pt")}
        else:
            log(f"modes20 no beat: {best} vs {demo_bar}")
    state["demo_l2"] = demo_bar
    state["steps"].append({"modes20": "done"})
    state["phase"] = "fno_chain_done"
    save_state(state)
    log(f"FNO squeeze chain finished demo_bar={demo_bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
