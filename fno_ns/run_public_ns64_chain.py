#!/usr/bin/env python3
"""Autonomous PUBLIC NS64 FNO chain (offline-safe).

Stages (sequential):
  1) ensure public .pt present
  2) wait for / run scratch train_public_ns64.py (100 ep, 1000/128)
  3) continue polish from public best (50 ep, lower lr)
  4) final eval + write summary / disclosure / optional ai4s sync

SpectralConv does NOT use this dataset — skipped on purpose.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve().parent
SUB = THIS.parent
LOG_DIR = SUB / "results" / "run_logs"
CKPT = THIS / "checkpoints"
DATA = THIS / "data"
PUBLIC_PT = DATA / "navier_stokes_v1e-3_N1200_T20.pt"
PUBLIC_BEST = CKPT / "fno_ns_public_ns64_best.pt"
PUBLIC_META = CKPT / "fno_ns_public_ns64_meta.json"
PUBLIC_DEMO = CKPT / "fno_ns_public_demo.pt"
SCRATCH_SUMMARY = LOG_DIR / "fno_public_ns64_summary.json"
CHAIN_LOG = LOG_DIR / "fno_public_ns64_chain.log"
CHAIN_STATE = LOG_DIR / "fno_public_ns64_chain_state.json"
AI4S = Path("/workspace/ai4s/submission")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with CHAIN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_state(**kwargs) -> None:
    prev = {}
    if CHAIN_STATE.exists():
        try:
            prev = json.loads(CHAIN_STATE.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = utc()
    CHAIN_STATE.write_text(json.dumps(prev, indent=2, ensure_ascii=False) + "\n")


def pids_matching(pattern: str) -> list[int]:
    """Return PIDs whose cmdline contains pattern and looks like python (not a wrapper shell)."""
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    pids = []
    self_pid = os.getpid()
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if pid == self_pid:
            continue
        if "run_public_ns64_chain.py" in cmd:
            continue
        if "python" not in cmd:
            continue
        if pattern not in cmd:
            continue
        pids.append(pid)
    return pids


def wait_pids(pids: list[int], poll: float = 30.0) -> None:
    pending = set(pids)
    while pending:
        alive = set()
        for pid in pending:
            try:
                os.kill(pid, 0)
                alive.add(pid)
            except OSError:
                log(f"pid {pid} exited")
        pending = alive
        if pending:
            log(f"waiting pids={sorted(pending)}")
            time.sleep(poll)


def run_cmd(cmd: list[str], cwd: Path) -> int:
    log("RUN " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        with CHAIN_LOG.open("a", encoding="utf-8") as f:
            f.write(line)
    return int(proc.wait())


def scratch_done() -> bool:
    if not SCRATCH_SUMMARY.exists() or not PUBLIC_BEST.exists():
        return False
    try:
        s = json.loads(SCRATCH_SUMMARY.read_text())
    except Exception:
        return False
    return int(s.get("epochs", 0)) >= 100 and str(s.get("data_source", "")).startswith(
        "file:navier_stokes"
    )


def ensure_public_data() -> None:
    if not PUBLIC_PT.exists() or PUBLIC_PT.stat().st_size < 100_000_000:
        raise SystemExit(
            f"missing/incomplete public data: {PUBLIC_PT}. "
            "Download first (hf-mirror) before offline chain."
        )
    log(f"public_data_ok size_mb={PUBLIC_PT.stat().st_size / 1e6:.1f}")


def stage_scratch(epochs: int) -> None:
    save_state(stage="scratch")
    running = pids_matching("train_public_ns64.py")
    if running:
        log(f"scratch already running pids={running}; wait")
        wait_pids(running)
    if scratch_done():
        log("scratch already complete; skip")
        return
    rc = run_cmd(
        [sys.executable, "-u", "train_public_ns64.py", "--epochs", str(epochs)],
        cwd=THIS,
    )
    if rc != 0:
        raise SystemExit(f"scratch train failed rc={rc}")
    if not scratch_done():
        raise SystemExit("scratch finished but summary incomplete")
    log("scratch done")


def stage_continue(epochs: int, lr: float) -> None:
    save_state(stage="continue")
    if not PUBLIC_BEST.exists():
        raise SystemExit("public best missing before continue")
    backup = CKPT / "fno_ns_public_ns64_best_pre_continue.pt"
    out_ckpt = CKPT / "fno_ns_public_ns64_continue_best.pt"
    shutil.copy2(PUBLIC_BEST, backup)
    before = json.loads(PUBLIC_META.read_text()) if PUBLIC_META.exists() else {}
    before_l2 = float(before.get("best_test_l2", 1e9))

    rc = run_cmd(
        [
            sys.executable,
            "-u",
            "train_public_ns64.py",
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--init-from",
            str(backup),
            "--ckpt-name",
            "fno_ns_public_ns64_continue_best.pt",
        ],
        cwd=THIS,
    )
    if rc != 0:
        raise SystemExit(f"continue train failed rc={rc}")

    cont_meta = out_ckpt.with_name(out_ckpt.stem + "_meta.json")
    after_l2 = before_l2
    if cont_meta.exists():
        after_l2 = float(json.loads(cont_meta.read_text()).get("best_test_l2", before_l2))
    if out_ckpt.exists() and after_l2 < before_l2 - 1e-12:
        shutil.copy2(out_ckpt, PUBLIC_BEST)
        PUBLIC_META.write_text(
            json.dumps(
                {
                    "best_test_l2": after_l2,
                    "data_source": "file:navier_stokes_v1e-3_N1200_T20.pt",
                    "checkpoint": str(PUBLIC_BEST),
                    "from": "continue",
                },
                indent=2,
            )
            + "\n"
        )
        log(f"continue improved {before_l2:.6f} -> {after_l2:.6f}; promoted to public best")
    else:
        log(f"continue no improve vs {before_l2:.6f} (got {after_l2:.6f}); keep scratch best")
    save_state(continue_best_l2=min(before_l2, after_l2), before_l2=before_l2, after_l2=after_l2)


def stage_finalize(sync_ai4s: bool) -> dict:
    save_state(stage="finalize")
    sys.path.insert(0, str(THIS))
    import torch
    from torch.utils.data import DataLoader

    from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
    from model import FNO2d
    from test_forward import relative_l2

    SEED = 20260722
    data, src = load_or_build_ns_like(
        n_samples=1128, resolution=64, n_times=20, seed=SEED, version="v2"
    )
    if not str(src).startswith("file:navier_stokes"):
        raise SystemExit(f"finalize expected public file, got {src}")
    _, te = split_train_test(data, 1000, 128, seed=SEED)
    loader = DataLoader(SequenceVorticityDataset(te, 10, 1), batch_size=16, shuffle=False)

    ck = torch.load(PUBLIC_BEST, map_location="cpu", weights_only=False)
    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4, in_channels=10, out_channels=1)
    model.load_state_dict(ck["model"])
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            scores.append(relative_l2(model(x, use_supa=False), y))
    l2 = sum(scores) / len(scores)

    ck["test_l2"] = l2
    ck["data_source"] = src
    ck["promoted_tag"] = "public_ns64_chain"
    ck["split"] = {"n_train": 1000, "n_test": 128, "seed": SEED}
    torch.save(ck, PUBLIC_BEST)
    shutil.copy2(PUBLIC_BEST, PUBLIC_DEMO)

    result = {
        "updated_at": utc(),
        "data_source": src,
        "data_shape": list(data.shape),
        "public_official_l2_1000_128": l2,
        "checkpoint": str(PUBLIC_BEST),
        "public_demo": str(PUBLIC_DEMO),
        "note": "SpectralConv does not use NS64; FNO-only public chain.",
    }
    (LOG_DIR / "fno_public_ns64_final.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    log(f"FINAL public L2={l2:.8f}")

    # patch summary.json (keep spectral + legacy v2 fields)
    summary_path = SUB / "results" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.setdefault("meta", {})["updated_at"] = utc()
    summary["meta"]["notes"] = (
        f"Public NS64 FNO L2={l2:.6f} (chain). Legacy v2 continue3 kept separately."
    )
    fno = summary.setdefault("fno_ns", {})
    fno["public_ns64"] = {
        "status": "trained",
        "data_source": src,
        "relative_l2": l2,
        "n_train": 1000,
        "n_test": 128,
        "seed": SEED,
        "checkpoint": "fno_ns/checkpoints/fno_ns_public_demo.pt",
        "protocol": "official_gate_1000_128",
    }
    fno["data_disclosure"] = (
        "Primary public-NS64 metrics under fno_ns.public_ns64; "
        "legacy generated_ns_like_v2 metrics retained in history. "
        "See results/data_disclosure.md."
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    disclosure = SUB / "results" / "data_disclosure.md"
    stamp = utc()
    block = f"""

## 公开 NS64（自动链更新 {stamp}）

- 文件：`fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`（HF `abelsr1710/navier-stokes-2d-fno`）
- 布局：`[1200,20,64,64]`（loader 已从 `[N,H,W,T]` permute）
- 划分：n_train=1000 / n_test=128，seed=`20260722`
- 任务：T_in=10 → T_out=1
- 公开集 relative L2：**{l2:.8f}**
- checkpoint：`fno_ns/checkpoints/fno_ns_public_demo.pt`
- 说明：SpectralConv 必选项不使用该 NS 数据；本链仅覆盖 FNO-NS。
- 遗留：自建 `generated_ns_like_v2` 上的 continue3（L2≈0.005144）仍保留作工程对照，不表述为公开集成绩。
"""
    if disclosure.exists():
        text = disclosure.read_text(encoding="utf-8")
        marker = "## 公开 NS64（自动链更新"
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"
        disclosure.write_text(text + block, encoding="utf-8")
    else:
        disclosure.write_text("# FNO-NS 数据与训练披露\n" + block, encoding="utf-8")

    devlog = SUB / "development_log.md"
    if devlog.exists():
        entry = (
            f"\n### {stamp[:10]} · public NS64 autochain\n\n"
            f"- 公开 NS64 训练链完成，official 1000/128 relative L2 = **{l2:.6f}**\n"
            f"- ckpt: `fno_ns/checkpoints/fno_ns_public_demo.pt`\n"
            f"- SpectralConv 未使用该数据（算子题独立）\n"
        )
        with devlog.open("a", encoding="utf-8") as f:
            f.write(entry)

    if sync_ai4s and AI4S.is_dir():
        pairs = [
            (PUBLIC_DEMO, AI4S / "fno_ns" / "checkpoints" / "fno_ns_public_demo.pt"),
            (PUBLIC_BEST, AI4S / "fno_ns" / "checkpoints" / "fno_ns_public_ns64_best.pt"),
            (disclosure, AI4S / "results" / "data_disclosure.md"),
            (LOG_DIR / "fno_public_ns64_final.json", AI4S / "results" / "run_logs" / "fno_public_ns64_final.json"),
            (summary_path, AI4S / "results" / "summary.json"),
            (devlog, AI4S / "development_log.md"),
            (THIS / "train_public_ns64.py", AI4S / "fno_ns" / "train_public_ns64.py"),
            (THIS / "run_public_ns64_chain.py", AI4S / "fno_ns" / "run_public_ns64_chain.py"),
        ]
        for src_p, dst in pairs:
            if not src_p.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_p, dst)
            log(f"sync {src_p} -> {dst}")
    save_state(stage="done", final_l2=l2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-epochs", type=int, default=100)
    ap.add_argument("--continue-epochs", type=int, default=50)
    ap.add_argument("--continue-lr", type=float, default=1.0e-4)
    ap.add_argument("--skip-scratch", action="store_true")
    ap.add_argument("--skip-continue", action="store_true")
    ap.add_argument("--no-sync-ai4s", action="store_true")
    ap.add_argument(
        "--wait-pid",
        type=int,
        default=0,
        help="extra PID to wait before starting (e.g. already-running train)",
    )
    args = ap.parse_args()

    log("=== public NS64 autochain START ===")
    log("NOTE: SpectralConv does not consume NS64; chain is FNO-only.")
    save_state(stage="start")
    ensure_public_data()

    if args.wait_pid:
        wait_pids([args.wait_pid])

    # Also wait any stray public trainers before we decide
    extra = pids_matching("train_public_ns64.py")
    if extra and not args.skip_scratch:
        log(f"pre-wait existing trainers {extra}")
        wait_pids(extra)

    if not args.skip_scratch:
        stage_scratch(args.scratch_epochs)
    else:
        log("skip-scratch")

    if not args.skip_continue:
        stage_continue(args.continue_epochs, args.continue_lr)
    else:
        log("skip-continue")

    result = stage_finalize(sync_ai4s=not args.no_sync_ai4s)
    log("=== public NS64 autochain DONE === " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
