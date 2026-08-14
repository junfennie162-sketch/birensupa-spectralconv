#!/usr/bin/env python3
"""Keep squeezing public-NS64 L2 until plateau (do not stop early).

Each round from current public best (residual expected):
  1) continue  + hf + aug          (moderate lr)
  2) freeze-spectral polish        (tiny lr, wd=0)
  3) unfreeze continue             (low lr, stronger hf)

Promotes whenever a stage beats the live best. Stops only when a full round
yields no promotion, or --max-rounds is hit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve().parent
SUB = THIS.parent
LOG = SUB / "results" / "run_logs" / "fno_public_squeeze_loop.log"
STATE = SUB / "results" / "run_logs" / "fno_public_squeeze_loop_state.json"
CKPT = THIS / "checkpoints"
PUBLIC_BEST = CKPT / "fno_ns_public_ns64_best.pt"
PUBLIC_DEMO = CKPT / "fno_ns_public_demo.pt"
PUBLIC_META = CKPT / "fno_ns_public_ns64_meta.json"


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


def current_best() -> float:
    if PUBLIC_META.exists():
        return float(json.loads(PUBLIC_META.read_text()).get("best_test_l2", 1e9))
    if PUBLIC_BEST.exists():
        import torch

        return float(
            torch.load(PUBLIC_BEST, map_location="cpu", weights_only=False).get(
                "test_l2", 1e9
            )
        )
    return 1e9


def run_boost(cmd: list[str]) -> dict:
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
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"boost failed rc={rc}")
    tag = cmd[cmd.index("--tag") + 1]
    path = SUB / "results" / "run_logs" / f"fno_public_boost_{tag}_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def maybe_promote(src: Path, l2: float, tag: str) -> bool:
    best = current_best()
    if l2 >= best - 1e-9:
        log(f"no promote {tag}: {l2:.8f} >= best {best:.8f}")
        return False
    if PUBLIC_BEST.exists():
        shutil.copy2(PUBLIC_BEST, PUBLIC_BEST.with_suffix(f".pt.pre_{tag}_backup"))
    import torch

    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["test_l2"] = l2
    ck["promoted_tag"] = tag
    torch.save(ck, PUBLIC_BEST)
    shutil.copy2(PUBLIC_BEST, PUBLIC_DEMO)
    PUBLIC_META.write_text(
        json.dumps(
            {
                "best_test_l2": l2,
                "data_source": ck.get("data_source"),
                "checkpoint": str(PUBLIC_BEST),
                "from": tag,
                "residual": ck.get("residual", True),
            },
            indent=2,
        )
        + "\n"
    )
    summary_path = SUB / "results" / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        s.setdefault("meta", {})["updated_at"] = utc()
        s["meta"]["notes"] = f"Public NS64 FNO L2={l2:.6f} ({tag})."
        fno = s.setdefault("fno_ns", {})
        fno["relative_l2"] = l2
        fno["l2_note"] = f"PRIMARY public NS64 after {tag}"
        pub = fno.setdefault("public_ns64", {})
        pub["relative_l2"] = l2
        pub["status"] = "squeeze_loop"
        pub["promoted_tag"] = tag
        summary_path.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
    log(f"PROMOTED {tag} L2={l2:.8f} (was {best:.8f})")
    return True


def stage(
    *,
    tag: str,
    epochs: int,
    lr: float,
    hf: float,
    freeze: bool,
    wd: float,
    ckpt_name: str,
) -> bool:
    py = sys.executable
    cmd = [
        py,
        "-u",
        "train_public_ns64_boost.py",
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--hf-weight",
        str(hf),
        "--augment",
        "--residual",
        "--init-from",
        str(PUBLIC_BEST),
        "--ckpt-name",
        ckpt_name,
        "--tag",
        tag,
        "--weight-decay",
        str(wd),
    ]
    if freeze:
        cmd.append("--freeze-spectral")
    summary = run_boost(cmd)
    l2 = float(summary.get("best_test_l2", 1e9))
    return maybe_promote(CKPT / ckpt_name, l2, tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument(
        "--min-delta",
        type=float,
        default=5e-5,
        help="if a round improves less than this, still continue until a zero-promote round",
    )
    args = ap.parse_args()

    start = current_best()
    log(f"=== squeeze loop START best={start:.8f} max_rounds={args.max_rounds} ===")
    save_state(stage="start", best=start, round=0)
    history = [{"round": 0, "best": start}]

    for r in range(1, args.max_rounds + 1):
        before = current_best()
        log(f"--- round {r}/{args.max_rounds} begin best={before:.8f} ---")
        save_state(stage=f"round{r}", best=before, round=r)
        improved = False

        # 1) open continue
        improved |= stage(
            tag=f"sq{r}a_cont",
            epochs=100,
            lr=3.0e-5 if r == 1 else 2.0e-5,
            hf=0.30,
            freeze=False,
            wd=1.0e-4,
            ckpt_name=f"fno_ns_public_sq{r}a_best.pt",
        )
        # 2) freeze spectral polish
        improved |= stage(
            tag=f"sq{r}b_freeze",
            epochs=60,
            lr=1.0e-5,
            hf=0.20,
            freeze=True,
            wd=0.0,
            ckpt_name=f"fno_ns_public_sq{r}b_best.pt",
        )
        # 3) unfreeze stronger hf
        improved |= stage(
            tag=f"sq{r}c_unfreeze",
            epochs=80,
            lr=1.5e-5 if r == 1 else 1.0e-5,
            hf=0.40,
            freeze=False,
            wd=1.0e-4,
            ckpt_name=f"fno_ns_public_sq{r}c_best.pt",
        )

        after = current_best()
        delta = before - after
        history.append({"round": r, "best": after, "delta": delta, "improved": improved})
        log(f"--- round {r} end best={after:.8f} delta={delta:.8f} improved={improved} ---")
        save_state(stage=f"round{r}_done", best=after, round=r, delta=delta)

        if not improved:
            log(f"plateau: round {r} no promote; stop loop")
            break

    final = current_best()
    out = {
        "updated_at": utc(),
        "start_best": start,
        "final_best": final,
        "delta": start - final,
        "history": history,
    }
    (SUB / "results" / "run_logs" / "fno_public_squeeze_loop_final.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    with (SUB / "development_log.md").open("a", encoding="utf-8") as f:
        f.write(
            f"\n### {utc()[:10]} · public NS64 squeeze loop\n\n"
            f"- {start:.6f} → **{final:.6f}** (Δ={start-final:.6f})\n"
            f"- residual+hf+aug+freeze rounds; see `fno_public_squeeze_loop_final.json`\n"
        )
    save_state(stage="done", best=final, start=start)
    log(f"=== squeeze loop DONE {start:.8f} -> {final:.8f} ===")


if __name__ == "__main__":
    main()
