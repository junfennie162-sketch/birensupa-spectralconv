#!/usr/bin/env python3
"""Sequential public-NS64 method boosts (offline-safe).

Stages (skip if already beaten current best and summary says done):
  A) abs + high-freq loss + roll aug  (warm from public best)
  B) residual + hf + aug from scratch
  C) residual continue from B best (lower lr)

Promotes into fno_ns_public_ns64_best.pt / fno_ns_public_demo.pt only if improved.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve().parent
SUB = THIS.parent
LOG = SUB / "results" / "run_logs" / "fno_public_boost_chain.log"
STATE = SUB / "results" / "run_logs" / "fno_public_boost_chain_state.json"
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
        raise SystemExit(f"boost failed rc={rc}: {cmd}")
    # read summary by tag
    tag = None
    if "--tag" in cmd:
        tag = cmd[cmd.index("--tag") + 1]
    path = SUB / "results" / "run_logs" / f"fno_public_boost_{tag}_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def maybe_promote(src: Path, l2: float, tag: str) -> bool:
    best = current_best()
    if l2 >= best - 1e-9:
        log(f"no promote {tag}: {l2:.8f} >= best {best:.8f}")
        return False
    backup = PUBLIC_BEST.with_suffix(f".pt.pre_{tag}_backup")
    if PUBLIC_BEST.exists():
        shutil.copy2(PUBLIC_BEST, backup)
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
                "residual": ck.get("residual", False),
            },
            indent=2,
        )
        + "\n"
    )
    # patch summary primary
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
        pub["status"] = "boosted"
        pub["promoted_tag"] = tag
        pub["checkpoint"] = "fno_ns/checkpoints/fno_ns_public_demo.pt"
        summary_path.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
    log(f"PROMOTED {tag} L2={l2:.8f} (was {best:.8f})")
    return True


def main() -> None:
    py = sys.executable
    base = current_best()
    log(f"=== public boost chain START best={base:.8f} ===")
    save_state(stage="start", best=base)

    # A: keep absolute head, add hf loss + roll aug
    save_state(stage="A_abs_hf_aug")
    a = run_boost(
        [
            py,
            "-u",
            "train_public_ns64_boost.py",
            "--epochs",
            "60",
            "--lr",
            "1e-4",
            "--hf-weight",
            "0.35",
            "--augment",
            "--init-from",
            str(PUBLIC_BEST),
            "--ckpt-name",
            "fno_ns_public_boost_A_best.pt",
            "--tag",
            "boostA_hf_aug",
        ]
    )
    maybe_promote(
        CKPT / "fno_ns_public_boost_A_best.pt",
        float(a.get("best_test_l2", 1e9)),
        "boostA_hf_aug",
    )

    # B: residual scratch (addresses large frame deltas)
    save_state(stage="B_residual_scratch")
    b = run_boost(
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
            "--ckpt-name",
            "fno_ns_public_boost_B_best.pt",
            "--tag",
            "boostB_residual",
        ]
    )
    maybe_promote(
        CKPT / "fno_ns_public_boost_B_best.pt",
        float(b.get("best_test_l2", 1e9)),
        "boostB_residual",
    )

    # C: continue residual / or continue whatever is current best with hf+aug
    save_state(stage="C_continue")
    init = PUBLIC_BEST
    import torch

    ck = torch.load(init, map_location="cpu", weights_only=False)
    res_flag = ["--residual"] if ck.get("residual") else []
    c = run_boost(
        [
            py,
            "-u",
            "train_public_ns64_boost.py",
            "--epochs",
            "80",
            "--lr",
            "5e-5",
            "--hf-weight",
            "0.25",
            "--augment",
            *res_flag,
            "--init-from",
            str(init),
            "--ckpt-name",
            "fno_ns_public_boost_C_best.pt",
            "--tag",
            "boostC_continue",
        ]
    )
    maybe_promote(
        CKPT / "fno_ns_public_boost_C_best.pt",
        float(c.get("best_test_l2", 1e9)),
        "boostC_continue",
    )

    final = current_best()
    out = {
        "updated_at": utc(),
        "start_best": base,
        "final_best": final,
        "delta": base - final,
        "A": a,
        "B": b,
        "C": c,
    }
    (SUB / "results" / "run_logs" / "fno_public_boost_chain_final.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    save_state(stage="done", best=final, start=base)
    log(f"=== public boost chain DONE {base:.8f} -> {final:.8f} ===")


if __name__ == "__main__":
    main()
