#!/usr/bin/env python3
"""Operator / FNO optimization loop: dry-run SOP + summary/asset gate checks.

Default: no training, no formal perf overwrite. Use --strict for non-zero exit
when consistency or required assets fail.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SUB = Path(__file__).resolve().parents[2]
SUMMARY = SUB / "results" / "summary.json"
PHASE = SUB / "results" / "phase_status.json"
OUT = SUB / "results" / "run_logs" / "operator_opt_loop_last.json"
RUN_LOGS = SUB / "results" / "run_logs"
WORKSPACE = SUB.parent.parent  # /workspace when live tree is ai4s-f/submission

# Formal idle board (2026-08-14 recheck); drift beyond noise is a check failure signal.
EXPECTED_MS = {"64x64": 3.797, "128x128": 8.037, "256x256": 29.295}
MS_NOISE = 0.05
GATE_DELTA = 1e-4
PRIMARY_L2_TOL = 1e-6


def _log_hits(glob_pat: str) -> list[Path]:
    hits = list(RUN_LOGS.glob(glob_pat))
    hist = RUN_LOGS / "_history"
    if hist.is_dir():
        hits.extend(hist.glob(glob_pat))
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)


def _find_latest(glob_pat: str) -> Path | None:
    hits = _log_hits(glob_pat)
    return hits[0] if hits else None


def _log_exists(name: str) -> bool:
    return (RUN_LOGS / name).exists() or (RUN_LOGS / "_history" / name).exists()


def _latest_opt_round() -> dict:
    plans = _log_hits("OPT_ROUND*_PLAN_*.md")
    if not plans:
        return {"exists": False, "path": None, "round": None}
    p = plans[0]
    m = re.search(r"OPT_ROUND(\d+)", p.name)
    return {"exists": True, "path": str(p.relative_to(SUB)), "round": int(m.group(1)) if m else None}


def _eval_report() -> dict:
    reports = sorted(WORKSPACE.glob("评测报告_最新指标_*.md"))
    return {
        "count": len(reports),
        "single": len(reports) == 1,
        "latest": str(reports[-1]) if reports else None,
    }


def _count_agent_records(text: str) -> int:
    return len(re.findall(r"^## Agent 交互记录\s+\d+", text, flags=re.M))


def main() -> None:
    ap = argparse.ArgumentParser(description="operator_opt_loop SOP dry-run")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--run-accuracy", action="store_true")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if required assets/consistency fail",
    )
    ap.add_argument(
        "--json-only",
        action="store_true",
        help="print JSON payload only (no human checklist)",
    )
    args = ap.parse_args()

    s = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    phase = json.loads(PHASE.read_text()) if PHASE.exists() else {}
    sc = s.get("spectral_conv") or {}
    perf = sc.get("perf") or {}
    rows = perf.get("rows") or []
    fno = s.get("fno_ns") or {}
    pub = fno.get("public_ns64") or {}
    l2 = pub.get("relative_l2", fno.get("relative_l2"))
    tag = pub.get("promoted_tag")
    ckpt = pub.get("checkpoint") or fno.get("checkpoint")
    chain = fno.get("chain_consistency") or {}
    ckpt_rel = ((chain.get("checkpoint_model") or {}).get("relative_error"))
    batch16 = ((fno.get("perf_batch16") or {}).get("pure_forward") or {}).get(
        "grid_points_per_second"
    )

    baseline = float(l2) if isinstance(l2, (int, float)) else None
    gate = (baseline - GATE_DELTA) if baseline is not None else None

    # Next probe tag = max(existing sched_samp_rN)+1 (not merely promoted+1).
    sched_nums: set[int] = set()
    if isinstance(tag, str):
        m = re.search(r"sched_samp_r(\d+)$", tag)
        if m:
            sched_nums.add(int(m.group(1)))
    for p in RUN_LOGS.glob("*sched_samp_r*"):
        for n in re.findall(r"sched_samp_r(\d+)", p.name):
            sched_nums.add(int(n))
    for p in (SUB / "fno_ns" / "checkpoints").glob("*sched_samp_r*"):
        for n in re.findall(r"sched_samp_r(\d+)", p.name):
            sched_nums.add(int(n))
    next_tag = f"sched_samp_r{max(sched_nums) + 1}" if sched_nums else "sched_samp_r1"

    opt_latest = _latest_opt_round()
    report = _eval_report()

    log_path = SUB / "development_log.md"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    n_records = _count_agent_records(log_text)

    required_assets = {
        "summary": SUMMARY.exists(),
        "phase_status": PHASE.exists(),
        "readme": (SUB / "README.md").exists(),
        "results_md": (SUB / "results.md").exists(),
        "skill_md": (SUB / "skill.md").exists(),
        "submission_checklist": (SUB / "SUBMISSION_CHECKLIST.md").exists(),
        "development_log": log_path.exists() and n_records >= 5,
        "brsmi_snapshot": (SUB / "demo/media/brsmi_snapshot.txt").exists(),
        "official_alignment": _find_latest("OFFICIAL_ASSET_ALIGNMENT_*.md") is not None,
        "current_pointer": (RUN_LOGS / "CURRENT.md").exists(),
        "file_conventions": (SUB / "FILE_CONVENTIONS.md").exists(),
        "public_demo_ckpt": (SUB / "fno_ns/checkpoints/fno_ns_public_demo.pt").exists(),
        "opt_round_latest": bool(opt_latest["exists"]),
        "eval_report_single": report["single"],
    }

    narrative_artifacts = {
        "opt_master": _log_exists("OPT_MASTER_PLAN_2026-07-31.md"),
        "innovation_plan": _log_exists("OPT_INNOVATION_PLAN_2026-08-01.md"),
        "sol_gap": (SUB / "skills/sol_gap_analysis.md").exists(),
        "experiment_matrix": (SUB / "results/experiment_matrix.md").exists(),
        "supa_diff_story": _log_exists("supa_diff_loop_story.md"),
        "fused_segments": _find_latest("spectral_fused_segments_*.md") is not None,
        "tune_disclaimer": _find_latest("tune_skill_disclaimer_*.md") is not None,
        "multires_story": _find_latest("spectral_multires_story_*.md") is not None,
        "extension_showcase": _log_exists("extension_showcase.md"),
        "audit_card": _log_exists("SPECTRAL_BONUS_AUDIT_CARD.md"),
        "opt_round2": _log_exists("OPT_ROUND2_PLAN_2026-08-02.md"),
        "opt_round3": _log_exists("OPT_ROUND3_PLAN_2026-08-02.md"),
        "opt_round4": _log_exists("OPT_ROUND4_PLAN_2026-08-02.md"),
        "opt_round5": _log_exists("OPT_ROUND5_PLAN_2026-08-02.md"),
        "opt_round6": _log_exists("OPT_ROUND6_PLAN_2026-08-02.md"),
        "opt_round7": _log_exists("OPT_ROUND7_PLAN_2026-08-02.md"),
    }

    formal_ms_ok = bool(rows)
    for r in rows:
        res = r.get("resolution")
        ms = r.get("forward_time_ms")
        if res in EXPECTED_MS and isinstance(ms, (int, float)):
            if abs(float(ms) - EXPECTED_MS[res]) > MS_NOISE:
                formal_ms_ok = False
        elif res in EXPECTED_MS:
            formal_ms_ok = False
    if len([r for r in rows if r.get("resolution") in EXPECTED_MS]) < 3:
        formal_ms_ok = False

    consistency = {
        "public_l2_present": isinstance(l2, (int, float)),
        "public_l2_matches_summary_fields": (
            isinstance(l2, (int, float))
            and isinstance(fno.get("relative_l2"), (int, float))
            and abs(float(l2) - float(fno["relative_l2"])) < PRIMARY_L2_TOL
        ),
        "promoted_tag_present": isinstance(tag, str) and bool(tag),
        "formal_ms_board": formal_ms_ok,
        "chain_ckpt_under_1e-4": (
            isinstance(ckpt_rel, (int, float)) and float(ckpt_rel) < 1e-4
        ),
        "spectral_rel_under_1e-4": (
            isinstance(sc.get("rel_error"), (int, float)) and float(sc["rel_error"]) < 1e-4
        ),
        "phase_submit_gate_done": (
            (phase.get("phases") or {}).get("submit_gate", {}).get("status") == "done"
        ),
    }

    process_sop = [
        {
            "id": "P0",
            "name": "环境与单卡",
            "rule": "source brsw_set_env; 同时只跑一个 SUPA/GPU 任务",
        },
        {
            "id": "P1",
            "name": "读主报",
            "rule": "只读 summary.json；勿把 v2 / SOL / tune 当正式得分",
        },
        {
            "id": "P2",
            "name": "精度探针（可选）",
            "rule": (
                "nohup 后台；--stop-on-gate；epochs≤4；patience≤2；"
                "禁长 AwaitShell；秒查 /tmp 日志"
            ),
        },
        {
            "id": "P3",
            "name": "gate 裁决",
            "rule": "best<gate → promote+visualize；近失/NO_SIGNAL → 停精度或换机制",
        },
        {
            "id": "P4",
            "name": "护栏",
            "rule": "accuracy / chain；勿默认 test_perf 覆写 formal ms",
        },
        {
            "id": "P5",
            "name": "材料闭环",
            "rule": "checklist + Agent 日志段 + 唯一评测报告换戳；maintain check",
        },
        {
            "id": "P6",
            "name": "合入",
            "rule": "稳定文件 sync ai4s；有 promote 再 pack",
        },
    ]

    # Precision line posture: ROUND7 closed with NO_SIGNAL — default stop.
    precision_posture = "stopped_default"
    if opt_latest.get("round") is not None and opt_latest["round"] >= 7:
        precision_posture = "stopped_after_round7_no_signal"

    probe_cmd = None
    if baseline is not None and next_tag:
        probe_cmd = (
            f"cd fno_ns && nohup python3 train_public_sched_sampling.py "
            f"--tag {next_tag} --baseline {baseline:.12g} --gate {gate:.12g} "
            f"--epochs 4 --early-stop-patience 2 --stop-on-gate "
            f"> /tmp/{next_tag}.log 2>&1 &"
        )

    next_actions = [
        "python3 skills/operator_opt_loop/run_loop.py --dry-run --strict",
        "./scripts/maintain_assets.sh check submit_gate",
    ]
    if precision_posture.startswith("stopped"):
        next_actions.append(
            "精度线默认停：仅当有新机制（非同构 deepen）再用 probe_cmd；否则走材料/答辩"
        )
    else:
        next_actions.append(probe_cmd or "配置 baseline/gate 后再发探针")

    red_lines = [
        "禁止把 SOL/proxy/tune median 写成正式得分句",
        "禁止无争用证明时重跑 test_perf 写 summary.spectral_conv.perf",
        "禁止长 AwaitShell 挂起训练；用 nohup + cat 秒查",
        "禁止 f/n 并发占同一张 GPU",
        "禁止未破 gate 的探针编入评测报告正式 v 号",
    ]

    suggested_gates = [
        "cd spectral_conv && python3 test_accuracy.py",
        "cd fno_ns && python3 test_chain_cpu_supa_consistency.py",
        "./scripts/maintain_assets.sh check submit_gate",
    ]

    required_ok = all(required_assets.values())
    consistency_ok = all(consistency.values())
    narrative_ok = all(narrative_artifacts.values())

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": not args.run_accuracy,
        "strict": bool(args.strict),
        "public_ns64": {
            "relative_l2": l2,
            "promoted_tag": tag,
            "checkpoint": ckpt,
            "baseline": baseline,
            "gate": gate,
            "next_probe_tag": next_tag,
        },
        "engineering": {
            "spectral_rel_error": sc.get("rel_error"),
            "chain_ckpt_rel": ckpt_rel,
            "batch16_gps": batch16,
        },
        "spectral_rows": rows,
        "expected_ms": EXPECTED_MS,
        "required_assets": required_assets,
        "narrative_artifacts": narrative_artifacts,
        "consistency": consistency,
        "opt_round_latest": opt_latest,
        "eval_report": report,
        "agent_records": n_records,
        "precision_posture": precision_posture,
        "process_sop": process_sop,
        "probe_cmd": probe_cmd,
        "next_actions": next_actions,
        "red_lines": red_lines,
        "suggested_gates": suggested_gates,
        "pass": {
            "required_assets": required_ok,
            "consistency": consistency_ok,
            "narrative_artifacts": narrative_ok,
            "all_hard": required_ok and consistency_ok,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    if args.json_only:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("=== operator_opt_loop (SOP) ===")
        print(f"public_l2 {l2}  tag {tag}  baseline→gate {baseline} → {gate}")
        print(f"precision_posture {precision_posture}")
        print(f"opt_round_latest {opt_latest}")
        print(f"eval_report {report}")
        print(f"agent_records {n_records}")
        print("spectral_idle_rows", [(r.get("resolution"), r.get("forward_time_ms")) for r in rows])
        print("required_assets", required_assets)
        print("consistency", consistency)
        print("pass", payload["pass"])
        print("--- process_sop ---")
        for step in process_sop:
            print(f"  {step['id']} {step['name']}: {step['rule']}")
        print("--- red_lines ---")
        for line in red_lines:
            print(f"  - {line}")
        print("--- suggested_gates ---")
        for g in suggested_gates:
            print(f"  {g}")
        if probe_cmd:
            print("--- probe_cmd (only if new mechanism) ---")
            print(f"  {probe_cmd}")
        print("--- next_actions ---")
        for a in next_actions:
            print(f"  - {a}")
        print("wrote", OUT)

    if args.run_accuracy:
        rc = subprocess.call(
            [sys.executable, "test_accuracy.py"],
            cwd=str(SUB / "spectral_conv"),
        )
        if rc != 0:
            raise SystemExit(rc)

    hard_fail = not (required_ok and consistency_ok)
    soft_fail = not narrative_ok
    if hard_fail or soft_fail:
        print(
            "WARN: some checks failed "
            f"(required={required_ok} consistency={consistency_ok} narrative={narrative_ok}).",
            file=sys.stderr,
        )
    if args.strict and hard_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
