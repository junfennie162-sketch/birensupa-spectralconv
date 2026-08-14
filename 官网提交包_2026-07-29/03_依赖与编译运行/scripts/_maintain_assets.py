#!/usr/bin/env python3
"""Maintain contest submission assets after each development phase."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE_ORDER = (
    "skeleton",
    "spectral_accuracy",
    "spectral_perf",
    "fno_forward",
    "demo",
    "submit_gate",
)

PHASE_TO_SUMMARY_HINT = {
    "spectral_accuracy": (
        "Write spectral_conv.rel_error / status into summary.json, "
        "then fill results.md correctness section."
    ),
    "spectral_perf": (
        "Write spectral_conv.perf for 64/128/256 into summary.json, "
        "then fill results.md performance table."
    ),
    "fno_forward": (
        "Write fno_ns.rel_l2 / figures paths into summary.json, "
        "then fill results.md FNO section and keep figures/."
    ),
    "demo": "Refresh demo/scp_description.md and copy key figures into demo/media/.",
    "submit_gate": "Ensure development_log >= 5 entries covering >= 3 categories.",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: {path} must contain a JSON object")
    return data


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_asset(repo_root: Path, rel: str) -> Path:
    return repo_root / rel.rstrip("/")


def asset_exists(repo_root: Path, rel: str) -> bool:
    return resolve_asset(repo_root, rel).exists()


def require_phase(status: dict[str, Any], phase: str) -> dict[str, Any]:
    phases = status.get("phases")
    if not isinstance(phases, dict) or phase not in phases:
        raise SystemExit(f"ERROR: unknown phase '{phase}'")
    meta = phases[phase]
    if not isinstance(meta, dict):
        raise SystemExit(f"ERROR: invalid phase metadata for '{phase}'")
    return meta


def cmd_status(status: dict[str, Any]) -> int:
    print(f"current_phase: {status.get('current_phase')}")
    print(f"updated_at: {status.get('updated_at')}")
    print("")
    for name in PHASE_ORDER:
        meta = status.get("phases", {}).get(name, {})
        print(f"- {name}: {meta.get('status')}  # {meta.get('title')}")
    return 0


def cmd_check(repo_root: Path, status: dict[str, Any], phase: str) -> int:
    meta = require_phase(status, phase)
    missing: list[str] = []
    print(f"phase: {phase} ({meta.get('title')})")
    print(f"status: {meta.get('status')}")
    print("assets:")
    for rel in meta.get("assets", []):
        ok = asset_exists(repo_root, rel)
        print(f"  [{'OK' if ok else 'MISSING'}] {rel}")
        if not ok:
            missing.append(rel)
    print("checks (Agent must confirm):")
    for item in meta.get("checks", []):
        print(f"  - {item}")
    hint = PHASE_TO_SUMMARY_HINT.get(phase)
    if hint:
        print(f"hint: {hint}")
    if missing:
        print("")
        print(f"FAIL: {len(missing)} asset path(s) missing")
        return 2
    print("")
    print("PASS: required asset paths exist (semantic checks still manual)")
    return 0


def advance_current_phase(status: dict[str, Any], phase: str) -> str:
    idx = PHASE_ORDER.index(phase)
    for name in PHASE_ORDER[idx + 1 :]:
        if status["phases"][name].get("status") != "done":
            return name
    return phase


def cmd_mark_done(
    repo_root: Path,
    status_path: Path,
    summary_path: Path,
    status: dict[str, Any],
    phase: str,
) -> int:
    check_code = cmd_check(repo_root, status, phase)
    if check_code != 0:
        print("ERROR: refuse mark-done; fix missing assets first", file=sys.stderr)
        return check_code

    now = utc_now()
    status["phases"][phase]["status"] = "done"
    status["phases"][phase]["completed_at"] = now.split("T", 1)[0]
    status["updated_at"] = now
    status["current_phase"] = advance_current_phase(status, phase)
    dump_json(status_path, status)

    summary = load_json(summary_path)
    meta = summary.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        summary["meta"] = meta
    meta["last_phase_marked"] = phase
    meta["updated_at"] = now
    dump_json(summary_path, summary)

    print(f"marked done: {phase}")
    print(f"current_phase -> {status['current_phase']}")
    print("Agent follow-up:")
    print("  1) 回填 submission/results.md 对应章节")
    print("  2) 若本 phase 有有效交互，追加 submission/development_log.md")
    print("  3) 指标写入 submission/results/summary.json 业务字段")
    if phase in PHASE_TO_SUMMARY_HINT:
        print(f"  4) {PHASE_TO_SUMMARY_HINT[phase]}")
    return 0


def cmd_next(status: dict[str, Any]) -> int:
    pending = [
        name
        for name in PHASE_ORDER
        if status.get("phases", {}).get(name, {}).get("status") != "done"
    ]
    if not pending:
        print("all phases done; re-run: ./scripts/maintain_assets.sh check submit_gate")
        return 0
    name = pending[0]
    meta = status["phases"][name]
    print(f"next: {name}")
    print(f"title: {meta.get('title')}")
    print("assets:")
    for rel in meta.get("assets", []):
        print(f"  - {rel}")
    print("checks:")
    for item in meta.get("checks", []):
        print(f"  - {item}")
    print("")
    print(f"run: ./scripts/maintain_assets.sh check {name}")
    print(f"then: ./scripts/maintain_assets.sh mark-done {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maintain ai4s submission assets per development phase."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show phase statuses")
    check_p = sub.add_parser("check", help="Validate asset paths for a phase")
    check_p.add_argument("phase", choices=PHASE_ORDER)
    done_p = sub.add_parser("mark-done", help="Mark phase done after check passes")
    done_p.add_argument("phase", choices=PHASE_ORDER)
    sub.add_parser("next", help="Show next pending phase and required assets")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root: Path = args.repo_root.resolve()
    submission_root: Path = args.submission_root.resolve()
    status_path = submission_root / "results" / "phase_status.json"
    summary_path = submission_root / "results" / "summary.json"

    if not status_path.exists():
        print(f"ERROR: missing {status_path}", file=sys.stderr)
        return 1

    status = load_json(status_path)
    if args.command == "status":
        return cmd_status(status)
    if args.command == "check":
        return cmd_check(repo_root, status, args.phase)
    if args.command == "mark-done":
        return cmd_mark_done(
            repo_root, status_path, summary_path, status, args.phase
        )
    if args.command == "next":
        return cmd_next(status)
    print(f"ERROR: unknown command {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
