#!/usr/bin/env python3
"""Convert Cursor agent JSONL transcripts into Markdown judges can open.

Keeps the original .jsonl. Does not invent dialogue.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("/workspace/ai4s-f/submission/agent_logs")

CHATS = [
    {
        "src": Path(
            "/root/.cursor/projects/workspace/agent-transcripts/"
            "8258b9af-17b1-4d78-89fc-12fd826b64e9/"
            "8258b9af-17b1-4d78-89fc-12fd826b64e9.jsonl"
        ),
        "zh_name": "01_2026-07-31_确认主力工作区.md",
        "en_name": "01_2026-07-31_which_workspace.md",
        "zh_title": "交互记录 · 2026-07-31 · 确认主力工作区",
        "en_title": "Chat log · 2026-07-31 · which workspace is primary",
        "zh_intro": "确认 `ai4s-f` / `ai4s-n` 哪边是主力，以及当时公开集指标。下面按时间顺序还原当时的提问、Agent 回复和工具调用。",
        "en_intro": "Which tree is primary (`ai4s-f` vs `ai4s-n`) and the public-set metrics at the time. Turns below are the original user questions, agent replies, and tool calls, in order.",
    },
    {
        "src": Path(
            "/root/.cursor/projects/workspace/agent-transcripts/"
            "b2a7a02d-aa1c-4a6a-bb3b-d59c969db60e/"
            "b2a7a02d-aa1c-4a6a-bb3b-d59c969db60e.jsonl"
        ),
        "zh_name": "02_2026-08-02_新一轮优化.md",
        "en_name": "02_2026-08-02_optimization_round.md",
        "zh_title": "交互记录 · 2026-08-02 · 新一轮优化",
        "en_title": "Chat log · 2026-08-02 · next optimization round",
        "zh_intro": "根据当时评测报告开新一轮算子/模型优化。下面按时间顺序还原当时的提问、Agent 回复和工具调用。",
        "en_intro": "A new operator/model optimization round from the evaluation report at the time. Turns below are the original user questions, agent replies, and tool calls, in order.",
    },
]

USER_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.S)
TS_RE = re.compile(r"<timestamp>(.*?)</timestamp>", re.S)
MAX_ARG = 400


def _plain(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _clip(text: str, limit: int = MAX_ARG) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n…（过长已截断，完整参数见同名 .jsonl）"


def _user_text(raw: str) -> tuple[str, str]:
    ts = ""
    found_ts = TS_RE.search(raw)
    if found_ts:
        ts = found_ts.group(1).strip()
    found = USER_RE.search(raw)
    body = found.group(1).strip() if found else raw.strip()
    body = re.sub(r"</?user_query>", "", body).strip()
    return ts, body


def _tool_line(block: dict) -> str:
    name = block.get("name") or block.get("toolName") or "tool"
    args = block.get("input") or block.get("arguments") or {}
    desc = ""
    if isinstance(args, dict):
        desc = str(args.get("description") or "")
        bits = []
        for key in ("path", "command", "glob_pattern", "pattern", "url", "target_directory"):
            if key in args and args[key]:
                bits.append(f"{key}={_clip(str(args[key]), 220)}")
        extra = "; ".join(bits)
    else:
        extra = _clip(_plain(args), 220)
    parts = [f"`{name}`"]
    if desc:
        parts.append(desc)
    if extra:
        parts.append(extra)
    return "- " + " — ".join(parts)


def _blocks(obj: dict) -> list:
    msg = obj.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(obj.get("content"), list):
        return obj["content"]
    return []


def parse_turns(path: Path) -> list[dict]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        role = obj.get("role") or obj.get("type") or "unknown"
        texts = []
        tools = []
        for block in _blocks(obj):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                t = (block.get("text") or "").strip()
                if t:
                    texts.append(t)
            elif kind in {"tool_use", "tool_call"}:
                tools.append(_tool_line(block))
        if role in {"tool", "tool_result"} or (role == "unknown" and not texts and not tools):
            # skip bulky tool-result payloads; calls already listed
            continue
        if not texts and not tools:
            continue
        ts = ""
        body = "\n\n".join(texts)
        if role == "user":
            ts, body = _user_text(body or "\n".join(texts))
        turns.append({"role": role, "ts": ts, "text": body, "tools": tools})
    return turns


def render(turns: list[dict], *, title: str, intro: str, src_name: str, labels: dict) -> str:
    lines = [
        f"# {title}",
        "",
        intro,
        "",
        f"原始机器文件（未改）：`{src_name}`。本页只是换成 Markdown，方便直接打开。",
        "",
        f"- {labels['turns']}: **{len(turns)}**",
        f"- {labels['users']}: **{sum(1 for t in turns if t['role']=='user')}**",
        "",
        "---",
        "",
    ]
    n_user = 0
    n_as = 0
    for turn in turns:
        if turn["role"] == "user":
            n_user += 1
            head = f"## {labels['user']} {n_user}"
            if turn["ts"]:
                head += f" · {turn['ts']}"
            lines.append(head)
            lines.append("")
            lines.append(turn["text"] or "（空）")
            lines.append("")
        else:
            n_as += 1
            lines.append(f"## {labels['assistant']} {n_as}")
            lines.append("")
            if turn["text"]:
                lines.append(turn["text"])
                lines.append("")
            if turn["tools"]:
                lines.append(labels["tools"])
                lines.append("")
                lines.extend(turn["tools"])
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zh_lab = {
        "turns": "记录条数",
        "users": "用户提问次数",
        "user": "用户",
        "assistant": "Agent",
        "tools": "**调用的工具：**",
    }
    en_lab = {
        "turns": "records",
        "users": "user turns",
        "user": "User",
        "assistant": "Agent",
        "tools": "**Tools called:**",
    }
    for spec in CHATS:
        src = spec["src"]
        if not src.exists():
            raise SystemExit(f"missing {src}")
        (OUT / src.name).write_bytes(src.read_bytes())
        turns = parse_turns(src)
        (OUT / spec["zh_name"]).write_text(
            render(
                turns,
                title=spec["zh_title"],
                intro=spec["zh_intro"],
                src_name=src.name,
                labels=zh_lab,
            ),
            encoding="utf-8",
        )
        (OUT / spec["en_name"]).write_text(
            render(
                turns,
                title=spec["en_title"],
                intro=spec["en_intro"],
                src_name=src.name,
                labels=en_lab,
            ),
            encoding="utf-8",
        )
        print(src.name, "turns", len(turns), "->", spec["zh_name"], spec["en_name"])


if __name__ == "__main__":
    main()
