# 脏树整理（2026-08-16）

> 不 commit、不 `git add -A`、不 `--hard`。

## 做了什么

| 动作 | 对象 | 结果 |
|------|------|------|
| `git reset`（mixed） | 嵌套仓 `submission/` | 清掉 257 条半成品暂存（曾把 `AGENT_OFFICIAL.md` 等标成删除） |
| `git checkout HEAD -- _history/README.md` | 同上 | 只恢复误删的历史目录说明 |
| 父仓 | `/workspace/ai4s-f` | 未 reset；工作区保持现行 KEEP |

## 现在还脏、但该留

| 类 | 处理 |
|----|------|
| 现行主报 / KEEP `.su` / `summary.json` | 留下，等你点名再 commit |
| 已删的 `train_public_*.py`、旧 20260814 tar | 保持删除（精度线已停，旧包已换英文包） |
| 未跟踪的 `pruned_*.su`、探针 log | 留下（热路径源码） |
| `*.o` / `*.so` | gitignore 已覆盖；本机增量编译要用 |

官方必须项仍在磁盘上。嵌套仓不要整树 `git add -A`。
