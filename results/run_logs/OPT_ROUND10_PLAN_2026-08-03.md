# OPT_ROUND10 · freeze 续抛光快路径（2026-08-03）

> 接棒 ROUND8/9 · 现行指针 [`CURRENT.md`](CURRENT.md) · 流程 [`../../skills/operator_opt_loop/LOOP_PROCESS.md`](../../skills/operator_opt_loop/LOOP_PROCESS.md)  
> 主报已对齐 **v8 · `freeze_r9` · L2 0.035302**。本轮只做**新机制续抛光**，禁同构 sched deepen。

## 基线与 gate

| 项 | 值 |
|----|-----|
| 正式主报 | **0.03530218452215195**（`freeze_r9` · v8） |
| gate | **&lt; 0.03520218452215195**（baseline − 1e-4） |
| Spectral formal | **冻结**（勿 `test_perf`） |
| GPU | 单卡串行；`nohup`；禁长 AwaitShell |

## Wave

| Wave | 动作 | 停条件 |
|------|------|--------|
| R10-0 | 材料对齐 A/B（报告 v8 · CURRENT · checklist · Agent 日志） | maintain / loop strict |
| R10-1 | `freeze_r10`：冻 spectral + 更低 lr 续抛光；`stop-on-gate` · patience≤3 | best&lt;gate → promote；近失/无提升 → NO_SIGNAL |
| R10-2 | SIGNAL → visualize + 报告 v9 + pack/sync；NO_SIGNAL → 停精度，转性能叙事 | — |
| R10-3 | （仅 NO_SIGNAL 后备）轻量 hf/energy 变体 ≤4ep **或** 材料收口 | 不与 R10-1 并行 |

## R10-1 命令（草案）

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission/fno_ns
nohup python3 -u train_public_ns64_boost.py \
  --epochs 40 --lr 5e-6 --hf-weight 0.15 \
  --augment --residual --freeze-spectral --weight-decay 0 \
  --modes 16 --width 32 \
  --init-from checkpoints/fno_ns_public_demo.pt \
  --ckpt-name fno_ns_public_freeze_r10_best.pt \
  --tag freeze_r10 \
  --gate 0.03520218452215195 --stop-on-gate --early-stop-patience 3 \
  > /tmp/fno_freeze_r10.txt 2>&1 &
```

秒查：`tail -30 /tmp/fno_freeze_r10.txt`

## 红线

1. 不同构再开 `sched_samp_r*` deepen  
2. 不默认 `test_perf`  
3. 未破 gate 不编评测报告 v 号  
4. f/n 禁止并发占卡  

## 状态

| 步 | 状态 |
|----|------|
| R10-0 材料 | **done**（报告 v8 · CURRENT · checklist · 记录 32–33） |
| R10-1 探针 | **NO_SIGNAL** · best=**0.035287** · gate=0.035202 · Δ≈1.55e-5 |
| R10-2 收口 | 跳过 promote（未破 gate） |
| R10-3 后备 | **NO_SIGNAL** · soft best=baseline · early_stop ep3 → **停精度** |
