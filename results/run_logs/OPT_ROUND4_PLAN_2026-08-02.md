# OPT_ROUND4 · 2026-08-02 细致优化总方针

> **历史 · 已完成** → [`OPT_ROUND5_PLAN_2026-08-02.md`](OPT_ROUND5_PLAN_2026-08-02.md)  
> **现行**见 [`CURRENT.md`](CURRENT.md) · [`OPT_ROUND7_PLAN_2026-08-02.md`](OPT_ROUND7_PLAN_2026-08-02.md)。  
> 历史接棒自 [`OPT_ROUND3_PLAN_2026-08-02.md`](OPT_ROUND3_PLAN_2026-08-02.md)。  
> 工作区：`/workspace/ai4s-f` · 合入：`/workspace/ai4s/submission` · **勿改 `ai4s-n`**  
> 评测报告：全局只留**一份**最新 `评测报告_最新指标_*.md`（换戳删旧，不做 `.bak`）。

## 0. 基线（滚动）

| 角色 | 指标 | 数值 |
|------|------|------|
| FNO 精度 | public NS64 rel-L2 | **0.035855**（`sched_samp_r3`） |
| Round4 gate | best &lt; baseline−1e-4 | **&lt; 0.035755** |
| Spectral 性能 | idle 64/128/256 | **3.811 / 8.054 / 29.560 ms**（冻结） |
| Spectral 正确性 | worst rel | ~2.17e-7 |
| 3D 四角 | worst rel | ~1.19e-7 |

### 0.1 为何开 Round4（证据）

- r3 resume 轨迹在 ep3–5 **仍连升**后被用户叫停；停时 `p_ar≈0.16/0.45`，**未吃满** exposure schedule。
- 对外入口仍有危急漂移：`demo/scp` / `metrics_snapshot` / `skill.md` 停在 r2=0.036092。
- promote 后 chain / batch16 机读时间戳早于新 demo.pt，需短复测。

## 1. 主攻序（细致版）

```text
Wave-R4-0  零 GPU：危急口径闸 + OPT 指针 + dry-run + 脚本 default 防呆
Wave-R4-1  精度：ROUND4_SCHED_DEEPEN（早停耐心=3；破 gate 立即停训→promote）
           失败后备（各一轮）：soft → geom+roll；再失败则停精度
Wave-R4-1b 低 GPU 串行（CPU 训可交错）：chain → batch16 → bwd/3d/irregular
           严禁 perf / all / 写 formal ms
Wave-R4-2  零 GPU：叙事交叉钉 + development_log + matrix
Wave-R4-3  maintain + pack/sync + 唯一体评测报告（含 §9 累计提升）
```

### 1.1 交错纪律

| 状态 | 允许 | 禁止 |
|------|------|------|
| CPU 精度训进行中 | R4-0 / R4-2 文案；可读日志 | 第二个长训；解冻 formal |
| GPU 空窗 | R4-1b 单任务串行 | `test_perf` / `run_tests.sh all|perf` |
| 破 gate | **立刻**停训 → promote → visualize → 回写材料 | 为「再抠一点」空等剩余 ep |
| soft/geom | 仅 deepen `NO_SIGNAL` 后各开一轮 | 与 deepen 并行 |

## 2. Wave 状态（滚动）

| Wave | 状态 | 备注 |
|------|------|------|
| R4-0 | **done** | 危急口径 + stop-on-gate + dry-run |
| R4-1 | **NO_SIGNAL / 快路径停** | ep1 best=**0.035812**（优于 r3，未破 gate 0.035755）；跳过 soft/geom |
| R4-1b | **done** | chain PASS；batch16 复测（空闲再刷） |
| R4-2 | **done** | PPT§7 / matrix / log |
| R4-3 | **done** | maintain PASS；包 `20260802_115949`；精度无 promote |

## 3. 精度主线 CLI（P0）

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission/fno_ns

python3 train_public_sched_sampling.py \
  --epochs 10 --batch-size 16 --p-ar-max 0.55 --soft-alpha-max 0.0 \
  --lr 3e-6 --early-stop-patience 3 \
  --baseline 0.03585514333099127 --gate 0.03575514333099127 \
  --tag sched_samp_r4 --init-from checkpoints/fno_ns_public_demo.pt \
  2>&1 | tee /tmp/sched_samp_r4.txt
```

| 参数 | 取值 | 理由 |
|------|------|------|
| epochs | 10 | 比 12 短；靠 early-stop / 破 gate 早停 |
| p_ar-max | 0.55 | r3 只吃到 ~0.16；略抬后半程 |
| lr | 3e-6 | r3 用 5e-6 已贴底，再稳半档 |
| patience | 3 | 无提升即停，避免空等 |
| gate | 0.035755 | baseline−1e-4 |

**Promote 条件**：`best < 0.035755` 且 `promote_public_ckpt` 复验一致。

### 3.1 后备（仅 NO_SIGNAL）

- soft：`--soft-alpha-max 0.30 --p-ar-max 0 --tag soft_sched_r4 --epochs 8`
- geom：`train_public_geom_noise_probe.py --enable-roll --tag geom_roll_r4 --gate 0.035755 --epochs 8`

## 4. No-Go（继承 + 本轮强调）

同构 squeeze；解冻 formal ms；`test_perf`/`run_tests.sh all|perf` 写 formal；Plan2d/真融合；F-FNO 换主报；TTA 主报；v2 伪官方；SOL 得分句；fused 长训；width48→KD；B 站成片；改 `ai4s-n`；为 B16 chain 噪声去拧 fused。

## 5. 验收清单

- [ ] 危急三文件（scp / metrics / skill）锚 r3
- [ ] deepen 有 SIGNAL 或已记 NO_SIGNAL + 后备结论
- [ ] chain / batch16 时间戳晚于 demo promote（若有 promote）
- [ ] `maintain check submit_gate` PASS
- [ ] 新提交包；评测报告唯一时间戳 + §9
