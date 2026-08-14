# 评委 3 分钟抽查一页包（2026-08-04 文件名 · 正文 2026-08-14 对齐）

> **唯一朗读入口**。主报锁定：公开 L2 **0.035012**（`spec_ref_r2` · **v10**）；Spectral idle **3.797 / 8.037 / 29.295 ms**。  
> SOL / proxy 仅旁注，禁止得分句。  
> 指针：[`CURRENT.md`](CURRENT.md) · README「评委 3 分钟路径」。

---

## A. 30 秒数字（一眼）

| 项 | 值 | 打开 |
|----|-----|------|
| 公开 NS64 相对 L2 | **0.035012** · `spec_ref_r2` · **v10** | [`summary.json`](../summary.json) · [`/workspace/评测报告_最新指标_2026-08-14_095200.md`](/workspace/评测报告_最新指标_2026-08-14_095200.md) |
| Spectral idle ms | **3.797 / 8.037 / 29.295**（64/128/256） | [`results.md`](../../results.md) §性能 |
| vs 官网 CPU 加速比 | ≈**19.5× / 11.1× / 10.0×**（旁注，≠ SOL） | [`SPECTRAL_SIX_AXIS_ORAL_2026-08-04.md`](SPECTRAL_SIX_AXIS_ORAL_2026-08-04.md) |
| Checkpoint | `fno_ns/checkpoints/fno_ns_public_demo.pt` | — |
| 正确性 worst rel | ≈**2.17e-7** ≪ 1e-4 | `summary.spectral_conv.rel_error` |

---

## B. 90 秒口播（压缩）

| 秒 | 动作 | 一句 |
|----|------|------|
| 0–15 | `brsmi` + 架构 | Biren 单卡；SUPA SpectralConv；FNO 复用必选算子 |
| 15–30 | accuracy 表 | worst≈2e-7，远低于 1e-4 |
| 30–50 | 三档 ms | mul 噪声级；墙在 C2R；formal **冻结**；讲 CPU 加速比 |
| 50–70 | 流场图 | 公开 NS64 L2=0.035012；Spectral-Refiner → v10 |
| 70–85 | 本包 C+D | 失败诚实 ABORT；扩展三命令可抽查 |
| 85–90 | Agent 索引 | 精品：26 / 30 / 32 / 33 / **35**（回滚与门禁） |

全文分镜：[`demo_storyboard_90s.md`](demo_storyboard_90s.md) · PPT：[`PPT答辩冻结稿_2026-08-04.md`](../PPT答辩冻结稿_2026-08-04.md)

---

## C. 失败三行（证明不是没试）

| 裁决 | 实验 | 数字 |
|------|------|------|
| **KEEP 主报** | `spec_ref_r2` | L2 **0.035012** · v10 |
| **ABORT / 已回滚** | freeze_r10 autochain 弱 promote | 0.035252 未破 gate(0.035202) → 回滚 v8 |
| **NO_SIGNAL** | A1 `hard_reweight` / soft_r10 / hybrid·modes20 | Δ=0 或远差主报；精度停 |

全表：[`experiment_matrix.md`](../experiment_matrix.md)

---

## D. 扩展三命令（可复现）

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission
./scripts/run_tests.sh backward   # grad ~6e-8
./scripts/run_tests.sh 3d         # 四角；≠3D FNO
./scripts/run_tests.sh irregular  # 9/9
```

细则：[`extension_showcase.md`](extension_showcase.md) · [`SPECTRAL_BONUS_AUDIT_CARD.md`](SPECTRAL_BONUS_AUDIT_CARD.md)  
**禁止**默认 `test_perf` 覆写 formal idle。

---

## E. Agent 五锚（约 15%）

| 记录 | 一句话 |
|------|--------|
| [26](../../development_log.md#agent-交互记录-26--stop-on-gate-快路径与-r5-promote2026-08-02) | stop-on-gate 快路径 |
| [30](../../development_log.md#agent-交互记录-30--operator_opt_loop-规范流程优化2026-08-02) | OPT Loop P0–P6 |
| [32](../../development_log.md#agent-交互记录-32--freeze_r9-promote-与-round10-启动2026-08-03) | freeze_r9 → v8 |
| [33](../../development_log.md#agent-交互记录-33--多-agent-交叉裁决与材料-p0-执行2026-08-03) | 多 Agent 交叉 + 材料 |
| [35](../../development_log.md#agent-交互记录-35--回滚-v8-收口与-promote-门禁2026-08-04) | 回滚 0.035252 + promote_guard + A1 NO_SIGNAL |

索引：[`development_log.md`](../../development_log.md) 文首场景对照表。

---

## 评委路径速查（与 README 对齐）

1. 本页（数字 + 口播 + 失败 + 命令 + Agent）  
2. [`results.md`](../../results.md) / 评测报告 §1（持平 0%）  
3. Demo 现行图：[`demo/media/README.md`](../../demo/media/README.md)（只看 08-02）  
4. 六轴口播：[`SPECTRAL_SIX_AXIS_ORAL_2026-08-04.md`](SPECTRAL_SIX_AXIS_ORAL_2026-08-04.md)  
