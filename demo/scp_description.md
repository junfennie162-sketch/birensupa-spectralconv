# SCP 广场简介 · 翻斗花园

**SpectralConv + FNO on BIREN（壁仞飞翔杯 · 模型与算子）**

## 一句话

在壁仞 GPU 上用 SUPA / PyTorch Extension 实现 FNO 核心 Spectral Convolution（fused 正式路径），并搭建 ≥4 层 FNO 做二维涡度场前向预测；全程 Cursor Agent 人机协同，结果可复现。

## 创新点卡片

| 卡片 | 要点 |
|------|------|
| 双角 fused | suFFT + SUPA mul 常驻；相对官网 CPU 参考约 **19.5× / 11.1× / 10.0×**（64/128/256）；墙在 C2R |
| residual+hf | 公开 NS 残差头 + 频域加权 + 周期增广 → 轨迹至 sq3b 0.037520 |
| multistep TF+soft | 多步 teacher-forcing + 轻量谱/能量 soft → 0.036576 |
| **sched → freeze → Spec-Refiner** | 冻 spectral 后只训 spectral + H⁻¹ → 主报 L2 **0.035012** |
| 版本链（30 秒） | v9 0.035115 → v10 0.035012（只比上一版 **+0.29%**；自 v1 累计约 +16.3%） |
| 口径闸 | 公开 1000/128 与自建 v2 分栏；失败实验见 `results/experiment_matrix.md` |

## 做了什么

| 模块 | 说明 | 实测（本机 Biren106B，2026-08-02） |
|------|------|-------------------------------------|
| 必选 SpectralConv | SUPA fused + trunc/pack/scatter（P8b） | 正确性 worst ≈ **2.17e-7**；idle 64/128/256 ≈ **3.797 / 8.037 / 29.295 ms** |
| 扩展 | Backward + 3D + irregular + auto-tune + SOL proxy | grad ≈ **6e-8**；3D 四角 ≈**1.19e-7**；tune/SOL Skill 可复现（SOL 为队内 proxy） |
| 进阶 FNO-NS | 4 层 Fourier Layer，复用必选算子 | **公开 NS64** 1000/128 相对 L2 ≈ **0.035012**（`spec_ref_r2`；`fno_ns_public_demo.pt`） |
| FNO chain | device-resident + host-seeded D2D fallback | vs CPU rel ≈ **4.758e-5** PASS（阈值 1e-4） |
| FNO batch=16 | 官方吞吐协议（公开 NS64 + public_demo） | ≈ **1.60M grid_points/s**（纯 forward；peak≈202 MB） |
| 可视化 | Pred/GT 共用色标 + abs/rel 误差 + 多样本条带 | `demo/media/fno_ns_pred_vs_gt_2026-08-02.png`（评委入口见 `demo/media/README.md`） |

旁注：自建 NS-like v2 continue3 L2≈0.005144 仅为工程对照，**不是**公开集成绩。

## 怎么跑

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd submission
./scripts/run_tests.sh fno-chain
./scripts/run_tests.sh fno-batch16
cd fno_ns && python3 test_forward.py && python3 visualize.py

# 扩展闭环（bwd / 3D / irregular）
cd ../spectral_conv
python3 test_backward.py
python3 test_3d_accuracy.py
python3 test_irregular_shapes.py

# Agent 抽查 dry-run（零 GPU）
cd .. && python3 skills/operator_opt_loop/run_loop.py --dry-run
```

扩展说明：`results/run_logs/extension_showcase.md` · 抽查卡：`results/run_logs/SPECTRAL_BONUS_AUDIT_CARD.md`  
性能故事：`results/run_logs/SPECTRAL_SIX_AXIS_ORAL_2026-08-04.md` · 评委一页包：`JUDGE_3MIN_PACK_2026-08-04.md`

演示材料见 `demo/media/README.md`（现行仅 08-02 主图/strip + metrics + brsmi；旧图在 `archive_history/`）。

## 队伍

翻斗花园 · 中北大学 · 赛道五（模型与算子）

## 清单

- [x] 单卡运行快照（`demo/media/brsmi_snapshot.txt`）
- [x] 流场对比图（`demo/media/fno_ns_pred_vs_gt_2026-08-02.png`）
- [x] 多样本条带（`demo/media/fno_ns_sample_strip_2026-08-02.png`）
- [x] 指标摘要（`demo/media/metrics_snapshot.md`）
- [ ] 可选：B 站演示（话题 #书生国智科探挑战赛）
