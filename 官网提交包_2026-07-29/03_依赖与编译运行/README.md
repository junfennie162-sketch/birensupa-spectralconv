# 翻斗花园 · Spectral Convolution + FNO-NS

## 赛道与选题

- **赛事**：书生国智科探挑战赛
- **赛道**：模型与算子（壁仞飞翔杯）
- **必选**：Spectral Convolution（2D 频域卷积前向）
- **进阶**：FNO 求解二维 Navier-Stokes 涡度方程
- **队伍**：翻斗花园 · 中北大学
- **路线**：方式二 · SUPA + PyTorch Extension（正式路径：suFFT + `spectral_mul` fused）
- **提交对照**：[`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) · 必须项 [`skill.md`](skill.md)

## 环境

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
```

- SDK：`1.11.0.0.rc2` · 单卡 Biren106B · `device="supa"`（先 `import torch_br`）

## 目录

| 路径 | 说明 |
|------|------|
| `spectral_conv/` | 必选算子、构建、正确性/性能 |
| `fno_ns/` | 进阶 FNO、前向、可视化 |
| `scripts/` | 环境与一键脚本 |
| `results/` | 日志、图、`summary.json`、`phase_status.json` |
| `skills/` | Agent / Skill 说明 |
| `demo/` | SCP 简介与 media |
| `development_log.md` | Agent 交互记录（≥5 段） |
| `results.md` | 正确性与性能汇总 |

## 怎么测（先看这里）

前置条件（每次新开终端都要）：

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
# 确认没人占用 GPU：brsmi
```

| 你刚做完什么 | 跑哪个文件/命令 | 测的是什么 | 通过标准 |
|--------------|-----------------|------------|----------|
| 改了 `.su` / `.cpp` / `build.sh` | `cd spectral_conv && ./build.sh && python3 test_accuracy.py` | 必选算子算得对不对 | 相对误差 ≤ `1e-4` |
| 加强：suFFT 正确性 | `cd spectral_conv && python3 test_sufft_accuracy.py` | GPU FFT 路径是否对准 | ≤ `1e-4` |
| 加强：suFFT 性能 | `cd spectral_conv && python3 test_sufft_perf.py` | v1 vs suFFT 64/128/256 | 有对比表 |
| 改了算子或组装路径，关心速度 | `cd spectral_conv && python3 test_perf.py` | 64/128/256 耗时与显存 | 有数字写入 `results/` |
| 改了 FNO `model.py` / 训练逻辑 | `cd fno_ns && python3 test_forward.py` | 模型能前向 + 相对 L2 | 层数 ≥4，L2 有报告 |
| 改了可视化 | `cd fno_ns && python3 visualize.py` | 预测 vs 真值图 | `results/figures/` 有图 |
| 交卷前总回归 | `./scripts/run_tests.sh` 或 `./scripts/run_tests.sh all` | 主链路 + chain + batch16 串行 | 全程 exit 0 |
| 只测 suFFT 加强 | `./scripts/run_tests.sh sufft` | suFFT 正确性 | exit 0 |
| FNO chain 一致性 | `./scripts/run_tests.sh fno-chain` | CPU vs SUPA chain | rel≤1e-4 |
| FNO batch=16 性能 | `./scripts/run_tests.sh fno-batch16` | grid_points/s 等 | exit 0 |
| FNO 训练吞吐加分 | `./scripts/run_tests.sh fno-train-throughput` | 含 bwd/opt 的 grid_points/s | exit 0 |
| 自动调优快扫 | `./scripts/run_tests.sh tune` | `tune.py --quick` | 写出/验证决策 |
| 只同步展示材料 | `./scripts/run_demo.sh` | media / SCP 文案 | `demo/media` 有图和快照 |

结果写哪里：`results/summary.json`、`results.md`、`results/run_logs/`。  
一键脚本：`submission/scripts/run_tests.sh`（推荐）或旧的 `run_all_accuracy.sh`。

## 编译与运行

```bash
cd /workspace/ai4s-f/submission
./scripts/setup_env.sh

cd spectral_conv && ./build.sh
python3 test_accuracy.py
python3 test_perf.py

cd ../fno_ns
python3 test_forward.py
python3 visualize.py

cd ..
./scripts/run_demo.sh
./scripts/maintain_assets.sh status
```

## 关键实测摘要

| 项 | 结果 |
|----|------|
| SpectralConv 相对误差（v1） | ≈ 1.24e-7（≤ 1e-4） |
| **双角 auto 性能 64/128/256** | **5.302 / 13.670 / 52.480 ms**；显存 146.6 / 238.5 / 582.7 MB（2026-07-26，warmup=10 / iters=100，auto→fused） |
| spectral_mul 反向（相对参考 grad） | worst ≈ 6.3e-8 |
| FNO 层数 | 4 |
| FNO 相对 L2 | **0.008768**（自生成 NS-like v2；非公开 NS64；150 epoch / 14400 step，含 R7 侧车细调） |
| FNO batch=16 性能 | **1,366,849 grid_points/s**；333.70 samples/s；2.997 ms/sample；47.947 ms/batch；171.6 MB（纯 forward，warmup=10 / iters=50，chain rel=4.80e-5；R7 host-seeded D2D） |
| FNO 训练吞吐（加分） | **34,712 grid_points/s**（CPU/`use_supa=False`；含 fwd+loss+bwd+opt；B=8） |
| 可视化 | `results/figures/fno_ns_pred_vs_gt_2026-07-26.png` + `fno_ns_sample_strip_2026-07-26.png` |

细节见 `results.md` / `results/summary.json`。

## GPU / SUPA

| 环节 | 承担任务 |
|------|----------|
| SpectralConv kernel | 频域复数乘（SUPA Extension） |
| 正式前向 | suFFT R2C → SUPA mul（常驻）→ suFFT C2R |
| FNO Fourier Layer | 复用 fused 双角 API；训练用 CPU torch 可微路径 |
| FFT/iFFT（v1 对照） | CPU torch |

## Agent / Skills

- 日志：`development_log.md`
- Skills：`skills/spectral_conv_dev`、`skills/fno_experiment`
- SCP：`demo/scp_description.md` + `demo/media/`

## 已知限制

- 正式 SpectralConv 路径为 fused；v1 保留为对照与可微训练后备。
- FNO 正式数据为自生成 `generated_ns_like_v2`，不是公开 NS64 benchmark；完整披露见 `results/data_disclosure.md`。
- FNO device-resident chain 必须先通过 CPU/SUPA 一致性门禁，未通过的快速结果不作为正式性能。
- SpectralConv3d 为扩展前向（见 `spectral_conv/`）；未做完整 3D FNO。
- 禁止与 `ai4s-n` 并发占用同一 GPU。

## 文档

- 选手手册：`/workspace/赛题文档/算子与模型赛道选手手册.md`
- 官网详情：`/workspace/赛题文档/官网-赛道五-模型与算子详情页.md`
