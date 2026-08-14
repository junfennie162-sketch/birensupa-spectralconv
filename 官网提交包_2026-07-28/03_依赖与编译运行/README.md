# 翻斗花园 · Spectral Convolution + FNO-NS

> 最终提交包（合并 `ai4s-f` 主线 + `ai4s-n` 提升版）。提交对照见 [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md)，提交摘要见 [`SUBMISSION_MANIFEST.md`](SUBMISSION_MANIFEST.md)，必须项 [`skill.md`](skill.md)。

## 赛道与选题

- **赛事**：书生国智科探挑战赛
- **赛道**：模型与算子（壁仞飞翔杯）
- **必选**：Spectral Convolution（2D 频域卷积前向）
- **进阶**：FNO 求解二维 Navier-Stokes 涡度方程
- **队伍**：翻斗花园 · 中北大学
- **路线**：方式二 · SUPA + PyTorch Extension（正式路径：suFFT + `spectral_mul` fused + R5 dual_out + auto-tuning）

## 环境

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
```

- SDK：`1.11.0.0.rc2` · 单卡 Biren106B · `device="supa"`（先 `import torch_br`）

## 目录

| 路径 | 说明 |
|------|------|
| `spectral_conv_combo/` | **必选算子主目录**（5-case + irregular + auto-tune，含 R5 dual_out） |
| `spectral_conv/` | 必选算子备份（4-case baseline） |
| `fno_ns/` | 进阶 FNO、前向、可视化、分层 profile |
| `scripts/` | 环境与一键脚本（`run_tests.sh` / `maintain_assets.sh` / `setup_env.sh`） |
| `results/` | 日志、图、`summary.json`、`phase_status.json` |
| `skills/` | Agent / Skill 说明（含 `spectral_chain_optimization.md` R3/R4/R5 lessons） |
| `demo/` | SCP 简介与 media |
| `development_log.md` | Agent 交互记录（17 段，覆盖 ≥3 类场景） |
| `results.md` | 正确性与性能汇总 |
| `SUBMISSION_MANIFEST.md` | 自动生成的提交摘要（含实测数据） |

## 怎么测（先看这里）

前置条件（每次新开终端都要）：

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
# 确认没人占用 GPU：brsmi
```

| 你刚做完什么 | 跑哪个文件/命令 | 测的是什么 | 通过标准 |
|--------------|-----------------|------------|----------|
| 改了 `.su` / `.cpp` / `build.sh` | `cd spectral_conv_combo && ./build.sh && python3 test_accuracy.py` | 必选算子算得对不对 | 5-case 全过，worst rel ≤ 1e-4 |
| 改 perf 路径 | `cd spectral_conv_combo && python3 test_perf.py` | 64/128/256 耗时与显存 | 有数字写入 `results/` |
| 跑了非标准 shape | `cd spectral_conv_combo && python3 test_irregular_shapes.py` | 9-shape 鲁棒性 | 9/9 OK |
| 反向 / 3D | `python3 test_backward.py` / `python3 test_3d_accuracy.py` | backward / 3D extension | rel ≤ 1e-4 |
| Auto-tune | `cd spectral_conv_combo && python3 tune.py --quick` | 25 秒出决议 | `tune_results.json` 写入 |
| FNO 模型 | `cd fno_ns && python3 test_forward.py` | FNO 前向 + L2 | 层数 ≥4，L2 有报告 |
| 可视化 | `cd fno_ns && python3 visualize.py` | 预测 vs 真值图 | `results/figures/` 有图 |
| 交卷前总回归 | `./scripts/run_tests.sh` 或 `./scripts/run_tests.sh all` | 上面主链路串行 | 全程 exit 0 |

结果写哪里：`results/summary.json`、`results.md`、`results/run_logs/`。  
一键脚本：`submission/scripts/run_tests.sh`（推荐）。

## 编译与运行

```bash
cd /workspace/ai4s/submission
./scripts/setup_env.sh

cd spectral_conv_combo && ./build.sh
python3 test_accuracy.py        # 5-case correctness
python3 test_perf.py            # 64/128/256 perf
python3 test_irregular_shapes.py
python3 tune.py --quick         # auto-tune 决议

cd ../fno_ns
python3 test_forward.py
python3 visualize.py

cd ..
./scripts/run_demo.sh
./scripts/maintain_assets.sh status
```

## 关键实测摘要（实测，2026-07-24 21:00 CST）

| 项 | 结果 |
|----|------|
| SpectralConv 相对误差（5-case） | **worst 2.83e-7**（≤ 1e-4） |
| SpectralConv 相对误差（irregular 9-shape） | worst 3.92e-7 |
| spectral_mul 反向（相对参考 grad） | worst ≈ 6.3e-8 |
| **双角 auto 性能 64/128/256** | **5.32 / 13.69 / 52.64 ms** |
| SpectralConv 3D（2 case 8³/16³） | worst rel 1.07e-7 |
| FNO 层数 | 4 |
| FNO 相对 L2 | **0.009516**（NS-like v2；约 110 epoch / n_train=768） |
| FNO supa_chain @ 64 | 16.099 ms median / 15.998 ms min |
| 可视化 | `results/figures/fno_ns_pred_vs_gt_2026-07-23.png` |

细节见 `results.md` / `results/summary.json`。

## GPU / SUPA

| 环节 | 承担任务 |
|------|----------|
| SpectralConv kernel | 频域复数乘（SUPA Extension `.su` + PyBind） |
| 正式前向 | suFFT R2C → SUPA mul（常驻）→ suFFT C2R |
| R5 dual_out | 单次 pybind 启动双角 kernel |
| FNO Fourier Layer | 复用 fused 双角 API；训练用 CPU torch 可微路径 |
| FFT/iFFT（v1 对照） | CPU torch（避开 `torch.fft@supa` 已知问题） |

## Agent / Skills

- 日志：`development_log.md`（17 段交互记录，覆盖 kernel/性能/超参/数据可视化等 ≥3 类场景）
- Skills：`skills/spectral_chain_optimization.md`（R3/R4/R5 lessons 12.5 KB）、`skills/spectral_conv_dev/`、`skills/fno_experiment/`
- Auto-Tuning：`spectral_conv_combo/tune.py`（`--quick` 25 秒出决议；写入 `tune_results.json` 与全局表 `_AUTO_TUNE_TABLE`）
- SCP：`demo/scp_description.md` + `demo/media/`（4 张展示图）

## 已知限制

- 正式路径为 fused（min(H,W) ≥ 64 自动走 suFFT）；v1 保留为对照与可微训练后备。
- FNO 数据为离线 NS-like 64×64；公开 NS HDF5 有网后可替换再训。
- SpectralConv3d 为扩展前向（见 `spectral_conv_combo/`）；未做完整 3D FNO。
- `sufftBuildPlan2d` 在 SDK 头声明但 `libsufft.so` 不导出（已记录）。如要进一步优化 rfft/irfft fused kernel，需等 SDK 升级。
- 禁止与 `ai4s-n` 并发占用同一 GPU。

## 文档

- 选手手册：`/workspace/赛题文档/算子与模型赛道选手手册.md`
- 官网详情：`/workspace/赛题文档/官网-赛道五-模型与算子详情页.md`
- 队友交接总结：`/workspace/ai4s-n/测试/提升版/results/run_logs/交接总结_2026-07-25.md`