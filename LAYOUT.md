# 提交树目录分类（v10 · 2026-08-14）

> 官方必须项仍在根目录（`skill.md` / `development_log.md` / `results.md`），本文件只做**归类索引**。  
> 数字真源：[`results/summary.json`](results/summary.json) · 行动指针：[`results/run_logs/CURRENT.md`](results/run_logs/CURRENT.md) · **全过程 Case**：[`CASE_项目全过程_V0到V10.md`](CASE_项目全过程_V0到V10.md)

## 一眼

| 类 | 路径 | 留什么 |
|----|------|--------|
| 必选算子 | `spectral_conv/` | fused suFFT + SUPA mul 正式实现 |
| 算子对照（归档） | `results/archives/legacy_spectral_conv_combo/` | 历史 combo，非正式主路径 |
| 选修 FNO | `fno_ns/` | 公开 NS64 训练/评测/可视化 |
| 正式权重 | `fno_ns/checkpoints/` | v10 demo + v9 备份 + 工程 v2 |
| 公开数据 | `fno_ns/data/` | `navier_stokes_v1e-3_N1200_T20.pt` + 工程 v2 |
| 主报/日志 | `results/` | `summary.json`、figures、run_logs |
| 现行 run_logs | `results/run_logs/` | CURRENT / 08-14 复测 / 答辩入口 |
| 历史 run_logs | `results/run_logs/_history/` | 旧探针 JSON 与计划卡 |
| 现行提交包 | `results/archives/` | **只留 v10** tar + sha256 + README |
| 早期官网包 | `results/archives/legacy_official/` | 7/28、7/29 小包 |
| Demo / SCP | `demo/` | 展示图与简介 |
| Skills | `skills/` + 根 `skill.md` | 官方 skill + OPT 闭环 |
| **评委 Agent 抽查** | [`AGENT_OFFICIAL.md`](AGENT_OFFICIAL.md) | ≥5 段 × ≥3 类场景 |
| 全过程 Case | [`CASE_项目全过程_V0到V10.md`](CASE_项目全过程_V0到V10.md) | V0→V10 |

## 现行主报

| 项 | 值 |
|----|-----|
| FNO 公开 NS64 | **0.035012** · `spec_ref_r2` · **v10** |
| Spectral idle | **3.797 / 8.037 / 29.295 ms**（2026-08-14 复测；07-31 板 3.811/8.054/29.560） |
| 评测报告 | `/workspace/评测报告_最新指标_2026-08-14_095200.md` |
| 提交包 | `results/archives/fandougarden_submit_20260811_103945.tar.gz` |

## 清理纪律（2026-08-14 已执行）

已删：旧 tar 解压树、侧车 ckpt、`__pycache__`、空 `logs/`、ppt-master 缓存、工作区根重复 Case/旧评测报告。  
已迁：旧 `run_logs` → `_history/`；combo → `archives/legacy_spectral_conv_combo/`；过期 demo PNG → `demo/media/archive_history/`。
