# 提交对照清单（对照官方 · 2026-08-14 v10）

> **依据**：官网赛道五「统一要求 / 提交规范 / Agent 必须项」+《算子与模型赛道选手手册》最低提交物。  
> **主报**：公开 NS64 L2=**0.035012**（`spec_ref_r2` · **v10**）；Spectral 本次 idle 复测 **3.797 / 8.037 / 29.295 ms**。  
> **Agent 抽查页**：[`AGENT_OFFICIAL.md`](AGENT_OFFICIAL.md)（6 段 × 6 类场景）。  
> 状态：满足 / 部分满足 / 缺失。

---

## A. 官方最低提交物（必须）

| # | 官方要求 | 本仓库位置 | 状态 |
|---|----------|------------|:----:|
| A1 | 项目源码（SUPA / Extension 核心） | `spectral_conv/`（`.su`/`.cpp`/`build.sh`）；`fno_ns/` | 满足 |
| A2 | 依赖说明与**编译/运行命令** | `README.md`「交卷必跑」；`spectral_conv/build.sh`；`scripts/setup_env.sh` | 满足 |
| A3 | **正确性验证脚本与结果** | `test_accuracy.py`；[`正确性验证报告_2026-08-14.md`](results/run_logs/正确性验证报告_2026-08-14.md)；`results.md` | 满足 |
| A4 | **性能测试脚本与报告** | `test_perf.py`；[`性能检测报告_2026-08-14.md`](results/run_logs/性能检测报告_2026-08-14.md) | 满足 |
| A5 | BIREN 单卡运行日志/截图 | `demo/media/brsmi_snapshot.txt`（**2026-08-14 刷新**）；`official_recheck_2026-08-14.log` | 满足 |
| A6 | Agent/Skill 开发日志 ≥5 段、≥3 类场景 | **[`AGENT_OFFICIAL.md`](AGENT_OFFICIAL.md)**（6 段抽查）+ `development_log.md`（**43** 段） | 满足 |
| A7 | **`skill.md`（必须）** | 提交根 `skill.md` + `skills/*` | 满足 |
| A8 | 展示材料（可选建议） | `demo/media/` 流场图；`results/PPT答辩冻结稿_*.md` | 满足（建议项） |

---

## B. 必选 Spectral Convolution

| 项目 | 当前证据 | 状态 |
|------|----------|:----:|
| SUPA / Extension 核心 | `spectral_conv/*.su`、`*.cpp`、`build.sh`（2026-08-14 已重编） | 满足 |
| 2D forward、可配置 modes | fused 热路径 | 满足 |
| 正确性 rel≤1e-4 | 本次复测 worst **2.170e-7** | 满足 |
| 64/128/256 性能 | 本次 idle **3.797 / 8.037 / 29.295 ms** | 满足 |
| Backward / 3D / irregular | 历史 PASS（见 `results.md` §2.4） | 满足 |

---

## C. 进阶 FNO-NS

| 项目 | 当前证据 | 状态 |
|------|----------|:----:|
| ≥4 层 + 复用必选算子 | `fno_ns/model.py` | 满足 |
| 公开 NS64 L2 | **0.035012**（本次 clean 复评与 meta 一致 · `spec_ref_r2`） | 满足 |
| 可视化 | `demo/media/fno_ns_pred_vs_gt_*.png` + sample_strip | 满足 |
| 预训练 ckpt | `fno_ns/checkpoints/fno_ns_public_demo.pt` | 满足 |

---

## D. Agent（必须 · 约 15% · 不合格即整卷无效）

| 官方要求 | 落实 | 状态 |
|----------|------|:----:|
| ≥5 段有效交互 | 抽查页 **6** 段；全文 **43** 段 | 满足 |
| ≥3 类场景 | kernel / 瓶颈 / 超参 / 数据 / 可视化 / 平台 **6 类全覆盖** | 满足 |
| 固定字段 | 工具·场景·目标·摘要·建议·采纳·验证·未采纳 | 满足 |
| skill.md | 提交根存在并指向抽查页 | 满足 |

---

## E. 交卷命令（复制即跑）

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd spectral_conv && ./build.sh && python3 test_accuracy.py && python3 test_perf.py
```
