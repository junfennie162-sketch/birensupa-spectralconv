# SOL 差距分析 Skill（本地 proxy，非官方 SOL）

依据：`/workspace/赛题文档/赛道验收与提交清单.md` §3.3（SOL-Score / SOL-ExecBench **思路**）。

## 用途

用可复现脚本把「墙钟 / 显存 / 粗算带宽与算力占用」写成一页分析，回答：

1. SpectralConv formal 路径瓶颈更像算力、带宽还是 launch/FFT？
2. FNO batch16 相对 Spectral 多付了什么税（正确性物化、层间 IN/GELU）？
3. 下一轮优化该打哪一段？

**明确声明**：本 Skill 产出的数字是本地 proxy，**不是**官方硬件理论 SOL，也不是 SOL-ExecBench 榜单分。

## 怎么跑

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission/spectral_conv
python3 bench_sol_proxy.py
```

产物：

- `results/sol_proxy_r7.json`
- `results/run_logs/sol_proxy_r7_YYYY-MM-DD.md`

## 口径

| 量 | 定义 |
|---|---|
| GB/s proxy | `(bytes_rw_proxy / median_s) / 1e9`，bytes 含输入/双权重/输出粗估 |
| TFLOPS proxy | `(flops_proxy / median_s) / 1e12`，flops 按双角复乘累加粗估 |
| sol_*_vs_peak | 相对本机锚点（`PEAK_TFLOPS_FP32_PROXY` / `PEAK_HBM_TB_S_PROXY`）的占用比 |
| bottleneck_hint | 比较带宽占用比与算力占用比的粗标签 |

锚点是保守分析占位，换机器要改脚本常量并在报告重写 disclaimer。

## R7 观察（2026-07-25）

- Spectral formal（CPU 起源）不走 R7 safe-buffer；主耗在 suFFT + `spectral_mul`。
- FNO chain 每层仍需 host-seeded SUPA buffer 的 D2D `copy_`（suFFT 指针来源正确性）；相对 R6 pinned D2H+H2D 已降税。
- `spectral_mul` float2/unroll 相对旧 kernel 约噪声级（~1%）；FNO 吞吐跃迁主要来自物化路径，不是 mul 微优。

## 与其它 Skills 的关系

- 链路线索：`spectral_chain_optimization.md`（R4–R7）
- FNO 评测门禁：`fno_eval_protocol.md`
- 自动路径选择：`tune.py` + `tune_results.json`
