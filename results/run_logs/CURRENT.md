# CURRENT · pointers (2026-08-15)

| 项 | 现行值 |
|----|--------|
| 公开 NS64 L2 | **0.035012** |
| Spectral 64 / 128 / 256 | **0.599 / 1.405 / 5.099 ms**（`pinned_src_r1` · v13 · 2026-08-16） |
| 评测报告 | [`评测报告_最新指标_2026-08-16_094942.md`](../评测报告_最新指标_2026-08-16_094942.md) |
| Agent 抽查 | [`AGENT_OFFICIAL.md`](../../AGENT_OFFICIAL.md) |
| 正确性报告 | [`正确性验证报告_2026-08-14.md`](正确性验证报告_2026-08-14.md) |
| 性能报告 | [`性能检测报告_2026-08-14.md`](性能检测报告_2026-08-14.md) |
| 历史日志 | [`_history/`](_history/)（答辩轨迹，勿当现行入口） |
| 同赛道对比（旁注） | [`同赛道优化对比_2026-08-15.md`](同赛道优化对比_2026-08-15.md)（不进 v 号 / 不进 `skill.md`） |
| 裁剪 DFT 探针计划 | [`PRUNED_DFT_PROBE_PLAN_2026-08-15.md`](PRUNED_DFT_PROBE_PLAN_2026-08-15.md) |
| 裁剪 DFT 样本结果 | [`pruned_dft_probe_2026-08-15.md`](pruned_dft_probe_2026-08-15.md)（No-Go，未改热路径） |
| 最终结合版（只看这个） | [`FINAL_COMBO_RESULT.md`](FINAL_COMBO_RESULT.md) |
| trunc_off 修复复测 | [`trunc_off_fix_2026-08-15.json`](trunc_off_fix_2026-08-15.json)（PASS；默认仍 auto） |
| SUPA 裁剪探针 | [`pruned_supa_probe_2026-08-15.md`](pruned_supa_probe_2026-08-15.md)（默认 pruned；非正式 64 **1.865** / 128 **5.071** / 256 **18.16 ms**；未 promote idle） |
| pruned 续测（官网协议） | [`pruned_continue_test_2026-08-15.md`](pruned_continue_test_2026-08-15.md)（warmup=10/iters=100：默认 **1.069 / 2.331 / 7.957**；vs suFFT **3.672 / 7.953 / 29.245**；未 promote） |
| 几何 A/B（packed/融合逆） | [`pruned_geo_probe_2026-08-15.md`](pruned_geo_probe_2026-08-15.md)（特化 DFT KEEP；混合基 irfft + **128/256 irfft float4 加载** + ifft_h 64/128/256 32×8 + 前向 rfft/fft_h smem；64 前向 `16×4`；256 融合逆关） |
| 双路径（算子缓存 + FNO 回拷） | [`dual_path_probe_2026-08-15.md`](dual_path_probe_2026-08-15.md)（CPU 入 **0.961 / 2.207 / 7.870**；跳过回拷 FNO No-Go；FNO 旁注约 **5.3M** gps；未 promote） |
| 一键复现 | [`scripts/reproduce.sh`](../../scripts/reproduce.sh)（编译 + `probe_pruned_continue.py`；**不写**正式 idle） |
| 本轮 No-Go（旁注） | [`buffer_kernel_nogo_2026-08-15.md`](buffer_kernel_nogo_2026-08-15.md)（FNO ping-pong / pinned H2D / irfft dual-n1 f2 / ifft float2 store；热路径未改） |
| 256 smem 融合逆（2026-08-16） | [`fused_b256_devicein_probe_2026-08-16.md`](fused_b256_devicein_probe_2026-08-16.md)（16 行/256 线程 KEEP 进热路径；主表 ms 未改） |
| 64 smem 融合逆（2026-08-16） | [`fused64_keep_probe_2026-08-16.md`](fused64_keep_probe_2026-08-16.md)（KEEP 进热路径；非正式 64 约 **0.79–0.84 ms**） |
| 64 smem 融合前向（2026-08-16） | [`fused_fwd64_probe_2026-08-16.md`](fused_fwd64_probe_2026-08-16.md)（KEEP 默认 `SPECTRAL_FUSED_FWD64`；非正式 64 约 **0.75 ms**） |
| 128 smem 融合前向（2026-08-16） | [`fused_fwd128_probe_2026-08-16.md`](fused_fwd128_probe_2026-08-16.md)（KEEP 默认 `SPECTRAL_FUSED_FWD128`；非正式 128 约 **2.10–2.14 ms**） |
| C++ 一站式 / 256 前向上限（2026-08-16） | [`fused_e2e_smemcap_probe_2026-08-16.md`](fused_e2e_smemcap_probe_2026-08-16.md)（一站式默认关；256 整平面前向硬 No-Go） |
| 256 融合逆 4-n1（2026-08-16） | [`fused256_n4_probe_2026-08-16.md`](fused256_n4_probe_2026-08-16.md)（KEEP 默认 `SPECTRAL_FUSED_MIXED256_N4`；非正式 256 约 **6.82–6.88 ms**） |
| 128 融合逆 4-n1 / 64 n8（2026-08-16） | [`fused_inv_nmore_probe_2026-08-16.md`](fused_inv_nmore_probe_2026-08-16.md)（128 n4 KEEP 默认开；64 n8 默认关；256 n4 rest `/8`） |
| 256 流式融合前向（2026-08-16） | [`fused_fwd256_probe_2026-08-16.md`](fused_fwd256_probe_2026-08-16.md)（**No-Go**；dispatch 已撤） |
| 128 融合逆 8-n1（2026-08-16） | [`fused128_n8_probe_2026-08-16.md`](fused128_n8_probe_2026-08-16.md)（正确；e2e 不稳，默认关） |
| 256 rFFT 16 行（2026-08-16） | [`rfft256_n16_probe_2026-08-16.md`](rfft256_n16_probe_2026-08-16.md)（同 TU 污染 KEEP；独立 TU 正确但慢 42%；默认关） |
| fused KEEP 写入主表（2026-08-16） | [`promote_fused_keep_2026-08-16.md`](promote_fused_keep_2026-08-16.md)（**0.762 / 1.981 / 7.324** · v11） |
| 脏树整理（2026-08-16） | [`dirty_tree_cleanup_2026-08-16.md`](dirty_tree_cleanup_2026-08-16.md)（嵌套仓 `git reset`；不 commit） |
| 双流拆 batch（2026-08-16） | [`pipe_b_probe_2026-08-16.md`](pipe_b_probe_2026-08-16.md)（128/256 KEEP 默认开；64 关） |
| pipe KEEP 写入主表（2026-08-16） | [`promote_pipe_b_2026-08-16.md`](promote_pipe_b_2026-08-16.md)（**0.764 / 1.827 / 6.504** · v12） |
| pinned KEEP 写入主表（2026-08-16） | [`promote_pinned_src_2026-08-16.md`](promote_pinned_src_2026-08-16.md)（**0.599 / 1.405 / 5.099** · v13） |
| 四流拆 batch（2026-08-16） | [`pipe_n4_probe_2026-08-16.md`](pipe_n4_probe_2026-08-16.md)（256 KEEP 默认 N=4；128 关；主表未改） |
| 256 fft_h grid.y=4（2026-08-16） | [`ffth256_ny4_probe_2026-08-16.md`](ffth256_ny4_probe_2026-08-16.md)（**No-Go**；默认关） |
| 256 中间谱 MH（2026-08-16） | [`rowfreq_mh_probe_2026-08-16.md`](rowfreq_mh_probe_2026-08-16.md)（**No-Go**；默认关） |
| 摊还 pinned 输入（2026-08-16） | [`pinned_src_probe_2026-08-16.md`](pinned_src_probe_2026-08-16.md)（KEEP 默认开；已 promote v13） |
| pinned 后重调拆流（2026-08-16） | [`pipe_retune_probe_2026-08-16.md`](pipe_retune_probe_2026-08-16.md)（默认关 pipe；nopipe 正式 **1.405 / 5.099**） |
