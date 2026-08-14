# Auto-tune / SOL Skill 口径说明（2026-08-01）

- Formal Spectral 主表（冻结）：**3.811 / 8.054 / 29.560 ms** @64/128/256（idle）。
- `spectral_conv/tune_results.json` 为历史 sweep 决策痕迹；运行时 `use_sufft="auto"` 加载阈值，**不得**用其中旧 median_ms 覆盖 formal 主表。
- `skills/sol_gap_analysis.md` / `bench_sol_proxy.py`：**队内 proxy**，不是官方 SOL-ExecBench 得分句。
- fused 分段旁注：`spectral_fused_segments_2026-08-01.md`（C2R 墙）。
