# 评委 3 分钟抽查卡 · Spectral 扩展六轴

主报锚：公开 L2 **0.035012**（`spec_ref_r2` · v10）；Spectral idle **3.797 / 8.037 / 29.295 ms**。  
（历史 v7：`sched_samp_r5` 0.035725，仅版本链。）

| 轴 | 一句话主张 | 一条命令 | 文件锚 |
|----|------------|----------|--------|
| 正确性 | worst rel ≈2.17e-7 ≪ 1e-4 | `cd spectral_conv && python3 test_accuracy.py` | `summary.spectral_conv.rel_error` |
| 性能/多分辨率 | formal 冻结；vs CPU ≈19.5×/11.1×/10.0×；墙在 C2R | 读旁注即可（勿重写 formal） | `spectral_multires_story_2026-08-02.md` + `spectral_fused_segments_2026-08-01.md` |
| 显存 | formal 225/253/353 MB + packed/cache/scatter 手段 | 同上 idle 表 | `spectral_memory_story_2026-08-02.md` + `spectral_memory_footprint_2026-08-02.md` |
| 并行度 | dual_scatter/packed/unroll；mul 已噪声 | — | `spectral_parallelism_card_2026-08-02.md` |
| 融合诚实 | 设备常驻+稀疏 scatter；非真 FFT⊗mul 融合 | — | `spectral_fusion_honesty_card.md` |
| Backward | grad ~6e-8；长训仍 CPU | `python3 test_backward.py` | `supa_diff_loop_story.md` |
| 3D / irregular | 3D **四角** + irregular PASS；≠3D FNO | `./scripts/run_tests.sh 3d` / `irregular` | `extension_showcase.md` |
| Agent | ≥3 类场景可跳转 | `python3 skills/operator_opt_loop/run_loop.py --dry-run` | `development_log.md` + `demo_storyboard_90s.md` |

**禁区**：SOL/proxy 得分句；v2 L2=0.005144 冒充公开分；解冻 formal ms。
