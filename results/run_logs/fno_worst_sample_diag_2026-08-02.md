> **HISTORICAL / 旁注**：下文样本条曾绑 `sched_samp_r2`；现行主报见 `summary.json`（`freeze_r9` L2=**0.035302** · v8）。

# FNO 公开集 · worst 样本诊断旁注（2026-08-02）

> 主图绑定 `fno_ns_public_demo.pt`（`sched_samp_r2`，L2=**0.036092**）。本页不改主报。

## 样本条带摘要（visualize 2026-08-02，batch 内 best/median/worst）

| 角色 | index（batch 内） | sample rel-L2 |
|------|------------------:|--------------:|
| best | 10 | 0.022810 |
| median | 9 | 0.029638 |
| worst | 13 | **0.059400** |
| 主图 sample0（t=10） | 0 | 0.052659 |

全 test 集极端（128 条）：best #86≈0.0200；worst #54≈0.0916。

图：`results/figures/fno_ns_pred_vs_gt_2026-08-02.png` · `fno_ns_sample_strip_2026-08-02.png`

## 解读

- 主报是 **128 条平均相对 L2=0.035725**（现行 r5；下文样本条曾绑 r2），不是 worst 单点。  
- worst 尾部仍在；scheduled sampling 主要压平均误差与 exposure bias。  
- 禁止把 worst / TTA 写成正式成绩。
