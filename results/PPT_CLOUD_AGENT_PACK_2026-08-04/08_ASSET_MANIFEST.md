# 资产清单与占位规则

> 本 ZIP **仅含 Markdown**。云端 Agent 若拿不到原图，必须用**标注占位框**，禁止用无关图片冒充实验结果。

---

## 1. 现行可视化（优先使用）

| 角色 | 相对提交根路径 | PPT 用途 |
|------|----------------|----------|
| Pred / GT / error 主图 | `demo/media/fno_ns_pred_vs_gt_2026-08-02.png` | 可视化主页 |
| sample strip | `demo/media/fno_ns_sample_strip_2026-08-02.png` | 可视化辅图 |
| brsmi 快照 | `demo/media/brsmi_snapshot.txt` | 可摘 3–5 行到环境页 |
| 指标快照 | `demo/media/metrics_snapshot.md` | 可选 |

**旧图勿用主位：** `results/figures/` 中 07-21～08-01 等历史 pred 图仅归档。

---

## 2. 旁注图（可选，不得当主报）

| 文件 | 含义 | 限制 |
|------|------|------|
| `demo/media/error_decomp_e1_tf_ar.png` | Error Autopsy 分解 | 旁注；不编 v9 |
| `demo/media/spectrum_best_median_worst.png` | 频谱分析 | 旁注 |
| `demo/media/spectrum_best_median_worst_heatmaps.png` | 频谱热力 | 旁注 |
| `demo/media/protocol_vs_fno_paper_0128.png` | 协议对照 | 旁注 |

---

## 3. 占位框写法（无图时强制）

在幻灯片中画矩形，内文示例：

```
【图位：公开 NS64 Pred / GT / error】
文件：fno_ns_pred_vs_gt_2026-08-02.png
数据：public_ns64 · L2=0.035302
（请插入提交包 demo/media 现行图）
```

---

## 4. 文档真源（人类侧，不在本 ZIP）

若委托方后续补文件给云端 Agent，优先追加：

1. 上述 PNG  
2. （可选）`JUDGE_3MIN_PACK_2026-08-04.md` 原文  
3. （可选）`PPT答辩冻结稿_2026-08-04.md` 原文  

本包已吸收其信息要点，**不依赖**再读原文件即可出稿。

---

## 5. 图表应由 PPT 原生绘制（推荐）

下列内容**不要**截图糊上去，请用 PPT 表/柱状图：

- Spectral 三档 ms 柱图  
- 官方权重饼/条  
- v1→v8 L2 折线（数字见 `04_FROZEN_METRICS.md`）  
- KEEP / ABORT / NO_SIGNAL 三行表  
