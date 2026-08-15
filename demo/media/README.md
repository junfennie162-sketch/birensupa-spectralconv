# Demo media

Two flow-field figures for judges. Both are official public NS64 test samples with the formal checkpoint — not sketches, not synthetic data.

Formal score is in `results.md`: **mean relative L2 over 128 test samples = 0.035012**. A single-sample L2 on a figure is not that mean.

How they were made: `fno_ns/render_official_demo.py` reads the official `.pt` + `fno_ns_public_demo.pt`, then `visualize.py`.

---

## Figure 1 · `01_typical_sample_pred_vs_gt.png`

Same test sample, left to right: last input frame → next-frame ground truth → prediction.

| Panel | Meaning |
|-------|---------|
| Left | Last of the 10 input vorticity frames |
| Middle | Ground truth frame 11 |
| Right | Model prediction of frame 11 |

Red / blue are vorticity sign. Ground truth and prediction share one colormap.

**Typical** means: among 128 test samples, the one whose relative L2 is **closest to 0.035012**. Not the best, not the first.

---

## Figure 2 · `02_best_typical_worst.png`

Three rows: best / typical / worst single-sample L2 on the official test split.

Each row, left to right: ground truth · prediction · **absolute** error `|pred − gt|` (bright = larger miss). That column is not a relative-error heatmap.

---

## Other files (not figures)

| File | What it is |
|------|------------|
| `brsmi_snapshot.txt` | Single-card GPU snapshot |
| `official_recheck_2026-08-14.log` | Raw recheck terminal log |
| `metrics_snapshot.md` | Metric excerpt |
