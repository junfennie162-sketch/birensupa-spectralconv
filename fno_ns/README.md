# FNO · Navier-Stokes（进阶 C）

基于 `../spectral_conv_combo` 的 **SUPA SpectralConv Extension**（提升版；原版目录仍保留）。

## 验收

- Fourier Layer ≥ 4
- 数据说明见 `data/README.md`（默认合成涡度；可换公开 64×64 NS）
- 单卡前向：`test_forward.py`
- 可视化：`visualize.py` → `../results/figures/fno_ns_pred_vs_gt.png`
- 可选短训：`train_or_infer.py`（合成数据，加分演示；`use_supa=False` 可微路径）

## 运行

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
# 需先编译提升版 extension
cd /workspace/ai4s-n/submission/spectral_conv_combo && ./build.sh
cd /workspace/ai4s-n/submission/fno_ns
python3 test_forward.py
python3 visualize.py
# optional
python3 train_or_infer.py --epochs 2 --resolution 32
```

## 状态

混合路线：必选算子 SUPA mul（combo）+ Python FNO 组装。
