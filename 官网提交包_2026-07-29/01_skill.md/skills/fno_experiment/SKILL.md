# Skill: fno-experiment

## 目标

基于已验证的 SpectralConv 扩展，组织 FNO-NS 单次前向与结果可视化。

## 典型自然语言输入

- 「用 64×64 Navier-Stokes 样例跑 FNO 前向，并画出预测与真值对比」

## 步骤

1. 确认 `spectral_conv` 扩展已构建且 accuracy 通过
2. 按 `fno_ns/data/README.md` 准备数据
3. 运行 `python3 test_forward.py`，记录相对 L2
4. 运行 `python3 visualize.py`，图写入 `submission/results/figures/`

## 输出

- 指标摘要、流场对比图、日志路径

## 边界

- 海选以可复现前向为主；长时间训练非必须
- Fourier Layer 不少于 4 层
