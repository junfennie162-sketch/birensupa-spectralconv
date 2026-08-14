# Skill: spectral-conv-dev

## 目标

在 BIREN / SUPA 上构建并验证 2D Spectral Convolution 扩展。

## 典型自然语言输入

- 「编译 SpectralConv 扩展并与 PyTorch reference 对比，误差要小于 1e-4」

## 步骤

1. `source` 环境：`submission/scripts/setup_env.sh`
2. 进入 `submission/spectral_conv/`，执行 `./build.sh`
3. 运行 `python3 test_accuracy.py`，记录相对误差到 `submission/results/`
4. 若失败：缩小 B/C/H/W 与 modes，固定种子，对比 CPU reference

## 输出

- 控制台 JSON / 日志
- `submission/results/run_logs/` 下的运行记录

## 边界

- 仅覆盖前向正确性与基础性能；反向传播为加分项
- 禁止与其它 GPU 任务并发
